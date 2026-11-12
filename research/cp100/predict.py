"""C1_PREDICT: exact raw1280 1-NN maps for the approved CP100 banks.

Consumes C0_SELECT read-only and never opens ground truth.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(os.environ.get("MVTEC_GRID_ROOT", "/workspace/mvtec_grid_v1"))
SELECT = ROOT / "outputs/closest_pair_pruning_100k_8cat_v1/C0_SELECT"
OUT = ROOT / "outputs/closest_pair_pruning_100k_8cat_v1/C1_PREDICT"
INPUTS = ROOT / "INPUTS.json"
CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
LAYERS = (7, 15, 23, 31)
DIM = 1280
TARGET = 100_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def resolve_rows(category: str) -> list[dict]:
    source_rows = json.loads(INPUTS.read_text(encoding="utf-8"))["images"][category]
    by_source = {row["path"]: row for row in source_rows}
    rows: list[dict] = []
    if category == "can":
        for split in ("normal", "public"):
            base = ROOT / "can_cache" / split
            manifest = json.loads((base / "MANIFEST.json").read_text(encoding="utf-8"))
            for row in manifest["records"]:
                key = "mvtec_ad_2/can/" + row["relative_path"]
                rows.append(dict(by_source[key], artifact=str(base / row["artifact"])))
    else:
        marker = json.loads((ROOT / "outputs/cache" / category / "GENERATION_COMPLETE.json").read_text(encoding="utf-8"))
        for row in marker["records"]:
            rows.append(dict(by_source[row["source"]], artifact=str(ROOT / row["path"])))
    return sorted(rows, key=lambda row: row["path"])


def verify_select(category: str, layer: int) -> tuple[np.ndarray, str]:
    unit = SELECT / category / f"L{layer}"
    marker_path = unit / "C0_SELECT_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker["status"] != "complete" or marker["selected_count"] != TARGET or marker["gt_opened"]:
        raise RuntimeError(f"invalid C0 marker: {unit}")
    for item in marker["artifacts"]:
        path = unit / item["name"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"C0 artifact mismatch: {path}")
    ids_path = unit / "selected_ids.npy"
    ids = np.load(ids_path).astype(np.int64)
    if ids.shape != (TARGET,) or len(np.unique(ids)) != TARGET:
        raise RuntimeError(f"selected ID contract failed: {unit}")
    return ids, sha256(ids_path)


def reconstruct_bank(rows: list[dict], layer: int, ids: np.ndarray) -> np.ndarray:
    positions = {int(value): index for index, value in enumerate(ids)}
    ordered = np.sort(ids)
    bank = np.empty((TARGET, DIM), np.float16)
    offset = 0
    filled = 0
    layer_index = LAYERS.index(layer)
    for row in rows:
        if row["split"] != "fit":
            continue
        raw = np.load(row["artifact"], mmap_mode="r")[layer_index].reshape(-1, DIM)
        lo, hi = offset, offset + len(raw)
        local = ordered[(ordered >= lo) & (ordered < hi)] - lo
        if len(local):
            target = np.asarray([positions[int(lo + value)] for value in local], np.int64)
            bank[target] = raw[local]
            filled += len(local)
        offset = hi
    if filled != TARGET:
        raise RuntimeError(f"bank reconstruction incomplete: {filled}/{TARGET}")
    return bank


def batches(rows: list[dict], size: int):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def load_query_batch(batch: list[dict], layer_index: int,
                     expected_shape: tuple[int, int] | None) -> tuple[np.ndarray, tuple[int, int], list[int]]:
    arrays: list[np.ndarray] = []
    lengths: list[int] = []
    shape = expected_shape
    for row in batch:
        raw = np.load(row["artifact"], mmap_mode="r")[layer_index]
        current = tuple(int(value) for value in raw.shape[:2])
        if shape is None:
            shape = current
        if current != shape:
            raise RuntimeError(f"variable query grid: {current} != {shape}")
        flat = np.asarray(raw).reshape(-1, DIM)
        arrays.append(flat); lengths.append(len(flat))
    assert shape is not None
    return np.concatenate(arrays, axis=0), shape, lengths


@torch.inference_mode()
def exact_l2_1nn(query_np: np.ndarray, bank: torch.Tensor, bank_norm: torch.Tensor,
                 query_chunk: int, bank_chunk: int) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(query_np), query_chunk):
        query = torch.as_tensor(np.asarray(query_np[start:start + query_chunk]), device="cuda").float()
        query_norm = query.square().sum(1)
        best = torch.full((len(query),), float("inf"), device="cuda")
        for bank_start in range(0, len(bank), bank_chunk):
            block = bank[bank_start:bank_start + bank_chunk]
            block_norm = bank_norm[bank_start:bank_start + len(block)]
            distance = (query_norm[:, None] + block_norm[None] - 2.0 * (query @ block.T)).clamp_min_(0)
            best = torch.minimum(best, distance.min(1).values)
        output.append((best.sqrt() / math.sqrt(DIM)).cpu().numpy().astype(np.float32))
    return np.concatenate(output)


def unit_valid(unit: Path) -> bool:
    marker_path = unit / "C1_PREDICT_COMPLETE.json"
    if not marker_path.exists():
        return False
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for item in marker.get("artifacts", []):
        path = unit / item["name"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            return False
    return marker.get("status") == "complete" and not marker.get("gt_opened", True)


def run_unit(category: str, layer: int, query_chunk: int, bank_chunk: int, image_batch: int) -> None:
    unit = OUT / category / f"L{layer}"
    if unit_valid(unit):
        print(f"SKIP {category} L{layer} complete", flush=True)
        return
    unit.mkdir(parents=True, exist_ok=True)
    ids, select_sha = verify_select(category, layer)
    rows = resolve_rows(category)
    bank_np = reconstruct_bank(rows, layer, ids)
    bank_content_sha = hashlib.sha256(bank_np.tobytes()).hexdigest()
    bank = torch.as_tensor(bank_np, device="cuda").float()
    bank_norm = bank.square().sum(1)
    layer_index = LAYERS.index(layer)
    shape: tuple[int, int] | None = None
    outputs: dict[str, list[np.ndarray]] = {"calibration": [], "public": []}
    timings: list[float] = []
    records: dict[str, list[str]] = {}
    for split in ("calibration", "public"):
        split_rows = [row for row in rows if row["split"] == split]
        records[split] = [row["path"] for row in split_rows]
        seen = 0
        for batch in batches(split_rows, image_batch):
            query, shape, lengths = load_query_batch(batch, layer_index, shape)
            torch.cuda.synchronize(); started = time.perf_counter()
            score = exact_l2_1nn(query, bank, bank_norm, query_chunk, bank_chunk)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - started
            cursor = 0
            for length in lengths:
                outputs[split].append(score[cursor:cursor + length].reshape(shape))
                if split == "public":
                    timings.append(elapsed / len(lengths))
                cursor += length
            seen += len(batch)
            print(f"{split.upper()} {category} L{layer} {seen}/{len(split_rows)} sec/image={elapsed/len(lengths):.4f}", flush=True)
    calibration = np.stack(outputs["calibration"]).astype(np.float32)
    public = np.stack(outputs["public"]).astype(np.float32)
    atomic_npy(unit / "calibration_scores.npy", calibration)
    atomic_npy(unit / "public_scores.npy", public)
    atomic_npy(unit / "public_seconds.npy", np.asarray(timings, np.float64))
    atomic_json(unit / "records.json", records)
    atomic_json(unit / "MODEL.json", {
        "category": category, "layer": layer, "method": "CP100-subset raw1280 exact 1-NN",
        "dimension": DIM, "bank_count": TARGET, "selected_ids_sha256": select_sha,
        "bank_content_sha256": bank_content_sha, "score": "min raw Euclidean / sqrt(1280)",
        "calibration_shape": list(calibration.shape), "public_shape": list(public.shape),
        "query_chunk": query_chunk, "bank_chunk": bank_chunk, "image_batch": image_batch,
        "gt_opened": False,
    })
    artifacts = []
    for name in ("calibration_scores.npy", "public_scores.npy", "public_seconds.npy", "records.json", "MODEL.json"):
        path = unit / name
        artifacts.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(unit / "C1_PREDICT_COMPLETE.json", {
        "status": "complete", "category": category, "layer": layer,
        "parent_selected_ids_sha256": select_sha, "gt_opened": False, "artifacts": artifacts,
    })
    print(f"COMPLETE {category} L{layer}", flush=True)
    del bank, bank_norm, bank_np, calibration, public
    gc.collect(); torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--layers", nargs="+", type=int, choices=LAYERS, default=list(LAYERS))
    parser.add_argument("--query-chunk", type=int, default=4096)
    parser.add_argument("--bank-chunk", type=int, default=100_000)
    parser.add_argument("--image-batch", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for category in args.categories:
        for layer in args.layers:
            run_unit(category, layer, args.query_chunk, args.bank_chunk, args.image_batch)
    units = sorted(OUT.glob("*/L*/C1_PREDICT_COMPLETE.json"))
    if len(units) == len(CATEGORIES) * len(LAYERS):
        atomic_json(OUT / "C1_PREDICT_COMPLETE.json", {
            "status": "complete", "units": [str(path.parent.relative_to(OUT)) for path in units],
            "gt_opened": False,
        })


if __name__ == "__main__":
    main()
