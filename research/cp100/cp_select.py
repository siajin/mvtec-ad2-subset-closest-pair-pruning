"""C0_SELECT: subset-local closest-pair pruning on frozen raw1280 caches.

No ground truth is opened.  The only scientific change from SuperADD100 is the
representative selector described in the approved 2026-10-12 protocol.
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
OUT = ROOT / "outputs/closest_pair_pruning_100k_8cat_v1/C0_SELECT"
INPUTS = ROOT / "INPUTS.json"
CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
LAYERS = (7, 15, 23, 31)
DIM = 1280
TARGET = 100_000
ITERATIONS = 100
TARGET_PER_SUBSET = TARGET // ITERATIONS
ROUND_REMOVE_FRACTION = 0.20
SEED = 42
FILE_SHA_CACHE: dict[str, str] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_sha256(path: Path) -> str:
    key = str(path)
    if key not in FILE_SHA_CACHE:
        FILE_SHA_CACHE[key] = sha256(path)
    return FILE_SHA_CACHE[key]


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def serialize_rng(rng: np.random.RandomState) -> dict:
    name, keys, position, has_gauss, cached = rng.get_state()
    return {
        "name": name,
        "keys": keys.tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def restore_rng(payload: dict) -> np.random.RandomState:
    rng = np.random.RandomState()
    rng.set_state((payload["name"], np.asarray(payload["keys"], np.uint32),
                   int(payload["position"]), int(payload["has_gauss"]),
                   float(payload["cached_gaussian"])))
    return rng


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
        marker_path = ROOT / "outputs/cache" / category / "GENERATION_COMPLETE.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        for row in marker["records"]:
            rows.append(dict(by_source[row["source"]], artifact=str(ROOT / row["path"])))
    return sorted(rows, key=lambda row: row["path"])


def load_fit_pool(category: str, layer: int) -> tuple[np.ndarray, list[dict]]:
    layer_index = LAYERS.index(layer)
    arrays: list[np.ndarray] = []
    provenance: list[dict] = []
    cursor = 0
    for row in resolve_rows(category):
        if row["split"] != "fit":
            continue
        path = Path(row["artifact"])
        raw = np.load(path, mmap_mode="r")
        feature = np.asarray(raw[layer_index]).reshape(-1, DIM)
        if feature.dtype != np.float16:
            raise RuntimeError(f"expected float16 cache: {path}: {feature.dtype}")
        arrays.append(np.array(feature, copy=True))
        provenance.append({
            "source": row["path"], "artifact": str(path), "artifact_sha256": cached_sha256(path),
            "start": cursor, "stop": cursor + len(feature), "tokens": len(feature),
        })
        cursor += len(feature)
    if not arrays:
        raise RuntimeError(f"no fit features for {category}")
    pool = np.concatenate(arrays, axis=0)
    if pool.shape != (cursor, DIM):
        raise RuntimeError(f"pool shape mismatch: {pool.shape}, expected {(cursor, DIM)}")
    return pool, provenance


@torch.inference_mode()
def exact_two_other_neighbors(features: torch.Tensor, query_chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """Return two nearest non-self indices and squared FP32 L2 distances."""
    count = len(features)
    if count < 3:
        raise RuntimeError("closest-pair round requires at least three active points")
    norms = features.square().sum(1)
    all_indices: list[torch.Tensor] = []
    all_distances: list[torch.Tensor] = []
    for start in range(0, count, query_chunk):
        stop = min(count, start + query_chunk)
        query = features[start:stop]
        distances = query.square().sum(1, keepdim=True) + norms[None, :] - 2.0 * (query @ features.T)
        distances.clamp_min_(0)
        local = torch.arange(stop - start, device=features.device)
        distances[local, torch.arange(start, stop, device=features.device)] = torch.inf
        values, indices = torch.topk(distances, k=2, dim=1, largest=False, sorted=True)
        all_indices.append(indices.cpu())
        all_distances.append(values.cpu())
        del distances, values, indices
    return torch.cat(all_indices).numpy(), torch.cat(all_distances).numpy()


def prune_subset(subset_features: np.ndarray, subset_ids: np.ndarray, target: int,
                 device: str, query_chunk: int) -> tuple[np.ndarray, list[dict]]:
    if len(subset_ids) < target:
        raise RuntimeError(f"subset smaller than target: {len(subset_ids)} < {target}")
    active_features = torch.as_tensor(subset_features, device=device).float()
    active_ids = np.asarray(subset_ids, np.int64)
    rounds: list[dict] = []
    round_index = 0
    while len(active_ids) > target:
        started = time.perf_counter()
        neighbor_indices, neighbor_distances = exact_two_other_neighbors(active_features, query_chunk)
        nearest = neighbor_indices[:, 0]
        first_distance = neighbor_distances[:, 0]
        second_distance = neighbor_distances[:, 1]
        left = np.minimum(np.arange(len(active_ids), dtype=np.int64), nearest)
        right = np.maximum(np.arange(len(active_ids), dtype=np.int64), nearest)
        order = np.lexsort((active_ids[right], active_ids[left], first_distance))
        alive = np.ones(len(active_ids), dtype=bool)
        remove_limit = min(len(active_ids) - target,
                           max(1, int(math.floor(len(active_ids) * ROUND_REMOVE_FRACTION))))
        removed: list[int] = []
        removed_distances: list[float] = []
        ties = 0
        seen: set[tuple[int, int]] = set()
        for edge_index in order:
            i = int(left[edge_index]); j = int(right[edge_index])
            edge = (i, j)
            if i == j or edge in seen:
                continue
            seen.add(edge)
            if not alive[i] or not alive[j]:
                continue
            escape_i = float(second_distance[i]) if int(nearest[i]) == j else float(first_distance[i])
            escape_j = float(second_distance[j]) if int(nearest[j]) == i else float(first_distance[j])
            if escape_i < escape_j:
                victim = i
            elif escape_j < escape_i:
                victim = j
            else:
                ties += 1
                victim = i if active_ids[i] > active_ids[j] else j
            alive[victim] = False
            removed.append(victim)
            removed_distances.append(float(min(first_distance[i], first_distance[j])))
            if len(removed) == remove_limit:
                break
        if not removed:
            raise RuntimeError("closest-pair round made no progress")
        kept = np.flatnonzero(alive)
        elapsed = time.perf_counter() - started
        distance_array = np.sqrt(np.maximum(np.asarray(removed_distances, np.float64), 0.0))
        rounds.append({
            "round": round_index, "before": len(active_ids), "removed": len(removed),
            "after": len(kept), "tie_count": ties, "seconds": elapsed,
            "removed_l2_min": float(distance_array.min()),
            "removed_l2_median": float(np.median(distance_array)),
            "removed_l2_max": float(distance_array.max()),
        })
        active_features = active_features[torch.as_tensor(kept, device=device)]
        active_ids = active_ids[kept]
        round_index += 1
    return active_ids, rounds


def validate_synthetic(device: str, query_chunk: int) -> dict:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(32, 8)).astype(np.float32)
    values = np.concatenate((base, base[:16] + 1e-5, rng.normal(size=(16, 8)).astype(np.float32)), axis=0)
    ids = np.arange(len(values), dtype=np.int64)
    first, rounds_a = prune_subset(values.astype(np.float16), ids, 16, device, query_chunk)
    second, rounds_b = prune_subset(values.astype(np.float16), ids, 16, device, query_chunk)
    if not np.array_equal(first, second) or len(np.unique(first)) != 16:
        raise RuntimeError("synthetic determinism/uniqueness check failed")
    return {"input": len(values), "selected": len(first), "rounds": len(rounds_a),
            "selected_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "selected_replay_equal": bool(np.array_equal(first, second)),
            "round_structure_equal": [
                (row["before"], row["removed"], row["after"], row["tie_count"])
                for row in rounds_a
            ] == [
                (row["before"], row["removed"], row["after"], row["tie_count"])
                for row in rounds_b
            ]}


def run_unit(category: str, layer: int, device: str, query_chunk: int,
             max_subsets: int | None, output: Path) -> None:
    unit = output / category / f"L{layer}"
    unit.mkdir(parents=True, exist_ok=True)
    complete_path = unit / "C0_SELECT_COMPLETE.json"
    if max_subsets is None and complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        artifacts_ok = complete.get("status") == "complete" and complete.get("selected_count") == TARGET
        for item in complete.get("artifacts", []):
            path = unit / item["name"]
            artifacts_ok = artifacts_ok and path.is_file() and path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"]
        if artifacts_ok:
            print(f"SKIP {category} L{layer} complete", flush=True)
            return
        raise RuntimeError(f"invalid completed unit: {unit}")
    checkpoint_path = unit / "CHECKPOINT.json"
    selected_path = unit / "selected_ids.partial.npy"
    diagnostics_path = unit / "subset_diagnostics.partial.json"
    pool, provenance = load_fit_pool(category, layer)
    count = len(pool)
    subset_size = int(count / ITERATIONS)
    if subset_size < TARGET_PER_SUBSET:
        raise RuntimeError(f"candidate pool too small: {count}")
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        selected = np.load(selected_path).astype(np.int64).tolist()
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        rng = restore_rng(checkpoint["rng_state"])
        start_iteration = int(checkpoint["next_iteration"])
    else:
        selected = []
        diagnostics = []
        rng = np.random.RandomState(SEED)
        start_iteration = 0
    selected_mask = np.zeros(count, dtype=bool)
    if selected:
        selected_mask[np.asarray(selected, np.int64)] = True
    stop_iteration = ITERATIONS if max_subsets is None else min(ITERATIONS, start_iteration + max_subsets)
    for iteration in range(start_iteration, stop_iteration):
        candidates = np.flatnonzero(~selected_mask)
        subset_ids = rng.choice(candidates, size=min(subset_size, len(candidates)), replace=False)
        subset_features = pool[subset_ids]
        started = time.perf_counter()
        kept, rounds = prune_subset(subset_features, subset_ids, TARGET_PER_SUBSET, device, query_chunk)
        elapsed = time.perf_counter() - started
        if selected_mask[kept].any() or len(np.unique(kept)) != TARGET_PER_SUBSET:
            raise RuntimeError("duplicate selected ID")
        selected_mask[kept] = True
        selected.extend(int(value) for value in kept)
        diagnostics.append({"iteration": iteration, "subset_size": len(subset_ids),
                            "selected": len(kept), "seconds": elapsed, "rounds": rounds})
        atomic_npy(selected_path, np.asarray(selected, np.int64))
        atomic_json(diagnostics_path, diagnostics)
        atomic_json(checkpoint_path, {
            "status": "partial", "category": category, "layer": layer,
            "candidate_count": count, "subset_size": subset_size,
            "next_iteration": iteration + 1, "selected_count": len(selected),
            "rng_state": serialize_rng(rng), "gt_opened": False,
        })
        print(f"SUBSET {category} L{layer} {iteration + 1}/{ITERATIONS} "
              f"selected={len(selected)} seconds={elapsed:.3f}", flush=True)
    if stop_iteration == ITERATIONS:
        selected_array = np.asarray(selected, np.int64)
        if selected_array.shape != (TARGET,) or len(np.unique(selected_array)) != TARGET:
            raise RuntimeError(f"final selected ID contract failed: {selected_array.shape}")
        final_path = unit / "selected_ids.npy"
        atomic_npy(final_path, selected_array)
        atomic_json(unit / "source_provenance.json", provenance)
        artifacts = []
        for path in (final_path, diagnostics_path, unit / "source_provenance.json"):
            artifacts.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
        atomic_json(unit / "C0_SELECT_COMPLETE.json", {
            "status": "complete", "category": category, "layer": layer,
            "method": "CP100-subset", "candidate_count": count, "selected_count": TARGET,
            "dimension": DIM, "seed": SEED, "iterations": ITERATIONS,
            "target_per_subset": TARGET_PER_SUBSET, "round_remove_fraction": ROUND_REMOVE_FRACTION,
            "distance": "raw1280 FP32 squared Euclidean for ordering", "normalize": False,
            "endpoint_rule": "remove smaller escape distance; tie removes larger global ID",
            "gt_opened": False, "artifacts": artifacts,
        })
        print(f"COMPLETE {category} L{layer}", flush=True)
    del pool
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--layers", nargs="+", type=int, choices=LAYERS, default=list(LAYERS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--query-chunk", type=int, default=2048)
    parser.add_argument("--max-subsets", type=int)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    synthetic = validate_synthetic(args.device, min(args.query_chunk, 64))
    print("SYNTHETIC", json.dumps(synthetic, sort_keys=True), flush=True)
    if args.synthetic_only:
        return
    for category in args.categories:
        for layer in args.layers:
            run_unit(category, layer, args.device, args.query_chunk, args.max_subsets, args.output)


if __name__ == "__main__":
    main()
