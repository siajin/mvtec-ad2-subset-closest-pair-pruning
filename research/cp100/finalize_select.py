"""Validate all C0_SELECT units and publish the immutable parent marker."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("MVTEC_GRID_ROOT", "/workspace/mvtec_grid_v1"))
OUT = ROOT / "outputs/closest_pair_pruning_100k_8cat_v1/C0_SELECT"
BASE = ROOT / "outputs/superadd100_native_single_layer_8cat_v1/c0_generate"
CATEGORIES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
LAYERS = (7, 15, 23, 31)
TARGET = 100_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> None:
    summaries = []
    units = []
    for category in CATEGORIES:
        for layer in LAYERS:
            unit = OUT / category / f"L{layer}"
            marker_path = unit / "C0_SELECT_COMPLETE.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker["status"] != "complete" or marker["selected_count"] != TARGET or marker["gt_opened"]:
                raise RuntimeError(f"unit marker contract failed: {unit}")
            for item in marker["artifacts"]:
                path = unit / item["name"]
                if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                    raise RuntimeError(f"artifact mismatch: {path}")
            ids_path = unit / "selected_ids.npy"
            ids = np.load(ids_path).astype(np.int64)
            candidate_count = int(marker["candidate_count"])
            if ids.shape != (TARGET,) or len(np.unique(ids)) != TARGET or ids.min() < 0 or ids.max() >= candidate_count:
                raise RuntimeError(f"selected ID range/uniqueness failed: {unit}")
            baseline_path = BASE / category / f"L{layer}" / "selected_ids.npy"
            baseline = np.load(baseline_path).astype(np.int64)
            overlap = int(np.intersect1d(ids, baseline, assume_unique=False).size)
            provenance = json.loads((unit / "source_provenance.json").read_text(encoding="utf-8"))
            image_counts = []
            for row in provenance:
                image_counts.append(int(np.count_nonzero((ids >= row["start"]) & (ids < row["stop"]))))
            summaries.append({
                "category": category, "layer": layer, "candidate_count": candidate_count,
                "selected_count": TARGET, "selected_ids_sha256": sha256(ids_path),
                "density_thinning_overlap_count": overlap,
                "density_thinning_overlap_fraction": overlap / TARGET,
                "source_images": len(image_counts),
                "source_images_represented": int(np.count_nonzero(image_counts)),
                "selected_per_image_min": int(np.min(image_counts)),
                "selected_per_image_median": float(np.median(image_counts)),
                "selected_per_image_max": int(np.max(image_counts)),
            })
            units.append({
                "category": category, "layer": layer,
                "marker_sha256": sha256(marker_path), "selected_ids_sha256": sha256(ids_path),
            })
    summary_path = OUT / "SELECTION_SUMMARY.json"
    atomic_json(summary_path, summaries)
    atomic_json(OUT / "C0_SELECT_COMPLETE.json", {
        "status": "complete", "method": "CP100-subset", "unit_count": len(units),
        "categories": list(CATEGORIES), "layers": list(LAYERS), "gt_opened": False,
        "units": units, "selection_summary_sha256": sha256(summary_path),
    })
    print(json.dumps({"status": "complete", "units": len(units),
                      "overlap_fraction_mean": float(np.mean([row["density_thinning_overlap_fraction"] for row in summaries]))}, indent=2))


if __name__ == "__main__":
    main()
