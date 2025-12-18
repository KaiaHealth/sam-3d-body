"""
Compute OKS accuracy (normalized space) for SAM predictions on a KAIA dataset release.

Given:
  - a provider YAML (to fetch GT frames via kaia-commons),
  - a SAM predictions directory (with descriptor.json + images/ produced by generate_kaia_sam_datasets.py),
the script loads the first N frames, matches them by frame id, and reports mean OKS accuracy
in the normalized 4:3 keypoint space used in descriptor.json.

Example:
    python evaluate_sam_accuracy.py \
        --pred-dir data/kaia_small_sam_exports/dataset_21_release_45_sam \
        --provider data_providers/kccd_2d/dataset_21_release_45.yaml \
        --limit 100
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from kaia_commons.kccd import DatasetManager
from kaia_commons.models import KeyPointConfigurationEnum, KeyPointConfigurationFactory
from kaia_commons.utils.file_utils import read_yaml
from kaia_commons.utils.keypoints_2d import rotate_keypoints_90
from training_engine.metrics.metrics_2d import oks_accuracy


def load_provider_info(provider_path: Path):
    cfg = read_yaml(provider_path)
    args = cfg.get("args", {})
    kp_name = args.get("source_kp_config") or KeyPointConfigurationEnum.KAIA_23.name
    kp_config = KeyPointConfigurationFactory.from_str(kp_name)
    return {
        "kp_config": kp_config,
        "dataset_id": args.get("dataset_id"),
        "release_id": args.get("release_id"),
        "filters": args.get("filters", {}),
    }


def normalized_body_to_arrays(
    body: Dict[str, Dict[str, float]],
    kp_config,
    is_landscape: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a normalized BODY dict (x in [0,0.75], y in [0,1]) to coordinates and visibility mask.
    Returns:
        keypoints: (K, 2) float array with NaNs for missing.
        mask: (K,) bool array, True when visible.
    """
    kp_arr = np.full((len(kp_config), 2), np.nan, dtype=np.float32)
    mask = np.zeros((len(kp_config),), dtype=bool)
    for idx, name in enumerate(kp_config.keypoint_names):
        kp = body.get(name)
        if kp and not kp.get("i"):
            kp_arr[idx, 0] = kp.get("x", 0.0)
            kp_arr[idx, 1] = kp.get("y", 0.0)
            mask[idx] = True

    if is_landscape:
        kp_arr = rotate_keypoints_90(kp_arr, clockwise=True)

    for idx in range(len(kp_config)):
        if not mask[idx] or np.isnan(kp_arr[idx, 0]) or np.isnan(kp_arr[idx, 1]):
            kp_arr[idx] = np.nan
            mask[idx] = False
    return kp_arr, mask


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM OKS accuracy against KAIA GT.")
    parser.add_argument(
        "--sam-dir",
        type=Path,
        default=Path("data/kaia_small_sam_exports/dataset_21_release_45_sam"),
        help="SAM predictions directory containing descriptor.json.",
    )
    parser.add_argument(
        "--provider",
        type=Path,
        default=Path("data_providers/kccd_2d/dataset_21_release_45.yaml"),
        help="Provider YAML for the KAIA dataset release.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Number of frames to evaluate (default: 100).")
    args = parser.parse_args()

    pred_dir = args.sam_dir
    descriptor_path = pred_dir / "descriptor.json"
    if not descriptor_path.exists():
        raise FileNotFoundError(f"descriptor.json not found in {pred_dir}")

    provider_info = load_provider_info(args.provider)
    kp_config = provider_info["kp_config"]

    dataset = DatasetManager.get_dataset(id=provider_info["dataset_id"], use_cache_on_failure=True).get_release(
        id=provider_info["release_id"]
    )
    gt_frames = dataset.get_frames(**(provider_info.get("filters") or {}))
    sam_descriptor = json.loads(descriptor_path.read_text())
    sam_map = {entry["id"]: entry for entry in sam_descriptor.get("frames", [])}

    oks_norm_scores: List[float] = []
    vis_acc_scores: List[float] = []
    processed = 0
    missing_pred = 0

    for frame in gt_frames[: args.limit]:
        frame_id = frame["id"]
        pred_frame = sam_map.get(frame_id)
        if pred_frame is None:
            missing_pred += 1
            continue

        gt_body = frame["ground_truth"]["body"]
        pred_body = pred_frame["ground_truth"]["body"]
        gt_norm_pts, gt_norm_mask = normalized_body_to_arrays(
            gt_body, kp_config, is_landscape=frame.get("is_landscape", False)
        )
        pred_norm_pts, pred_norm_mask = normalized_body_to_arrays(
            pred_body,
            kp_config,
            is_landscape=pred_frame.get("is_landscape", frame.get("is_landscape", False)),
        )
        eval_mask = gt_norm_mask & pred_norm_mask
        oks_norm = oks_accuracy(
            pred_norm_pts[np.newaxis, ...],
            gt_norm_pts[np.newaxis, ...],
            eval_mask[np.newaxis, ...],
        )
        oks_norm_val = float(oks_norm.item())
        if np.isfinite(oks_norm_val):
            oks_norm_scores.append(oks_norm_val)

        # Visibility flag accuracy (exact match of visibility mask)
        if gt_norm_mask.size == pred_norm_mask.size:
            vis_acc = np.mean(gt_norm_mask == pred_norm_mask)
            if np.isfinite(vis_acc):
                vis_acc_scores.append(float(vis_acc))
            processed += 1

    print(f"[report] frames processed: {processed}, missing_pred: {missing_pred}")
    mean_oks_norm = float(np.mean(oks_norm_scores)) if oks_norm_scores else float("nan")
    print(f"[report] mean OKS via training_engine.oks_accuracy (normalized space): {mean_oks_norm:.4f}")
    mean_vis_acc = float(np.mean(vis_acc_scores)) if vis_acc_scores else float("nan")
    print(f"[report] mean visibility accuracy (mask match): {mean_vis_acc:.4f}")


if __name__ == "__main__":
    main()
