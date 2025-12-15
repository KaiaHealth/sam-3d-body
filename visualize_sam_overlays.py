"""
Create quick overlays of ground-truth keypoints on dataset images.

Usage:
    python visualize_sam_overlays.py \\
        --dataset-dir data/kaia_small_sam_exports/dataset_21_release_45_sam \\
        --provider data_providers/kccd_2d/dataset_21_release_45.yaml

Outputs PNGs for the first N frames (default 10) into dataset-dir/overlays/.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from kaia_commons.kccd import DatasetManager
from kaia_commons.models import KeyPointConfigurationEnum, KeyPointConfigurationFactory
from kaia_commons.utils.file_utils import read_yaml
from kaia_commons.utils.keypoints_2d import rotate_keypoints_90


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


def body_to_pixels(body: Dict[str, Dict[str, float]], kp_config, img_shape: Tuple[int, int, int]):
    h, w = img_shape[:2]
    pts = []
    for name in kp_config.keypoint_names:
        kp = body.get(name)
        if not kp or kp.get("i"):
            pts.append(None)
            continue
        # x is normalized to 0-0.75; scale by 1/0.75 to map to full width
        x = float(kp.get("x", 0.0)) * (w / 0.75)
        y = float(kp.get("y", 0.0)) * h
        pts.append((x, y))
    return pts


def draw_points(img: np.ndarray, points: List[Tuple[float, float]], color: Tuple[int, int, int]):
    out = img.copy()
    for pt in points:
        if pt is None:
            continue
        cv2.circle(out, (int(pt[0]), int(pt[1])), 3, color, thickness=-1, lineType=cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="Visualize GT vs SAM keypoints on first N frames.")
    parser.add_argument(
        "--dataset-dir", type=Path, required=True, help="Path to exported dataset (with images/ & descriptor.json)."
    )
    parser.add_argument("--provider", type=Path, required=True, help="Data provider YAML (for keypoint config).")
    parser.add_argument("--limit", type=int, default=10, help="Number of frames to render.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    images_dir = dataset_dir / "images"
    provider_info = load_provider_info(args.provider)
    kp_config = provider_info["kp_config"]
    dataset = DatasetManager.get_dataset(id=provider_info["dataset_id"], use_cache_on_failure=True).get_release(
        id=provider_info["release_id"]
    )
    gt_frames = dataset.get_frames(**(provider_info.get("filters") or {}))
    gt_frames_map = {frame["id"]: frame for frame in gt_frames}

    # SAM outputs (normalized keypoints with is_landscape already applied in the export)
    sam_descriptor = json.loads((dataset_dir / "descriptor.json").read_text())
    sam_map = {entry["id"]: entry for entry in sam_descriptor.get("frames", [])}

    overlays_dir = dataset_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    frames = gt_frames[: args.limit]
    for frame in frames:
        frame_id = frame["id"]
        img_path = images_dir / f"{frame_id}.png"
        if not img_path.exists():
            print(f"[warn] missing image {img_path}, skipping")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] failed to read {img_path}, skipping")
            continue

        gt_frame = gt_frames_map.get(frame_id)
        if not gt_frame:
            print(f"[warn] GT frame {frame_id} not found in provider release, skipping")
            continue

        body = gt_frame["ground_truth"]["body"]
        # Rotate all GT keypoints 90° clockwise in normalized space before plotting
        kp_arr = np.full((len(kp_config), 2), np.nan, dtype=np.float32)
        for idx, name in enumerate(kp_config.keypoint_names):
            kp = body.get(name)
            if kp and not kp.get("i"):
                kp_arr[idx, 0] = kp.get("x", 0.0)
                kp_arr[idx, 1] = kp.get("y", 0.0)
        kp_arr = rotate_keypoints_90(kp_arr, clockwise=True)
        rotated_body = {}
        for idx, name in enumerate(kp_config.keypoint_names):
            kp = body.get(name, {})
            if np.isnan(kp_arr[idx, 0]) or np.isnan(kp_arr[idx, 1]) or kp.get("i"):
                rotated_body[name] = {"x": float("nan"), "y": float("nan"), "i": True}
            else:
                rotated_body[name] = {"x": float(kp_arr[idx, 0]), "y": float(kp_arr[idx, 1]), "c": kp.get("c", 1.0)}

        body_pts = body_to_pixels(rotated_body, kp_config, img.shape)

        # SAM keypoints (already normalized; apply same rotation and scaling)
        sam_frame = sam_map.get(frame_id)
        sam_pts = []
        if sam_frame:
            sam_body = sam_frame["ground_truth"]["body"]
            kp_arr_sam = np.full((len(kp_config), 2), np.nan, dtype=np.float32)
            mask_sam = np.zeros((len(kp_config),), dtype=bool)
            for idx, name in enumerate(kp_config.keypoint_names):
                kp = sam_body.get(name)
                if kp and not kp.get("i"):
                    # SAM keypoints already normalized to [0,0.75] x [0,1]; no rescale needed
                    kp_arr_sam[idx, 0] = kp.get("x", 0.0)
                    kp_arr_sam[idx, 1] = kp.get("y", 0.0)
                    mask_sam[idx] = True
            # Rotate SAM keypoints -90° (counterclockwise) to align with GT overlay
            kp_arr_sam = rotate_keypoints_90(kp_arr_sam, clockwise=False)
            rotated_sam_body = {}
            for idx, name in enumerate(kp_config.keypoint_names):
                kp = sam_body.get(name, {})
                if np.isnan(kp_arr_sam[idx, 0]) or np.isnan(kp_arr_sam[idx, 1]) or not mask_sam[idx] or kp.get("i"):
                    rotated_sam_body[name] = {"x": float("nan"), "y": float("nan"), "i": True}
                else:
                    rotated_sam_body[name] = {
                        "x": float(kp_arr_sam[idx, 0]),
                        "y": float(kp_arr_sam[idx, 1]),
                        "c": kp.get("c", 1.0),
                    }
            sam_pts = body_to_pixels(rotated_sam_body, kp_config, img.shape)
        else:
            sam_pts = [None] * len(body_pts)

        overlay = draw_points(img, body_pts, color=(0, 255, 0))  # GT in green
        overlay = draw_points(overlay, sam_pts, color=(255, 0, 0))  # SAM in blue

        out_path = overlays_dir / f"{frame_id}.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
