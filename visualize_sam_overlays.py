"""
Create quick overlays of ground-truth keypoints on dataset images.

Usage:
    python visualize_sam_overlays.py \\
        --dataset-dir data/kaia_small_sam_exports/dataset_21_release_45_sam \\
        --provider data_providers/kccd_2d/dataset_21_release_45.yaml \\
        [--show-mesh --mhr-model checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt]

Outputs PNGs for the first N frames (default 10) into dataset-dir/overlays/.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from kaia_commons.kccd import DatasetManager
from kaia_commons.models import KeyPointConfigurationEnum, KeyPointConfigurationFactory
from kaia_commons.utils.file_utils import read_yaml
from kaia_commons.utils.keypoints_2d import rotate_keypoints_90

MESH_COLOR = (0.65098039, 0.74117647, 0.85882353)


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


def normalized_body_to_pixel_points(
    body: Dict[str, Dict[str, float]],
    kp_config,
    img_shape: Tuple[int, int, int],
    is_landscape: bool,
) -> List[Tuple[float, float]]:
    """
    Convert a normalized BODY dict (x in [0,0.75], y in [0,1]) to pixel points,
    applying the landscape rotation back to image space when needed.
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

    h, w = img_shape[:2]
    pts: List[Tuple[float, float]] = []
    for idx in range(len(kp_config)):
        if not mask[idx] or np.isnan(kp_arr[idx, 0]) or np.isnan(kp_arr[idx, 1]):
            pts.append(None)
            continue
        x = float(kp_arr[idx, 0]) * (w / 0.75)
        y = float(kp_arr[idx, 1]) * h
        pts.append((x, y))
    return pts


def draw_points(img: np.ndarray, points: List[Tuple[float, float]], color: Tuple[int, int, int]):
    out = img.copy()
    for pt in points:
        if pt is None:
            continue
        cv2.circle(out, (int(pt[0]), int(pt[1])), 3, color, thickness=-1, lineType=cv2.LINE_AA)
    return out


def load_first_prediction(extras_dir: Path, frame_id: int) -> Optional[Dict]:
    """Load the first detection stored in extras/<frame_id>.npz."""
    npz_path = extras_dir / f"{frame_id}.npz"
    if npz_path.exists():
        # Some fields may have been saved as object arrays; allow pickle to handle them.
        with np.load(npz_path, allow_pickle=True) as data:
            has_prediction = bool(np.array(data["has_prediction"]).item()) if "has_prediction" in data.files else True
            if not has_prediction:
                return None
            return {k: np.array(data[k]) for k in data.files if k != "has_prediction"}
    return None


def kaia23_points_from_prediction(
    prediction: Dict,
    kp_config,
) -> List[Optional[Tuple[float, float]]]:
    """Extract KAIA_23 keypoints in pixel space from a prediction dict with mask."""
    keypoints = np.asarray(prediction.get("kaia23_keypoints", []))
    mask = np.asarray(prediction.get("kaia23_mask", []), dtype=bool)
    pts: List[Optional[Tuple[float, float]]] = []
    for idx in range(len(kp_config)):
        if idx >= len(keypoints) or idx >= len(mask) or not mask[idx]:
            pts.append(None)
            continue
        x, y = keypoints[idx]
        pts.append((float(x), float(y)))
    return pts


def load_mhr_faces(mhr_model_path: Path) -> Optional[np.ndarray]:
    """Load mesh faces from the TorchScript MHR asset (used for rendering meshes)."""
    try:
        import torch
    except ImportError:
        print("[warn] torch is required to load faces for mesh rendering")
        return None

    if not mhr_model_path.exists():
        print(f"[warn] mhr_model.pt not found at {mhr_model_path}, skipping mesh rendering")
        return None

    try:
        model = torch.jit.load(str(mhr_model_path), map_location="cpu")
        faces = model.character_torch.mesh.faces.to("cpu").numpy()
        return faces
    except Exception as exc:
        print(f"[warn] failed to load faces from {mhr_model_path}: {exc}")
        return None


def render_mesh_overlay(
    img_bgr: np.ndarray,
    prediction: Dict,
    faces: np.ndarray,
) -> np.ndarray:
    """Render the predicted mesh onto the input image."""
    from sam_3d_body.visualization.renderer import Renderer

    renderer = Renderer(focal_length=float(np.asarray(prediction["focal_length"])), faces=faces)
    rendered = renderer(
        np.asarray(prediction["pred_vertices"]),
        np.asarray(prediction["pred_cam_t"]),
        img_bgr.copy(),
        mesh_base_color=MESH_COLOR,
        scene_bg_color=(1, 1, 1),
    )
    return (rendered * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Visualize GT vs SAM keypoints on first N frames.")
    parser.add_argument(
        "--dataset-dir", type=Path, required=True, help="Path to exported dataset (with images/ & descriptor.json)."
    )
    parser.add_argument("--provider", type=Path, required=True, help="Data provider YAML (for keypoint config).")
    parser.add_argument("--limit", type=int, default=10, help="Number of frames to render.")
    parser.add_argument(
        "--show-mesh",
        action="store_true",
        help="Render the predicted mesh on top of the image using extras/<frame_id>.npz (requires mhr_model.pt).",
    )
    parser.add_argument(
        "--mhr-model",
        type=Path,
        default=Path("checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"),
        help="Path to mhr_model.pt (TorchScript). Needed when --show-mesh is set.",
    )
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

    overlays_dir = dataset_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    extras_dir = dataset_dir / "extras"
    faces = load_mhr_faces(args.mhr_model) if args.show_mesh else None
    sam_descriptor = json.loads((dataset_dir / "descriptor.json").read_text())
    sam_map = {entry["id"]: entry for entry in sam_descriptor.get("frames", [])}

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

        body_pts = normalized_body_to_pixel_points(
            gt_frame["ground_truth"]["body"],
            kp_config,
            img.shape,
            is_landscape=gt_frame.get("is_landscape", False),
        )

        # SAM keypoints from descriptor.json (normalized coords; rotate/scale to pixels)
        sam_frame = sam_map.get(frame_id)
        if sam_frame:
            sam_body = sam_frame["ground_truth"]["body"]
            sam_pts = normalized_body_to_pixel_points(
                sam_body,
                kp_config,
                img.shape,
                is_landscape=sam_frame.get("is_landscape", gt_frame.get("is_landscape", False)),
            )
        else:
            sam_pts = [None] * len(body_pts)

        overlay_base = img
        prediction = None
        if args.show_mesh and faces is not None:
            prediction = load_first_prediction(extras_dir, frame_id)
            if prediction is not None:
                try:
                    overlay_base = render_mesh_overlay(img, prediction, faces)
                except Exception as exc:
                    print(f"[warn] failed to render mesh for frame {frame_id}: {exc}")
                    overlay_base = img

        overlay = draw_points(overlay_base, body_pts, color=(0, 255, 0))  # GT in green
        overlay = draw_points(overlay, sam_pts, color=(255, 0, 0))  # SAM in blue

        out_path = overlays_dir / f"{frame_id}.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
