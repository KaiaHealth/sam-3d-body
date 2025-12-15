"""
Utility script to export SAM 3D Body predictions for the kaia_small_* splits.

It downloads a KAIA dataset release via kaia-commons (same flow the training
engine uses), runs the SAM inference engine on every frame, and writes a new
dataset with SAM outputs and KAIA_23 keypoints.

Each output dataset mirrors the usual layout:
    {split}_sam/
        images/            # copied PNGs
        descriptor.json    # KAIA_23 keypoints normalized to [0, 1]
        sam_outputs.json   # full SAM outputs per frame (all detections)
"""

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import orjson
from kaia_commons.kccd import DatasetManager
from kaia_commons.models import KeyPointConfigurationEnum, KeyPointConfigurationFactory
from kaia_commons.utils.file_utils import read_yaml
from kaia_commons.utils.keypoints_2d import rotate_keypoints_90
from tqdm import tqdm

from sam_3d_body import SAM3DBodyEstimator
from sam_inference_engine import Sam3DBodyInferenceEngine
from sam_utils import build_estimator


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to plain Python for JSON dumping."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def _build_body_dict(
    kp_config,
    keypoints_norm: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Create a BODY dict (KAIA_23) with normalized coordinates and visibility."""
    body: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(kp_config.keypoint_names):
        visible = bool(mask[idx])
        if visible:
            x = float(keypoints_norm[idx, 0])
            y = float(keypoints_norm[idx, 1])
            body[name] = {"x": x, "y": y, "c": 1.0}
        else:
            body[name] = {"x": float("nan"), "y": float("nan"), "i": True}
    return body


def _empty_body_dict(kp_config) -> Dict[str, Dict[str, float]]:
    """Return a BODY dict with zeroed coords and invisible keypoints."""
    body: Dict[str, Dict[str, float]] = {}
    for name in kp_config.keypoint_names:
        body[name] = {"x": 0.0, "y": 0.0, "c": 0.0, "i": True}
    return body


def _write_json(path: Path, obj: Any):
    path.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def process_dataset(
    name: str,
    dataset_info: Dict[str, Any],
    estimator: SAM3DBodyEstimator,
    bbox_thresh: float,
    use_mask: bool,
    output_root: Path,
):
    """Download a dataset release, run SAM inference, and write the exported files."""
    kp_name = dataset_info.get("source_kp_config") or KeyPointConfigurationEnum.KAIA_23.name
    kp_config = KeyPointConfigurationFactory.from_str(kp_name)
    engine = Sam3DBodyInferenceEngine(estimator, bbox_thresh=bbox_thresh, use_mask=use_mask, kp_config=kp_config)

    filters = dataset_info.get("filters") or {}
    dataset = DatasetManager.get_dataset(id=dataset_info["dataset_id"], use_cache_on_failure=True).get_release(
        id=dataset_info["release_id"]
    )
    frames = dataset.get_frames(**filters)
    img_folder: Path = dataset.download_images(n_jobs=4, **filters)

    split_out_dir = output_root / name
    images_out_dir = split_out_dir / "images"
    descriptor_path = split_out_dir / "descriptor.json"
    sam_outputs_path = split_out_dir / "sam_outputs.json"
    split_out_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir.mkdir(parents=True, exist_ok=True)

    descriptor_frames: List[dict] = []
    sam_outputs: List[dict] = []

    for frame in tqdm(frames, desc=f"Processing {name}", unit="frame"):
        frame_id = frame["id"]
        img_path = img_folder / f"{frame_id}.png"
        if not img_path.exists():
            print(f"[warn] missing image for frame {frame_id}, skipping")
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[warn] failed to read {img_path}, skipping")
            continue

        predictions = engine.run_single(image_bgr)
        h, w = image_bgr.shape[:2]
        if len(predictions) == 0:
            print(f"[warn] no detections for frame {frame_id}, writing empty body")
            body = _empty_body_dict(kp_config)
        else:
            # Pick the first person (dataset frames are single-person); keep all in sam_outputs.json
            first_pred = predictions[0]
            keypoints_norm = np.asarray(first_pred["kaia23_keypoints"], dtype=np.float32).copy()
            # Normalize to the same 4:3 space as GT (x in [0, 0.75], y in [0, 1])
            keypoints_norm[:, 0] = (keypoints_norm[:, 0] / float(w)) * 0.75
            keypoints_norm[:, 1] = keypoints_norm[:, 1] / float(h)
            if frame.get("is_landscape", False):
                keypoints_norm = rotate_keypoints_90(keypoints_norm, clockwise=False)
            body = _build_body_dict(
                kp_config=kp_config,
                keypoints_norm=keypoints_norm,
                mask=first_pred["kaia23_mask"],
            )

        descriptor_frames.append(
            {
                "id": frame_id,
                "is_landscape": frame.get("is_landscape", False),
                "ground_truth": {"body": body},
            }
        )
        sam_outputs.append({"frame_id": frame_id, "predictions": _to_serializable(predictions)})

        # Copy image to output dataset
        shutil.copy2(img_path, images_out_dir / img_path.name)

        # Persist progress after first 10 frames
        if len(descriptor_frames) == 10:
            _write_json(descriptor_path, {"frames": descriptor_frames})
            _write_json(sam_outputs_path, sam_outputs)

    # Final write at the end
    _write_json(descriptor_path, {"frames": descriptor_frames})
    _write_json(sam_outputs_path, sam_outputs)

    print(f"[done] wrote {descriptor_path} and {sam_outputs_path} ({len(descriptor_frames)} frames)")


def _dataset_info_from_provider(path: Path) -> Dict[str, Any]:
    cfg = read_yaml(path)
    args = cfg.get("args", {})
    return {
        "dataset_id": args.get("dataset_id"),
        "release_id": args.get("release_id"),
        "source_kp_config": args.get("source_kp_config"),
        "filters": args.get("filters", {}),
    }


def load_dataset_info(provider: Path) -> Dict[str, Any]:
    """Read dataset ids and filters from the given data provider YAML file."""
    return _dataset_info_from_provider(provider.resolve())


def parse_args():
    parser = argparse.ArgumentParser(description="Export kaia_small_* splits with SAM outputs.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/sam-3d-body-dinov3/model.ckpt"),
        help="Path to SAM 3D Body checkpoint (.ckpt). Defaults to DINOv3 checkpoint.",
    )
    parser.add_argument(
        "--mhr",
        type=Path,
        default=Path("checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"),
        help="Path to MHR assets (mhr_model.pt). Defaults to DINOv3 assets.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/kaia_small_sam_exports"))
    parser.add_argument("--bbox-thresh", type=float, default=0.8)
    parser.add_argument("--use-mask", action="store_true", help="Enable SAM2 mask-conditioned inference")
    parser.add_argument("--use-detector", action="store_true", default=False)
    parser.add_argument("--detector-path", type=str, default="")
    parser.add_argument("--use-segmentor", action="store_true", default=False)
    parser.add_argument("--segmentor-path", type=str, default="")
    parser.add_argument("--use-fov", action="store_true", default=False)
    parser.add_argument("--fov-path", type=str, default="")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument(
        "--provider",
        type=Path,
        required=True,
        help="Data provider YAML for the dataset release to process (kccd_2d).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Folder name under output-root for the exported dataset. Defaults to dataset_{id}_release_{release}_sam.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_info = load_dataset_info(args.provider)

    estimator, _ = build_estimator(
        checkpoint_path=str(args.checkpoint),
        mhr_path=str(args.mhr),
        use_cuda=not args.cpu,
        use_detector=args.use_detector,
        detector_name="vitdet",
        detector_path=args.detector_path,
        use_segmentor=args.use_segmentor,
        segmentor_name="sam2",
        segmentor_path=args.segmentor_path,
        use_fov=args.use_fov,
        fov_name="moge2",
        fov_path=args.fov_path,
    )

    output_root: Path = args.output_root
    output_name = args.output_name or f"dataset_{dataset_info['dataset_id']}_release_{dataset_info['release_id']}_sam"
    process_dataset(
        name=output_name,
        dataset_info=dataset_info,
        estimator=estimator,
        bbox_thresh=args.bbox_thresh,
        use_mask=args.use_mask,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()
