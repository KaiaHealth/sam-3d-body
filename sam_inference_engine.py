"""
Lightweight helpers to run SAM 3D Body inference in small apps.

The `Sam3DBodyInferenceEngine` wraps a `SAM3DBodyEstimator` so you can run a batch
of images with a consistent bbox threshold / mask flag. The module also exposes
`extract_kaia23_keypoints` to map mesh vertices back to KAIA_23 keypoints in pixel
space, which is useful for comparing against other pose estimators.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from kaia_commons.models import KeyPointConfigurationEnum, KeyPointConfigurationFactory

from mapping import MAPPING
from sam_3d_body import SAM3DBodyEstimator


def _build_cam_intrinsics(focal_length: float, img_w: int, img_h: int) -> np.ndarray:
    """Camera intrinsics assuming the principal point is the image center."""
    return np.array(
        [
            [focal_length, 0.0, img_w / 2.0],
            [0.0, focal_length, img_h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _project_vertices_to_image(vertices: np.ndarray, cam_t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Project mesh vertices to image plane using perspective projection.
    """
    verts_cam = vertices + cam_t.reshape(1, 3)
    z = np.clip(verts_cam[:, 2:3], 1e-6, None)
    verts_norm = verts_cam / z
    verts_h = verts_norm @ K.T
    return verts_h[:, :2]


def extract_kaia23_keypoints(
    prediction: Dict,
    image_shape: Tuple[int, int, int],
    kp_config=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map SAM 3D Body mesh vertices to KAIA_23 keypoints in pixel coordinates.

    Returns:
        keypoints (np.ndarray): (23, 2) pixel coordinates, np.nan when not available.
        mask (np.ndarray): boolean mask indicating which keypoints were filled.
    """
    if kp_config is None:
        kp_config = KeyPointConfigurationFactory.from_enum(KeyPointConfigurationEnum.KAIA_23)

    img_h, img_w = image_shape[:2]
    vertices = np.asarray(prediction["pred_vertices"], dtype=np.float32)
    cam_t = np.asarray(prediction["pred_cam_t"], dtype=np.float32)
    focal_length = float(prediction["focal_length"])

    verts_2d = _project_vertices_to_image(
        vertices,
        cam_t,
        _build_cam_intrinsics(focal_length, img_w=img_w, img_h=img_h),
    )

    keypoints = np.full((len(kp_config), 2), np.nan, dtype=np.float32)
    mask = np.zeros((len(kp_config),), dtype=bool)
    for idx, name in enumerate(kp_config.keypoint_names):
        vert_idx = MAPPING.get(name)
        if vert_idx is None or vert_idx >= len(verts_2d):
            continue
        keypoints[idx] = verts_2d[vert_idx]
        mask[idx] = True

    valid_x = ~np.isnan(keypoints[:, 0])
    valid_y = ~np.isnan(keypoints[:, 1])
    keypoints[valid_x, 0] = np.clip(keypoints[valid_x, 0], 0, img_w - 1)
    keypoints[valid_y, 1] = np.clip(keypoints[valid_y, 1], 0, img_h - 1)
    return keypoints, mask


class Sam3DBodyInferenceEngine:
    """
    Minimal inference helper that runs SAM 3D Body on batches of images.

    Example:
        engine = Sam3DBodyInferenceEngine(estimator, bbox_thresh=0.8, use_mask=False)
        batch_outputs = engine.run_batch([img1, img2])
    """

    def __init__(self, estimator: SAM3DBodyEstimator, bbox_thresh: float = 0.8, use_mask: bool = False):
        self.estimator = estimator
        self.bbox_thresh = bbox_thresh
        self.use_mask = use_mask

    def run_single(self, image_bgr: np.ndarray):
        """Run inference on a single BGR image."""
        return self.estimator.process_one_image(image_bgr, bbox_thr=self.bbox_thresh, use_mask=self.use_mask)

    def run_batch(self, images_bgr: Sequence[np.ndarray]) -> List[List[Dict]]:
        """Run inference on a batch of BGR images; returns per-image outputs."""
        results: List[List[Dict]] = []
        for img in images_bgr:
            if img is None:
                results.append([])
                continue
            results.append(self.run_single(img))
        return results
