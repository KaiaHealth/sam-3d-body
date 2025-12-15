"""
Shared helpers for building SAM 3D Body estimators and related utilities.
"""

from typing import Tuple

import torch

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body


def pick_device(use_cuda: bool) -> str:
    """Choose CUDA when requested and available, otherwise CPU."""
    if use_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_estimator(
    checkpoint_path: str,
    mhr_path: str,
    use_cuda: bool,
    use_detector: bool,
    detector_name: str,
    detector_path: str,
    use_segmentor: bool,
    segmentor_name: str,
    segmentor_path: str,
    use_fov: bool,
    fov_name: str,
    fov_path: str,
) -> Tuple[SAM3DBodyEstimator, str]:
    """Construct SAM3DBodyEstimator with optional detector/segmentor/FOV helpers."""
    device_str = pick_device(use_cuda=use_cuda)
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=checkpoint_path,
        device=device_str,
        mhr_path=mhr_path,
    )

    human_detector = None
    if use_detector:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(name=detector_name, device=device_str, path=detector_path)

    human_segmentor = None
    if use_segmentor:
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(name=segmentor_name, device=device_str, path=segmentor_path)

    fov_estimator = None
    if use_fov:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=fov_name, device=device_str, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )
    return estimator, device_str
