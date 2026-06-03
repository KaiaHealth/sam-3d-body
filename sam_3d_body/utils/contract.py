"""Export SAM3DBodyEstimator output to the Lance annotation contract format.

Converts the per-person dict returned by ``SAM3DBodyEstimator.process_one_image``
to the ``mhr_params_data`` content shape expected by the visionai-data-pipeline:

    {
      "sam3d_pipeline_outputs": {"bbox", "cam_int", "pred_cam_t",
                                 "pred_keypoints_2d",
                                 "lhand_bbox"?, "rhand_bbox"?},
      "mhr_decoder_inputs":     {"global_rot", "body_pose_params",
                                 "skeleton_trans", "shape_params",
                                 "expr_params",
                                 "pred_pose_raw"?, "hand_pose_params"?},
      "mhr_decoder_outputs":    {"pred_keypoints_3d", "pred_vertices",
                                 "pred_joint_coords", "pred_global_rots"},
    }

This module has no dependencies outside the standard library and numpy.
"""

from typing import Any

# MHR TorchScript model_parameters (204,) layout:
#   [global_trans*10 (3) | global_rot_ZYX (3) | body_pose_XYZ (130)] ++ skeleton_trans (68)
# body_pose_params from SAM3D head is (133,): 130 body + 3 jaw (always zero).
_BODY_POSE_SIZE = 130


def _flat(v: Any) -> list[float]:
    """Flatten a numpy array or nested list to a plain Python float list."""
    try:
        import numpy as np

        return np.asarray(v, dtype=float).reshape(-1).tolist()
    except ImportError:
        acc: list[float] = []
        _flatten_into(v, acc)
        return acc


def _flatten_into(v: Any, acc: list) -> None:
    try:
        for item in v:
            _flatten_into(item, acc)
    except TypeError:
        acc.append(float(v))


def sam3d_output_to_contract(sam3d_output: dict) -> dict:
    """Convert one element of ``SAM3DBodyEstimator.process_one_image()`` output
    to the Lance ``mhr_params_data`` annotation content dict.

    Args:
        sam3d_output:   Per-person dict from ``process_one_image()``.
                        Values may be numpy arrays or plain lists.
                        Must include ``cam_int`` and ``skeleton_trans``.

    Returns:
        Annotation content dict matching the Lance ``mhr_params_data`` contract.
    """
    # ── sam3d_pipeline_outputs ─────────────────────────────────────────────
    pipeline: dict = {
        "bbox": _flat(sam3d_output["bbox"]),
        "cam_int": _flat(sam3d_output["cam_int"]),
        "pred_cam_t": _flat(sam3d_output["pred_cam_t"]),
        "pred_keypoints_2d": _flat(sam3d_output["pred_keypoints_2d"]),
    }
    if sam3d_output.get("lhand_bbox") is not None:
        pipeline["lhand_bbox"] = _flat(sam3d_output["lhand_bbox"])
    if sam3d_output.get("rhand_bbox") is not None:
        pipeline["rhand_bbox"] = _flat(sam3d_output["rhand_bbox"])

    # ── mhr_decoder_inputs ─────────────────────────────────────────────────
    inputs: dict = {
        "global_rot": _flat(sam3d_output["global_rot"]),
        "body_pose_params": _flat(sam3d_output["body_pose_params"])[:_BODY_POSE_SIZE],
        "skeleton_trans": _flat(sam3d_output["skeleton_trans"]),
        "shape_params": _flat(sam3d_output["shape_params"]),
        "expr_params": _flat(sam3d_output["expr_params"]),
    }
    if sam3d_output.get("pred_pose_raw") is not None:
        inputs["pred_pose_raw"] = _flat(sam3d_output["pred_pose_raw"])
    if sam3d_output.get("hand_pose_params") is not None:
        inputs["hand_pose_params"] = _flat(sam3d_output["hand_pose_params"])

    # ── mhr_decoder_outputs ────────────────────────────────────────────────
    outputs: dict = {
        "pred_keypoints_3d": _flat(sam3d_output["pred_keypoints_3d"]),
        "pred_vertices": _flat(sam3d_output["pred_vertices"]),
        "pred_joint_coords": _flat(sam3d_output["pred_joint_coords"]),
        "pred_global_rots": _flat(sam3d_output["pred_global_rots"]),
    }

    return {
        "sam3d_pipeline_outputs": pipeline,
        "mhr_decoder_inputs": inputs,
        "mhr_decoder_outputs": outputs,
    }
