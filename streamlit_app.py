# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import plotly.graph_objects as go
import pyrootutils
import streamlit as st
import torch
from PIL import Image

MODEL_CHOICES = {
    "DINOv3-H+": {
        "label": "DINOv3-H+ (highest quality, heavier)",
        "repo_id": "facebook/sam-3d-body-dinov3",
        "default_dir": "checkpoints/sam-3d-body-dinov3",
    },
    "ViT-H": {
        "label": "ViT-H (lighter, very close quality)",
        "repo_id": "facebook/sam-3d-body-vith",
        "default_dir": "checkpoints/sam-3d-body-vith",
    },
}

# Ensure repo root is on the path, matching demo.py behavior
root = pyrootutils.setup_root(search_from=__file__, indicator=[".git", "pyproject.toml", ".sl"], pythonpath=True)

from tools.vis_utils import visualize_sample_together
from pose_inference.engine import PoseInferenceEngine
from kaia_commons.models import KeyPointConfigurationEnum, KeyPointConfigurationFactory
from sam_utils import build_estimator as build_estimator_impl
from sam_inference_engine import Sam3DBodyInferenceEngine


@st.cache_resource(show_spinner=True)
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
):
    """Load model and optional helper modules once."""
    return build_estimator_impl(
        checkpoint_path=checkpoint_path,
        mhr_path=mhr_path,
        use_cuda=use_cuda,
        use_detector=use_detector,
        detector_name=detector_name,
        detector_path=detector_path,
        use_segmentor=use_segmentor,
        segmentor_name=segmentor_name,
        segmentor_path=segmentor_path,
        use_fov=use_fov,
        fov_name=fov_name,
        fov_path=fov_path,
    )


def decode_image(uploaded_file) -> np.ndarray:
    """Convert an uploaded file to a CV2 BGR image."""
    data = uploaded_file.read()
    file_bytes = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def _homography_for_tilt(height: int, width: int, angle_degrees: float) -> np.ndarray:
    """Compute a homography that simulates rotating the camera around the X axis."""
    f = max(height, width)
    cx = width / 2.0
    cy = height / 2.0
    K = np.array(
        [
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    radians = np.radians(angle_degrees)
    c = np.cos(radians)
    s = np.sin(radians)
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )
    H = K @ R @ np.linalg.inv(K)

    center = np.array([cx, cy, 1.0], dtype=np.float32)
    proj = H @ center
    proj /= proj[2]
    delta_x = cx - proj[0]
    delta_y = cy - proj[1]
    T = np.array(
        [
            [1.0, 0.0, delta_x],
            [0.0, 1.0, delta_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return (T @ H).astype(np.float32)


def tilt_image_with_homography(img_bgr: np.ndarray, angle_degrees: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply a perspective warp that pitches the camera up/down while keeping image size."""
    if abs(angle_degrees) < 1e-3:
        return img_bgr, None
    h, w = img_bgr.shape[:2]
    H = _homography_for_tilt(h, w, angle_degrees)
    tilted = cv2.warpPerspective(
        img_bgr,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return tilted, H


def warp_image_to_shape(img: np.ndarray, homography: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Warp an image back using the inverse homography."""
    if homography is None:
        return img
    target_h, target_w = target_shape[:2]
    inv_H = np.linalg.inv(homography)
    return cv2.warpPerspective(
        img,
        inv_H,
        (target_w, target_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


@st.cache_resource(show_spinner=True)
def build_pose_engine(
    device: str,
    model_name: str = "2.5.6.0",
):
    """Load PoseInferenceEngine from KCCD once."""
    engine = PoseInferenceEngine.init_from_kccd(device=device, name=model_name)
    kp_config = engine.posemodel.keypoint_configuration
    return engine, kp_config


def body_dict_to_array(body: Dict[str, Dict[str, float]], kp_config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert PoseInferenceEngine body dict to arrays aligned with a KeyPointConfiguration.
    Returns (keypoints, mask) where keypoints are normalized and mask indicates presence.
    """
    kp_arr = np.full((len(kp_config), 2), np.nan, dtype=np.float32)
    kp_mask = np.zeros((len(kp_config),), dtype=bool)

    for idx, name in enumerate(kp_config.keypoint_names):
        entry = body.get(name)
        if entry and "x" in entry and "y" in entry:
            kp_arr[idx, 0] = float(entry["x"])
            kp_arr[idx, 1] = float(entry["y"])
            kp_mask[idx] = True
    return kp_arr, kp_mask


def draw_keypoints_overlay(
    base_img_bgr: np.ndarray,
    pose_kps_norm: np.ndarray,
    pose_mask: np.ndarray,
    sam_kps: np.ndarray,
    sam_mask: np.ndarray,
    kp_config,
) -> np.ndarray:
    """
    Draw PoseInferenceEngine keypoints (green) and SAM-mapped mesh keypoints (blue) on the same image.
    """
    img_h, img_w = base_img_bgr.shape[:2]
    canvas = base_img_bgr.copy()
    connections = np.asarray(kp_config.keypoint_connections)

    def draw_set(points_xy, mask, color):
        for a, b in connections:
            if mask[a] and mask[b] and np.isfinite(points_xy[[a, b]]).all():
                pa = tuple(np.round(points_xy[a]).astype(int))
                pb = tuple(np.round(points_xy[b]).astype(int))
                cv2.line(canvas, pa, pb, color, 2, cv2.LINE_AA)
        for idx, pt in enumerate(points_xy):
            if mask[idx] and np.isfinite(pt).all():
                cv2.circle(canvas, tuple(np.round(pt).astype(int)), 4, color, -1, lineType=cv2.LINE_AA)

    # PoseInferenceEngine outputs are normalized: x in [0,0.75], y in [0,1] relative to height
    pose_pixels = np.stack(
        [
            pose_kps_norm[:, 0] * img_h,
            pose_kps_norm[:, 1] * img_h,
        ],
        axis=1,
    )

    draw_set(sam_kps, sam_mask, (255, 0, 0))  # Blue for SAM mesh
    draw_set(pose_pixels, pose_mask, (0, 200, 0))  # Green for PoseInferenceEngine
    return canvas


def build_plotly_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    cam_t: Optional[np.ndarray] = None,
    mesh_color: str = "#c5d9ff",
):
    """Create a Plotly Mesh3d figure from predicted vertices."""
    verts = np.asarray(vertices, dtype=np.float32)
    if cam_t is not None:
        verts = verts + np.asarray(cam_t, dtype=np.float32).reshape(1, 3)
    x, y, z = verts.T
    i, j, k = faces.T

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color=mesh_color,
                opacity=1.0,
                flatshading=False,
                lighting=dict(
                    ambient=0.65,
                    diffuse=0.85,
                    fresnel=0.1,
                    specular=0.6,
                    roughness=0.3,
                ),
                lightposition=dict(x=0, y=-200, z=100),
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            bgcolor="rgba(15,15,30,1)",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_mask_overlay(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create an RGB overlay that highlights a binary mask on top of the image."""
    mask_2d = np.asarray(mask).squeeze()
    if mask_2d.dtype != np.uint8:
        mask_2d = (mask_2d > 0.5).astype(np.uint8)

    overlay = img_bgr.copy()
    mask_bool = mask_2d.astype(bool)
    if np.any(mask_bool):
        highlight_color = np.array([0, 165, 255], dtype=np.uint8)  # Orange tone in BGR
        blended = 0.6 * overlay[mask_bool].astype(np.float32) + 0.4 * highlight_color.astype(np.float32)
        overlay[mask_bool] = blended.astype(np.uint8)

    mask_uint8 = (mask_2d > 0).astype(np.uint8) * 255
    mask_edges = cv2.Canny(mask_uint8, 75, 150)
    overlay[mask_edges > 0] = (0, 0, 255)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=True, persist=True)
def process_image(estimator: SAM3DBodyEstimator, img_bgr: np.ndarray, bbox_thresh: float, use_mask: bool):
    """Run model inference on a single image."""
    outputs = estimator.process_one_image(img_bgr, bbox_thr=bbox_thresh, use_mask=use_mask)
    return outputs


def render_and_display(
    estimator: SAM3DBodyEstimator,
    images: List[Tuple[str, np.ndarray]],
    bbox_thresh: float,
    use_mask: bool,
    show_mesh: bool = False,
    show_mask: bool = False,
):
    for name, img_bgr in images:
        if img_bgr is None:
            st.warning(f"Could not read image {name}.")
            continue

        outputs = process_image(
            estimator=estimator,
            img_bgr=img_bgr,
            bbox_thresh=bbox_thresh,
            use_mask=use_mask,
        )
        st.json(outputs)
        if len(outputs) == 0:
            st.info(f"No humans detected in {name}.")
            continue

        render_bgr = visualize_sample_together(img_bgr, outputs, estimator.faces)
        render_bgr = render_bgr.astype(np.uint8)
        render_rgb = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2RGB)
        st.image(render_rgb, caption=f"Result: {name}")

        if show_mesh:
            tabs = st.tabs([f"Mesh {i+1}" for i in range(len(outputs))])
            for tab, prediction in zip(tabs, outputs):
                with tab:
                    fig = build_plotly_mesh(
                        vertices=prediction["pred_vertices"],
                        faces=estimator.faces,
                        cam_t=prediction.get("pred_cam_t"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        if show_mask:
            has_masks = any(prediction.get("mask") is not None for prediction in outputs)
            if not has_masks:
                st.info("Masks not available for this image. Enable SAM2 mask generation to visualize them.")
                continue

            mask_tabs = st.tabs([f"Mask {i+1}" for i in range(len(outputs))])
            for tab, prediction in zip(mask_tabs, outputs):
                with tab:
                    mask = prediction.get("mask")
                    if mask is None:
                        st.info("Mask missing for this detection.")
                        continue
                    overlay_rgb = build_mask_overlay(img_bgr, mask)
                    tab.image(overlay_rgb, caption="Mask overlay", width=640)


def download_checkpoint(repo_id: str, out_dir: str, token: str):
    """Download a checkpoint + assets from Hugging Face."""
    from huggingface_hub import snapshot_download

    os.makedirs(out_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=out_dir,
        local_dir_use_symlinks=False,
        token=token or None,
    )  # type: ignore[arg-type]


def main():
    st.set_page_config(page_title="SAM 3D Body Demo", layout="wide")
    st.title("SAM 3D Body – Streamlit Demo")
    st.markdown(
        "Upload one or more images and run the SAM 3D Body demo in the browser. "
        "You need local checkpoints and optional helper weights (detector/SAM2/FOV)."
    )

    # Initialize session state
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "image_name" not in st.session_state:
        st.session_state.image_name = None

    with st.sidebar:
        st.header("Model paths")
        model_choice = st.selectbox(
            "Choose model",
            list(MODEL_CHOICES.keys()),
            format_func=lambda x: MODEL_CHOICES[x]["label"],
        )
        model_info = MODEL_CHOICES[model_choice]
        default_dir = model_info["default_dir"]
        checkpoint_path = st.text_input(
            "Checkpoint (.ckpt)",
            value=os.path.join(default_dir, "model.ckpt"),
        )
        mhr_path = st.text_input(
            "MHR assets (mhr_model.pt)",
            value=os.path.join(default_dir, "assets/mhr_model.pt"),
        )
        hf_token = st.text_input(
            "Hugging Face token (optional if you already ran `huggingface-cli login`)",
            value=os.environ.get("HF_TOKEN", ""),
            type="password",
        )

        if st.button("Download selected model"):
            with st.spinner(f"Downloading {model_info['repo_id']}..."):
                try:
                    download_checkpoint(
                        repo_id=model_info["repo_id"],
                        out_dir=default_dir,
                        token=hf_token,
                    )
                    st.success(f"Downloaded to {default_dir}")
                except Exception as exc:
                    st.error(f"Download failed: {exc}")

        st.header("Runtime")
        use_cuda = st.checkbox("Use CUDA if available", value=torch.cuda.is_available())
        bbox_thresh = st.slider("BBox threshold", min_value=0.05, max_value=1.0, value=0.8, step=0.05)
        show_mesh = st.checkbox(
            "Show interactive 3D mesh",
            value=False,
            help="Adds a Plotly-based view of the predicted mesh for each detection.",
        )
        show_mask = st.checkbox(
            "Show SAM2 masks",
            value=False,
            help="Visualize segmentation overlays for each detected person (requires SAM2).",
        )

        st.header("Optional modules")
        use_detector = st.checkbox("Load detector (ViTDet)", value=True)
        detector_path = st.text_input(
            "Detector path (optional)",
            value=os.environ.get("SAM3D_DETECTOR_PATH", ""),
            help="Leave empty to download the default ViTDet checkpoint.",
        )

        use_segmentor = st.checkbox("Load SAM2 for mask conditioning", value=False)
        segmentor_path = st.text_input(
            "SAM2 repo path",
            value="/home/ec2-user/sam2",
            help="Path to SAM2 repo that contains checkpoints/sam2.1_hiera_large.pt",
        )

        use_fov = st.checkbox("Load FOV estimator (MoGe2)", value=False)
        fov_path = st.text_input(
            "MoGe path",
            value=os.environ.get("SAM3D_FOV_PATH", ""),
            help="Leave empty to pull from Hugging Face (requires network access).",
        )

        manual_tilt_override = st.checkbox(
            "Rotate image to compensate tilt",
            value=False,
            help="Rotate the uploaded image before inference to emulate a level camera.",
        )
        manual_tilt_degrees = st.slider(
            "Tilt compensation (degrees)",
            min_value=-60,
            max_value=60,
            value=0,
            step=1,
            disabled=not manual_tilt_override,
        )

        st.header("Pose inference overlay")
        enable_pose_overlay = st.checkbox(
            "Run PoseInferenceEngine and compare keypoints",
            value=False,
            help="Requires kaia-pose-core/commons and kaia-pose-core/engines/pose-inference to be present.",
        )
        pose_model_name = st.text_input(
            "Pose model name from KCCD",
            value="2.5.8.0",
            help="Model name/tag to fetch from Kaia Commons Central Database",
            disabled=not enable_pose_overlay,
        )
        pose_device = st.selectbox(
            "Pose model device",
            options=["cpu", "cuda", "mps"],
            index=1 if torch.cuda.is_available() else 0,
            disabled=not enable_pose_overlay,
        )

        st.caption("Tip: disable optional modules if you only want full-image inference " "without masks or FOV estimation.")

    # Main content area
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        help="Upload a single image for 3D body estimation",
    )

    if uploaded_file is None:
        st.stop()

    # Load image
    img_bgr = decode_image(uploaded_file)
    original_image_bgr = img_bgr.copy()
    tilt_transform = None
    if manual_tilt_override and abs(manual_tilt_degrees) > 1e-3:
        img_bgr, tilt_transform = tilt_image_with_homography(img_bgr, -manual_tilt_degrees)
    st.session_state.current_image = img_bgr
    st.session_state.image_name = uploaded_file.name

    if manual_tilt_override and tilt_transform is not None:
        before_col, after_col = st.columns(2)
        before_col.image(
            cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB),
            caption="Original (tilted)",
            use_container_width=True,
        )
        after_col.image(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            caption="Leveled input (used for inference)",
            use_container_width=True,
        )

    if st.session_state.current_image is not None:
        run_clicked = st.button("▶️ Run Inference", type="primary")

    if not run_clicked:
        st.stop()

    # Check checkpoints exist
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        ckpt_dir = os.path.dirname(checkpoint_path)
        try:
            with st.spinner("Checkpoint not found, downloading selected model..."):
                download_checkpoint(
                    repo_id=model_info["repo_id"],
                    out_dir=ckpt_dir,
                    token=hf_token,
                )
        except Exception as exc:
            st.error(f"Checkpoint path does not exist and auto-download failed: {exc}")
            st.stop()

    if not mhr_path or not os.path.exists(mhr_path):
        mhr_dir = os.path.dirname(os.path.dirname(mhr_path))
        try:
            with st.spinner("MHR asset not found, downloading selected model..."):
                download_checkpoint(
                    repo_id=model_info["repo_id"],
                    out_dir=mhr_dir,
                    token=hf_token,
                )
        except Exception as exc:
            st.error(f"MHR asset path does not exist and auto-download failed: {exc}")
            st.stop()

    if use_segmentor and not segmentor_path:
        st.error("SAM2 path is required when loading the segmentor.")
        st.stop()

    # Load model
    with st.spinner("Loading model and helpers..."):
        estimator, device_str = build_estimator(
            checkpoint_path=checkpoint_path,
            mhr_path=mhr_path,
            model_config_path=None,
            use_cuda=use_cuda,
            use_detector=use_detector,
            detector_name="vitdet",
            detector_path=detector_path,
            use_segmentor=use_segmentor,
            segmentor_name="sam2",
            segmentor_path=segmentor_path,
            use_fov=use_fov,
            fov_name="moge2",
            fov_path=fov_path,
        )
    st.success(f"✅ Model loaded. Running on {device_str}.")

    # Run inference
    with st.spinner("Running inference..."):
        sam_engine = Sam3DBodyInferenceEngine(
            estimator,
            bbox_thresh=bbox_thresh,
            use_mask=use_segmentor,
        )
        outputs = sam_engine.run_single(st.session_state.current_image)

    if len(outputs) == 0:
        st.warning("No humans detected in the image.")
    else:
        st.success(f"✅ Detected {len(outputs)} person(s)")

        # Visualize results
        render_bgr = visualize_sample_together(st.session_state.current_image, outputs, estimator.faces)
        render_bgr = render_bgr.astype(np.uint8)
        render_rgb_leveled = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2RGB)
        if tilt_transform is None:
            st.image(render_rgb_leveled, caption="Result with 3D mesh overlay", use_container_width=True)
        else:
            render_bgr_warped = warp_image_to_shape(render_bgr, tilt_transform, original_image_bgr.shape)
            render_rgb_warped = cv2.cvtColor(render_bgr_warped, cv2.COLOR_BGR2RGB)
            leveled_col, orig_col = st.columns(2)
            leveled_col.image(
                render_rgb_leveled,
                caption="Result on leveled image",
                use_container_width=True,
            )
            orig_col.image(
                render_rgb_warped,
                caption="Result reprojected to original orientation",
                use_container_width=True,
            )

        # Show 3D mesh if enabled
        if show_mesh:
            st.subheader("3D Mesh Visualization")
            tabs = st.tabs([f"Person {i+1}" for i in range(len(outputs))])
            for tab, prediction in zip(tabs, outputs):
                with tab:
                    fig = build_plotly_mesh(
                        vertices=prediction["pred_vertices"],
                        faces=estimator.faces,
                        cam_t=prediction.get("pred_cam_t"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Show masks if enabled
        if show_mask:
            has_masks = any(prediction.get("mask") is not None for prediction in outputs)
            if has_masks:
                st.subheader("Segmentation Masks")
                mask_tabs = st.tabs([f"Mask {i+1}" for i in range(len(outputs))])
                for tab, prediction in zip(mask_tabs, outputs):
                    with tab:
                        mask = prediction.get("mask")
                        if mask is not None:
                            overlay_rgb = build_mask_overlay(st.session_state.current_image, mask)
                            if tilt_transform is not None:
                                overlay_rgb = warp_image_to_shape(
                                    overlay_rgb, tilt_transform, original_image_bgr.shape
                                )
                            st.image(overlay_rgb, caption="Mask overlay", use_container_width=True)
            else:
                st.info("Masks not available. Enable SAM2 to visualize them.")

        # PoseInferenceEngine overlay
        if enable_pose_overlay:
            st.subheader("Keypoint comparison (PoseInferenceEngine vs SAM mesh)")
            if not pose_model_name:
                st.error("Provide a pose model name to run PoseInferenceEngine.")
            else:
                with st.spinner("Loading PoseInferenceEngine from KCCD..."):
                    try:
                        pose_engine, kp_config = build_pose_engine(
                            device=pose_device,
                            model_name=pose_model_name,
                        )
                    except Exception as exc:
                        st.error(f"Failed to load PoseInferenceEngine: {exc}")
                        pose_engine = None

                if pose_engine is not None:
                    with st.spinner("Running PoseInferenceEngine on the image..."):
                        try:
                            pose_body = pose_engine.perform_on_image(
                                cv2.cvtColor(st.session_state.current_image, cv2.COLOR_BGR2RGB)
                            )
                        except Exception as exc:
                            st.error(f"PoseInferenceEngine failed: {exc}")
                            pose_body = None

                if pose_body is None:
                    st.info("PoseInferenceEngine did not return any keypoints for this image.")
                else:
                    pose_kps_norm, pose_mask = body_dict_to_array(pose_body, kp_config)
                    if not pose_mask.any():
                        st.info("PoseInferenceEngine returned empty keypoints.")
                    else:
                        target_idx = 0
                        if len(outputs) > 1:
                            target_idx = st.number_input(
                                "Pick SAM detection index for comparison (0-based)",
                                min_value=0,
                                max_value=len(outputs) - 1,
                                value=0,
                            )

                        target_prediction = outputs[target_idx]
                        sam_kps = target_prediction.get("kaia23_keypoints")
                        sam_mask = target_prediction.get("kaia23_mask")
                        overlay_bgr = draw_keypoints_overlay(
                            st.session_state.current_image,
                            pose_kps_norm=pose_kps_norm,
                            pose_mask=pose_mask,
                            sam_kps=sam_kps,
                            sam_mask=sam_mask,
                            kp_config=kp_config,
                        )
                        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                        if tilt_transform is not None:
                            overlay_rgb = warp_image_to_shape(
                                overlay_rgb, tilt_transform, original_image_bgr.shape
                            )
                        st.image(
                            overlay_rgb,
                            caption="PoseInferenceEngine (green) vs SAM mesh vertices (blue)",
                            width=640,
                        )


if __name__ == "__main__":
    main()
