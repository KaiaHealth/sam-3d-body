# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streamlit app to explore the TorchScript MHR model without importing pymomentum."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch


st.set_page_config(page_title="MHR TorchScript Viewer", layout="wide")

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_TS_MODEL = ASSETS_DIR / "mhr_model.pt"


@st.cache_resource(show_spinner=False)
def load_torchscript_model(model_path: str, device: str) -> tuple[torch.jit.ScriptModule, np.ndarray]:
    """Load TorchScript MHR model and return it with mesh faces on CPU."""

    model = torch.jit.load(model_path)
    model.eval()
    model.to(torch.device(device))
    faces = model.character_torch.mesh.faces.to("cpu").numpy()
    return model, faces


def _init_coeffs(key: str, size: int, default: float = 0.0) -> None:
    if key not in st.session_state or len(st.session_state[key]) != size:
        st.session_state[key] = [default for _ in range(size)]


def coeff_editor(
    key: str,
    size: int,
    random_scale: float,
    value_range: tuple[float, float],
    help_text: str,
) -> list[float]:
    """Editable coefficient vector with per-index slider."""

    _init_coeffs(key, size)
    col_idx, col_val, col_actions = st.columns([1, 3, 2], vertical_alignment="bottom", width="stretch")
    with col_idx:
        idx = st.selectbox(
            "Index",
            options=list(range(size)),
            key=f"{key}_idx",
            help=help_text,
        )
    slider_key = f"{key}_slider_{idx}"

    with col_actions:
        btn_cols = st.columns(2)
        if btn_cols[0].button("Randomize", key=f"{key}_rand", type="tertiary", width="stretch"):
            rng = np.random.default_rng()
            st.session_state[key] = (random_scale * rng.standard_normal(size)).tolist()
        if btn_cols[1].button("Zero", key=f"{key}_zero", type="tertiary", width="stretch"):
            st.session_state[key] = [0.0 for _ in range(size)]

    current = float(st.session_state[key][idx])
    with col_val:
        new_value = st.slider(
            "Value",
            min_value=float(value_range[0]),
            max_value=float(value_range[1]),
            value=current,
            step=0.05,
            key=slider_key,
        )
    if new_value != current:
        st.session_state[key][idx] = float(new_value)
    return st.session_state[key]


def plot_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    container: st.delta_generator.DeltaGenerator | None = None,
    show_meshgrid_only: bool = False,
) -> None:
    """Render the mesh using Plotly."""

    target = container or st
    x, y, z = vertices.T
    intensity_raw = z
    intensity = (intensity_raw - intensity_raw.min()) / (np.ptp(intensity_raw) + 1e-8)
    face_indices = np.arange(faces.shape[0])
    face_centers = vertices[faces].mean(axis=1)
    hover_trace = go.Scatter3d(
        x=face_centers[:, 0],
        y=face_centers[:, 1],
        z=face_centers[:, 2],
        mode="markers",
        marker=dict(size=2, color="red", opacity=0.0),  # hide markers until hovered
        text=[str(idx) for idx in face_indices],
        hovertemplate="Face %{text}<extra></extra>",
        hoverlabel=dict(font_color="red"),
        showlegend=False,
        name="Face index",
    )
    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=1.0,
        flatshading=False,  # smooth shading to show surface nuance
        intensity=intensity,
        intensitymode="vertex",
        colorscale="Viridis",
        showscale=True,
        lighting=dict(
            ambient=0.35,
            diffuse=0.95,
            specular=0.4,
            roughness=0.35,
            fresnel=0.08,
        ),
        lightposition=dict(x=200, y=200, z=300),
    )
    if show_meshgrid_only:
        tri_edges = np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
            axis=0,
        )
        tri_edges = np.sort(tri_edges, axis=1)
        unique_edges = np.unique(tri_edges, axis=0)

        edge_points = vertices[unique_edges]  # (E, 2, 3)
        # Insert NaN separators so Plotly breaks the lines between edges.
        edge_points_with_breaks = np.concatenate(
            [edge_points, np.full((edge_points.shape[0], 1, 3), np.nan, dtype=edge_points.dtype)],
            axis=1,
        ).reshape(-1, 3)

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=edge_points_with_breaks[:, 0],
                    y=edge_points_with_breaks[:, 1],
                    z=edge_points_with_breaks[:, 2],
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0.6)", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                    name="Mesh grid",
                ),
                hover_trace,
            ]
        )
    else:
        fig = go.Figure(data=[mesh, hover_trace])
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#0b1220",
        ),
        paper_bgcolor="#0b1220",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    target.plotly_chart(fig, use_container_width=True, height=720)


def _as_tensor(values: Iterable[float], device: torch.device) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.float32, device=device).unsqueeze(0)


def mesh_to_ply(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """Convert mesh to ASCII PLY bytes for easy import into Blender."""

    vertex_lines = "\n".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    face_lines = "\n".join(f"3 {i} {j} {k}" for i, j, k in faces)
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(vertices)}",
            "property float x",
            "property float y",
            "property float z",
            f"element face {len(faces)}",
            "property list uchar int vertex_indices",
            "end_header",
        ]
    )
    ply_text = "\n".join([header, vertex_lines, face_lines])
    return ply_text.encode("utf-8")


def main() -> None:
    model_path = st.text_input("TorchScript model path", value=str(DEFAULT_TS_MODEL))
    model_path_obj = Path(model_path).expanduser()

    device_options = ["cpu"]
    if torch.cuda.is_available():
        device_options.append("cuda")
    device_choice = st.selectbox("Device", options=device_options, index=0)
    device = torch.device(device_choice)

    if not model_path_obj.exists():
        st.error(f"Model not found at {model_path_obj}. Download assets.zip and place mhr_model.pt in assets/.")
        st.info("Provide a valid model path to render the mesh.")
        return

    try:
        model, faces = load_torchscript_model(str(model_path_obj), device_choice)
    except Exception as exc:  # pragma: no cover - UI only
        st.exception(exc)
        st.info("Unable to load model.")
        return

    controls_col, view_col = st.columns(2)

    with controls_col:
        st.caption("Coefficients")
        with st.expander("Face expression (72 params)", expanded=True):
            face_expr_coeffs = coeff_editor(
                key="face_coeffs",
                size=72,
                random_scale=0.3,
                value_range=(-1.0, 1.0),
                help_text="Face expression parameters (std-normal latent; [-1, 1] covers ~99% of mass).",
            )
        with st.expander("Identity (45 params)", expanded=True):
            identity_coeffs = coeff_editor(
                key="identity_coeffs",
                size=45,
                random_scale=0.8,
                value_range=(-3.0, 3.0),
                help_text="Identity parameters (std-normal latent; [-3, 3] covers ~99% of mass).",
            )
        with st.expander("Model (204 params)", expanded=True):
            model_parameters = coeff_editor(
                key="model_parameters",
                size=204,
                random_scale=0.2,
                value_range=(-1.5, 1.5),
                help_text="Pose / scale parameters (std-normal latent assumption; adjust as needed).",
            )

        with st.spinner("Running TorchScript model..."):
            id_tensor = _as_tensor(identity_coeffs, device)
            model_tensor = _as_tensor(model_parameters, device)
            face_tensor = _as_tensor(face_expr_coeffs, device)
            with torch.no_grad():
                verts, _ = model(id_tensor, model_tensor, face_tensor)
            verts_np = verts[0].detach().cpu().numpy()
        st.session_state["last_mesh"] = verts_np
        st.session_state["last_faces"] = faces

    with view_col:
        st.caption("Mesh View")
        show_meshgrid_only = st.toggle(
            "Wireframe",
            value=False,
            help="Show only the mesh edges as a wireframe instead of the shaded surface.",
        )
        last_mesh = st.session_state.get("last_mesh")
        last_faces = st.session_state.get("last_faces", faces if "faces" in locals() else None)
        if last_mesh is not None and last_faces is not None:
            plot_mesh(last_mesh, last_faces, container=view_col, show_meshgrid_only=show_meshgrid_only)
            ply_bytes = mesh_to_ply(last_mesh, last_faces)
            st.download_button(
                "Download mesh as PLY",
                data=ply_bytes,
                file_name="mhr_mesh.ply",
                mime="application/octet-stream",
                help="Imports directly in Blender via File → Import → PLY.",
            )
        else:
            st.info("Adjust coefficients on the left to view the result here.")


if __name__ == "__main__":
    main()
