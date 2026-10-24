"""
CodeAlpha AI Internship - Task 3: Vision Lab

Object detection and scene analysis built on YOLOv8 (Ultralytics).

Beyond drawing bounding boxes, the app turns raw detections into something
a person can act on:

  * Detections    - annotated image with adjustable thresholds
  * Privacy blur  - blurs selected classes (for example people) before sharing
  * Density map   - heatmap of where detected objects are concentrated
  * Analysis      - plain-language scene summary, per-class counts, position
                    grid, and the full detection table (CSV / PNG export)

Only libraries already listed in requirements.txt are used (streamlit,
ultralytics, opencv-python-headless, pillow, numpy; pandas ships with
streamlit), so Task 1 and Task 2 are unaffected.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_FILE = "yolov8n.pt"
MAX_IMAGE_SIDE = 1280  # larger uploads are downscaled to keep memory low
INPUT_SIZES = (320, 480, 640, 960)
DEFAULT_PRIVACY_CLASSES = ("person",)

TABLE_COLUMNS = ["Object", "Confidence", "X1", "Y1", "X2", "Y2", "Width", "Height", "Area %"]

# Rough scene categories inferred from which COCO classes appear together.
SCENE_HINTS = {
    "Street or traffic": {
        "car", "truck", "bus", "motorcycle", "bicycle", "traffic light",
        "stop sign", "fire hydrant", "parking meter",
    },
    "Living space": {
        "couch", "chair", "bed", "tv", "potted plant", "vase", "clock", "remote",
    },
    "Kitchen or dining": {
        "oven", "microwave", "refrigerator", "sink", "toaster", "bowl", "cup",
        "fork", "knife", "spoon", "bottle", "wine glass", "dining table",
    },
    "Workspace": {"laptop", "keyboard", "mouse", "cell phone", "book", "tv"},
    "Sports or outdoor activity": {
        "sports ball", "tennis racket", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "skis", "snowboard", "frisbee", "kite",
    },
    "Animals or nature": {
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe",
    },
}

IRREGULAR_PLURALS = {
    "person": "people",
    "knife": "knives",
    "mouse": "mice",
    "sheep": "sheep",
    "skis": "skis",
    "scissors": "scissors",
}


# ---------------------------------------------------------------------------
# Model and image helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model() -> YOLO:
    """Load the bundled nano model; fall back to Ultralytics' own download."""
    local_weights = Path(__file__).resolve().parent / MODEL_FILE
    return YOLO(str(local_weights) if local_weights.exists() else MODEL_FILE)


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes into an RGB array, honouring EXIF orientation."""
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
    return np.asarray(image)


def to_png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Detection table
# ---------------------------------------------------------------------------
def build_table(
    xyxy: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    names: dict,
    shape: tuple,
) -> pd.DataFrame:
    """Convert raw model output into a tidy, clipped, confidence-sorted table."""
    height, width = shape[:2]
    frame_area = float(height * width)
    rows = []
    for (x1, y1, x2, y2), confidence, class_id in zip(xyxy, confidences, class_ids):
        x1, x2 = (int(round(min(max(v, 0), width))) for v in (x1, x2))
        y1, y2 = (int(round(min(max(v, 0), height))) for v in (y1, y2))
        box_w, box_h = x2 - x1, y2 - y1
        rows.append(
            {
                "Object": names[int(class_id)],
                "Confidence": round(float(confidence), 4),
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2,
                "Width": box_w,
                "Height": box_h,
                "Area %": round(100.0 * box_w * box_h / frame_area, 2),
            }
        )
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    return table.sort_values("Confidence", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner="Analysing image...", max_entries=8)
def run_detection(_model, data, confidence, iou, max_det, image_size, class_ids):
    """Run YOLOv8 once per (image, settings) combination and cache the result."""
    rgb = decode_image(data)
    # Ultralytics treats numpy input as BGR, so convert before and after.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    started = time.perf_counter()
    result = _model.predict(
        bgr,
        conf=confidence,
        iou=iou,
        max_det=max_det,
        imgsz=image_size,
        classes=list(class_ids) or None,
        verbose=False,
    )[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    boxes = result.boxes
    table = build_table(
        boxes.xyxy.cpu().numpy(),
        boxes.conf.cpu().numpy(),
        boxes.cls.cpu().numpy().astype(int),
        _model.names,
        rgb.shape,
    )
    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    return rgb, annotated, table, elapsed_ms


# ---------------------------------------------------------------------------
# Image transformations built on the detections
# ---------------------------------------------------------------------------
def apply_privacy_blur(rgb: np.ndarray, table: pd.DataFrame, targets: list) -> np.ndarray:
    """Blur the bounding box of every detection whose class is in `targets`."""
    output = rgb.copy()
    for row in table[table["Object"].isin(targets)].itertuples():
        region = output[row.Y1:row.Y2, row.X1:row.X2]
        if region.shape[0] < 2 or region.shape[1] < 2:
            continue
        kernel = max(15, (min(region.shape[:2]) // 3) | 1)  # always odd
        output[row.Y1:row.Y2, row.X1:row.X2] = cv2.GaussianBlur(region, (kernel, kernel), 0)
    return output


def density_heatmap(rgb: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    """Overlay a confidence-weighted heatmap showing where objects cluster."""
    height, width = rgb.shape[:2]
    density = np.zeros((height, width), dtype=np.float32)
    for row in table.itertuples():
        density[row.Y1:row.Y2, row.X1:row.X2] += row.Confidence
    if density.max() <= 0:
        return rgb.copy()

    kernel = int(max(height, width) * 0.08) | 1
    density = cv2.GaussianBlur(density, (kernel, kernel), 0)
    normalised = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    coloured = cv2.cvtColor(cv2.applyColorMap(normalised, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    # Opacity follows density, so empty regions keep the original pixels.
    alpha = (normalised.astype(np.float32) / 255.0 * 0.65)[..., None]
    return (rgb * (1.0 - alpha) + coloured * alpha).astype(np.uint8)


# ---------------------------------------------------------------------------
# Plain-language analysis
# ---------------------------------------------------------------------------
def pluralise(name: str, count: int) -> str:
    if count == 1:
        return name
    if name in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[name]
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    return name + "s"


def human_join(items: list) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def summarise(table: pd.DataFrame) -> str:
    """Describe the detections in one or two readable sentences."""
    if table.empty:
        return "No objects were found with the current settings."

    counts = table["Object"].value_counts()
    shown = [f"{n} {pluralise(name, n)}" for name, n in counts.head(6).items()]
    total, categories = len(table), len(counts)
    noun = "object" if total == 1 else "objects"
    tail = " (plus smaller groups of other objects)" if categories > 6 else ""
    return (
        f"The image contains {total} {noun} across {categories} "
        f"{'category' if categories == 1 else 'categories'}: {human_join(shown)}{tail}."
    )


def guess_scene(table: pd.DataFrame):
    """Return the most likely scene label, or None when evidence is weak."""
    if table.empty:
        return None
    best_label, best_score = None, 0
    for label, members in SCENE_HINTS.items():
        score = int(table["Object"].isin(members).sum())
        if score > best_score:
            best_label, best_score = label, score
    return best_label if best_score >= 2 else None


def spatial_grid(table: pd.DataFrame, shape: tuple) -> pd.DataFrame:
    """Count object centres in a 3x3 grid over the frame."""
    height, width = shape[:2]
    grid = np.zeros((3, 3), dtype=int)
    for row in table.itertuples():
        centre_x, centre_y = (row.X1 + row.X2) / 2, (row.Y1 + row.Y2) / 2
        grid[min(int(3 * centre_y / height), 2), min(int(3 * centre_x / width), 2)] += 1
    return pd.DataFrame(grid, index=["Top", "Middle", "Bottom"], columns=["Left", "Centre", "Right"])


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def configure_page() -> None:
    st.set_page_config(page_title="Vision Lab | Object Detection", page_icon="👁️", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 10px;
            padding: 0.75rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_settings(names: dict) -> dict:
    """Render the sidebar controls and return the chosen settings."""
    name_to_id = {name: class_id for class_id, name in names.items()}
    with st.sidebar:
        st.header("Detection settings")
        confidence = st.slider(
            "Confidence threshold", 0.05, 0.95, 0.35, 0.05,
            help="Detections below this score are discarded. Raise it to cut false positives.",
        )
        iou = st.slider(
            "Overlap threshold (IoU)", 0.10, 0.90, 0.50, 0.05,
            help="Lower values remove more overlapping boxes around the same object.",
        )
        max_det = st.slider("Maximum detections", 1, 300, 100)
        image_size = st.select_slider(
            "Model input size", options=INPUT_SIZES, value=640,
            help="Larger sizes find smaller objects but run slower.",
        )
        chosen = st.multiselect(
            "Only detect these classes (optional)", options=sorted(name_to_id),
            help="Leave empty to detect all 80 COCO classes.",
        )
        st.divider()
        st.caption("Model: YOLOv8 nano, trained on the 80-class COCO dataset.")
    return {
        "confidence": confidence,
        "iou": iou,
        "max_det": max_det,
        "image_size": image_size,
        "class_ids": tuple(sorted(name_to_id[name] for name in chosen)),
    }


def read_source():
    """Return raw image bytes from an upload or the camera, or None."""
    mode = st.radio("Image source", ["Upload a file", "Use camera"], horizontal=True)
    if mode == "Upload a file":
        source = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    else:
        source = st.camera_input("Take a photo")
    return source.getvalue() if source is not None else None


def render_metrics(table: pd.DataFrame, elapsed_ms: float) -> None:
    mean_confidence = f"{table['Confidence'].mean():.0%}" if not table.empty else "n/a"
    columns = st.columns(4)
    columns[0].metric("Objects detected", len(table))
    columns[1].metric("Categories", table["Object"].nunique())
    columns[2].metric("Mean confidence", mean_confidence)
    columns[3].metric("Inference time", f"{elapsed_ms:.0f} ms")


def render_detections_tab(rgb: np.ndarray, annotated: np.ndarray) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(rgb)
    with right:
        st.subheader("Detected")
        st.image(annotated)
    st.caption(f"Frame size: {rgb.shape[1]} x {rgb.shape[0]} px")


def render_privacy_tab(rgb: np.ndarray, table: pd.DataFrame) -> None:
    st.write(
        "Blur selected object types before sharing a photo. The whole bounding "
        "box of each chosen class is blurred."
    )
    if table.empty:
        st.info("Nothing was detected, so there is nothing to blur.")
        return
    available = sorted(table["Object"].unique())
    defaults = [name for name in DEFAULT_PRIVACY_CLASSES if name in available]
    targets = st.multiselect("Classes to blur", options=available, default=defaults)
    blurred = apply_privacy_blur(rgb, table, targets)
    st.image(blurred)
    st.download_button(
        "Download blurred image", to_png_bytes(blurred), "privacy_blurred.png", "image/png"
    )


def render_density_tab(rgb: np.ndarray, table: pd.DataFrame) -> None:
    st.write(
        "Warmer colours mark areas where detected objects are concentrated, "
        "weighted by detection confidence."
    )
    if table.empty:
        st.info("Nothing was detected, so there is no density to show.")
        return
    st.image(density_heatmap(rgb, table))


def render_analysis_tab(rgb: np.ndarray, annotated: np.ndarray, table: pd.DataFrame) -> None:
    st.subheader("Scene summary")
    st.write(summarise(table))
    scene = guess_scene(table)
    if scene:
        st.caption(f"Likely scene: {scene} (inferred from which objects appear together).")

    if table.empty:
        st.info("Try lowering the confidence threshold in the sidebar.")
        return

    chart_col, grid_col = st.columns(2)
    with chart_col:
        st.subheader("Objects per class")
        st.bar_chart(table["Object"].value_counts())
    with grid_col:
        st.subheader("Where the objects are")
        st.dataframe(spatial_grid(table, rgb.shape))
        st.caption("Count of object centres in each part of the frame.")

    st.subheader("Detection table")
    st.dataframe(
        table,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
        hide_index=True,
    )

    csv_col, png_col = st.columns(2)
    csv_col.download_button(
        "Download table (CSV)", table.to_csv(index=False).encode("utf-8"),
        "detections.csv", "text/csv",
    )
    png_col.download_button(
        "Download annotated image (PNG)", to_png_bytes(annotated),
        "annotated.png", "image/png",
    )


def main() -> None:
    configure_page()
    st.title("Vision Lab")
    st.caption("Object detection and scene analysis with YOLOv8 | CodeAlpha AI Internship, Task 3")

    model = load_model()
    settings = sidebar_settings(model.names)
    data = read_source()

    if data is None:
        st.info(
            "Upload a photo or take one with your camera. Adjust the thresholds in the "
            "sidebar and the results update automatically."
        )
        return

    try:
        rgb, annotated, table, elapsed_ms = run_detection(model, data, **settings)
    except Exception as error:  # unreadable file, corrupt image, model failure
        st.error(f"Could not process this image: {error}")
        return

    render_metrics(table, elapsed_ms)
    tab_detect, tab_privacy, tab_density, tab_analysis = st.tabs(
        ["Detections", "Privacy blur", "Density map", "Analysis"]
    )
    with tab_detect:
        render_detections_tab(rgb, annotated)
    with tab_privacy:
        render_privacy_tab(rgb, table)
    with tab_density:
        render_density_tab(rgb, table)
    with tab_analysis:
        render_analysis_tab(rgb, annotated, table)


if __name__ == "__main__":
    main()
