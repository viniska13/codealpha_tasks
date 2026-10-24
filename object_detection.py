"""
CodeAlpha AI Internship - Task 3: Vision Lab

Object detection and scene analysis built on YOLOv8 (Ultralytics), presented
in a black, silver and lilac-neon interface.

Beyond drawing bounding boxes, the app turns raw detections into something
a person can act on:

  * Detections    - HUD-style annotated image with adjustable thresholds
  * Privacy blur  - blurs selected classes (for example people) before sharing
  * Density map   - heatmap of where detected objects are concentrated
  * Analysis      - plain-language scene summary, per-class counts, position
                    grid, and the full detection table (CSV / PNG export)

Only libraries already listed in requirements.txt are used (streamlit,
ultralytics, opencv-python-headless, pillow, numpy; pandas ships with
streamlit). All styling is injected by this script alone, so Task 1 and
Task 2 are unaffected.
"""

from __future__ import annotations

import io
import time
from html import escape
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

# Annotation colours (RGB)
SILVER = (192, 192, 192)
SILVER_DIM = (120, 120, 128)
BLACK = (0, 0, 0)

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


def _build_heat_palette() -> np.ndarray:
    """256-step palette: black -> deep violet -> lilac -> pale lilac."""
    stops = np.array([0.0, 0.35, 0.70, 1.0])
    colours = np.array([[0, 0, 0], [70, 30, 130], [170, 120, 235], [214, 190, 255]], dtype=float)
    positions = np.linspace(0.0, 1.0, 256)
    channels = [np.interp(positions, stops, colours[:, c]) for c in range(3)]
    return np.stack(channels, axis=1).astype(np.uint8)


HEAT_PALETTE = _build_heat_palette()

# ---------------------------------------------------------------------------
# Theme (black background, silver text and controls, lilac neon accents)
# ---------------------------------------------------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&display=swap');

:root {
    --bg: #000000;
    --panel: #0b0b10;
    --silver: #c0c0c0;
    --silver-dim: #8a8a94;
    --line: rgba(192, 192, 192, 0.28);
    --lilac: #c8a2ff;
    --glow: 0 0 6px #c8a2ff, 0 0 18px rgba(200, 162, 255, 0.65);
}

:root, .stApp { color-scheme: dark; }

.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: var(--bg) !important;
}
[data-testid="stSidebar"] {
    background: #07070b !important;
    border-right: 1px solid var(--line);
}
[data-testid="stHeader"] * { color: var(--silver) !important; }

.stApp, .stApp p, .stApp label, .stApp li, .stApp span, .stApp small, .stApp div {
    color: var(--silver);
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp h1 *, .stApp h2 *, .stApp h3 *, .stApp h4 * {
    color: var(--lilac) !important;
    text-shadow: var(--glow);
    letter-spacing: 0.04em;
}
.stApp h2, .stApp h3 { font-family: 'Orbitron', sans-serif; font-weight: 500; font-size: 1.15rem; }
[data-testid="stCaptionContainer"], [data-testid="stImageCaption"] { color: var(--silver-dim) !important; }
hr { border-color: var(--line) !important; }

/* Hero */
.vl-hero { padding: 0.4rem 0 1rem 0; }
.vl-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: 2.8rem;
    letter-spacing: 0.22em;
    color: var(--lilac) !important;
    text-shadow: var(--glow);
    margin: 0;
    animation: vl-pulse 3.5s ease-in-out infinite;
}
.vl-sub { color: var(--silver) !important; margin: 0.2rem 0 0.8rem 0; letter-spacing: 0.08em; }
.vl-scan {
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--lilac), var(--silver), transparent);
    background-size: 200% 100%;
    animation: vl-sweep 4s linear infinite;
    box-shadow: 0 0 10px rgba(200, 162, 255, 0.6);
}
@keyframes vl-sweep { from { background-position: 200% 0; } to { background-position: -200% 0; } }
@keyframes vl-pulse {
    0%, 100% { text-shadow: 0 0 6px #c8a2ff, 0 0 16px rgba(200, 162, 255, 0.55); }
    50% { text-shadow: 0 0 10px #c8a2ff, 0 0 28px rgba(200, 162, 255, 0.85); }
}

/* Metric cards */
.vl-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.8rem; margin: 0.6rem 0 1rem 0; }
.vl-card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 0.8rem 1rem; }
.vl-card-label { color: var(--silver-dim) !important; font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase; }
.vl-card-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--lilac) !important;
    text-shadow: var(--glow);
}

/* Empty state */
.vl-panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.2rem; margin-top: 0.8rem; }
.vl-panel b { color: var(--lilac) !important; text-shadow: var(--glow); font-weight: 500; }

/* Class bars */
.vl-bar-row { display: grid; grid-template-columns: 7.5rem 1fr 2.2rem; align-items: center; gap: 0.6rem; margin: 0.45rem 0; }
.vl-bar-name { color: var(--silver) !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.vl-bar-track { background: rgba(192, 192, 192, 0.12); border-radius: 6px; height: 12px; overflow: hidden; }
.vl-bar-fill { height: 100%; background: linear-gradient(90deg, var(--silver), var(--lilac)); box-shadow: 0 0 10px rgba(200, 162, 255, 0.55); }
.vl-bar-count { color: var(--lilac) !important; text-shadow: var(--glow); text-align: right; }

/* Tables */
.vl-table-wrap { max-height: 360px; overflow: auto; border: 1px solid var(--line); border-radius: 10px; }
.vl-table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
.vl-table th {
    position: sticky; top: 0; background: #0f0f16;
    color: var(--lilac) !important; text-shadow: var(--glow);
    text-align: left; padding: 0.5rem 0.7rem; border-bottom: 1px solid var(--line);
    font-weight: 500; letter-spacing: 0.06em;
}
.vl-table td { padding: 0.4rem 0.7rem; border-bottom: 1px solid rgba(192, 192, 192, 0.1); color: var(--silver) !important; }
.vl-conf { display: flex; align-items: center; gap: 0.5rem; min-width: 120px; }
.vl-conf-track { flex: 1; background: rgba(192, 192, 192, 0.14); height: 6px; border-radius: 4px; overflow: hidden; }
.vl-conf-fill { height: 100%; background: linear-gradient(90deg, var(--silver), var(--lilac)); }
.vl-grid td, .vl-grid th { text-align: center; }
.vl-grid td { font-size: 1.05rem; }
.vl-grid td.hot { color: var(--lilac) !important; text-shadow: var(--glow); font-weight: 700; }

/* Tabs */
button[data-baseweb="tab"], button[data-baseweb="tab"] * { color: var(--silver) !important; background: transparent !important; }
button[data-baseweb="tab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] * {
    color: var(--lilac) !important; text-shadow: var(--glow);
}
[data-baseweb="tab-highlight"] { background-color: var(--lilac) !important; box-shadow: var(--glow); }
[data-baseweb="tab-border"] { background-color: var(--line) !important; }

/* Inputs */
[data-testid="stSlider"], [data-testid="stRadio"] { filter: hue-rotate(266deg) brightness(1.08); }
[data-baseweb="select"] > div, [data-baseweb="input"] > div {
    background: var(--panel) !important; border: 1px solid var(--line) !important; color: var(--silver) !important;
}
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background: #0b0b10 !important; }
[data-baseweb="popover"] li { color: var(--silver) !important; }
span[data-baseweb="tag"] {
    background: #14141c !important; border: 1px solid var(--lilac) !important;
    color: var(--silver) !important; box-shadow: 0 0 6px rgba(200, 162, 255, 0.4);
}
span[data-baseweb="tag"] * { color: var(--silver) !important; }
[data-testid="stFileUploaderDropzone"], [data-testid="stCameraInput"] > div {
    background: var(--panel) !important; border: 1px dashed var(--silver) !important; border-radius: 12px;
}
[data-testid="stFileUploaderDropzone"] *, [data-testid="stCameraInput"] * { color: var(--silver) !important; }

/* Buttons */
.stApp button[kind], .stApp [data-testid^="stBaseButton"] {
    background: var(--panel) !important; color: var(--silver) !important;
    border: 1px solid var(--silver) !important; border-radius: 10px !important;
}
.stApp button[kind]:hover, .stApp [data-testid^="stBaseButton"]:hover {
    border-color: var(--lilac) !important; color: var(--lilac) !important; box-shadow: var(--glow);
}
.stApp [data-testid^="stBaseButton"] *, .stApp button[kind] * { color: inherit !important; }

/* Alerts */
[data-testid="stAlert"], [data-testid="stAlertContainer"] {
    background: var(--panel) !important; border: 1px solid var(--line) !important; border-radius: 12px;
}
[data-testid="stAlert"] *, [data-testid="stAlertContainer"] * { color: var(--silver) !important; }

/* Images */
[data-testid="stImage"] img { border: 1px solid var(--line); border-radius: 10px; }
</style>
"""


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


def draw_detections(rgb: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    """Draw a silver HUD-style overlay: thin outline, corner brackets, label chip."""
    canvas = rgb.copy()
    height, width = canvas.shape[:2]
    scale = max(height, width) / 1000.0
    bracket_width = max(2, int(round(3 * scale)))
    font_scale = max(0.45, 0.6 * scale)
    text_width = max(1, int(round(1.5 * scale)))
    pad = 4

    for row in table.itertuples():
        x1, y1, x2, y2 = row.X1, row.Y1, row.X2, row.Y2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), SILVER_DIM, 1, cv2.LINE_AA)

        arm = max(8, int(min(x2 - x1, y2 - y1) * 0.22))
        for corner_x, corner_y, dx, dy in (
            (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
        ):
            cv2.line(canvas, (corner_x, corner_y), (corner_x + dx * arm, corner_y), SILVER, bracket_width, cv2.LINE_AA)
            cv2.line(canvas, (corner_x, corner_y), (corner_x, corner_y + dy * arm), SILVER, bracket_width, cv2.LINE_AA)

        label = f"{row.Object} {row.Confidence:.0%}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_width)
        chip_h = label_h + baseline + 2 * pad
        chip_top = y1 - chip_h if y1 - chip_h >= 0 else y1
        cv2.rectangle(canvas, (x1, chip_top), (x1 + label_w + 2 * pad, chip_top + chip_h), SILVER, -1)
        cv2.putText(
            canvas, label, (x1 + pad, chip_top + pad + label_h),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, BLACK, text_width, cv2.LINE_AA,
        )
    return canvas


@st.cache_data(show_spinner="Analysing image...", max_entries=8)
def run_detection(_model, data, confidence, iou, max_det, image_size, class_ids):
    """Run YOLOv8 once per (image, settings) combination and cache the result."""
    rgb = decode_image(data)
    # Ultralytics treats numpy input as BGR, so convert before predicting.
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
    return rgb, draw_detections(rgb, table), table, elapsed_ms


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
    """Overlay a confidence-weighted lilac/silver heatmap of object density."""
    height, width = rgb.shape[:2]
    density = np.zeros((height, width), dtype=np.float32)
    for row in table.itertuples():
        density[row.Y1:row.Y2, row.X1:row.X2] += row.Confidence
    if density.max() <= 0:
        return rgb.copy()

    kernel = int(max(height, width) * 0.08) | 1
    density = cv2.GaussianBlur(density, (kernel, kernel), 0)
    normalised = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    coloured = HEAT_PALETTE[normalised]

    # Opacity follows density, so empty regions keep the original pixels.
    alpha = (normalised.astype(np.float32) / 255.0 * 0.7)[..., None]
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
# HTML fragments (kept dark/silver/lilac regardless of Streamlit's own theme)
# ---------------------------------------------------------------------------
def hero_html() -> str:
    return (
        '<div class="vl-hero">'
        '<p class="vl-title">VISION LAB</p>'
        '<p class="vl-sub">Object detection and scene analysis with YOLOv8 '
        '&nbsp;|&nbsp; CodeAlpha AI Internship, Task 3</p>'
        '<div class="vl-scan"></div>'
        "</div>"
    )


def cards_html(items: list) -> str:
    cards = "".join(
        f'<div class="vl-card"><div class="vl-card-label">{escape(label)}</div>'
        f'<div class="vl-card-value">{escape(str(value))}</div></div>'
        for label, value in items
    )
    return f'<div class="vl-cards">{cards}</div>'


def class_bars_html(counts: pd.Series) -> str:
    peak = max(int(counts.max()), 1)
    rows = "".join(
        f'<div class="vl-bar-row"><span class="vl-bar-name">{escape(str(name))}</span>'
        f'<div class="vl-bar-track"><div class="vl-bar-fill" style="width:{100 * int(n) / peak:.0f}%"></div></div>'
        f'<span class="vl-bar-count">{int(n)}</span></div>'
        for name, n in counts.items()
    )
    return rows


def grid_html(grid: pd.DataFrame) -> str:
    peak = int(grid.values.max())
    head = "<tr><th></th>" + "".join(f"<th>{escape(c)}</th>" for c in grid.columns) + "</tr>"
    body = ""
    for label, row in grid.iterrows():
        cells = "".join(
            f'<td class="{"hot" if peak > 0 and int(v) == peak else ""}">{int(v)}</td>' for v in row
        )
        body += f"<tr><th>{escape(str(label))}</th>{cells}</tr>"
    return f'<div class="vl-table-wrap"><table class="vl-table vl-grid">{head}{body}</table></div>'


def table_html(table: pd.DataFrame) -> str:
    head = "".join(f"<th>{escape(column)}</th>" for column in table.columns)
    body = ""
    for row in table.itertuples(index=False):
        cells = ""
        for column, value in zip(table.columns, row):
            if column == "Confidence":
                cells += (
                    f'<td><div class="vl-conf"><div class="vl-conf-track">'
                    f'<div class="vl-conf-fill" style="width:{float(value) * 100:.0f}%"></div></div>'
                    f"<span>{float(value):.2f}</span></div></td>"
                )
            else:
                cells += f"<td>{escape(str(value))}</td>"
        body += f"<tr>{cells}</tr>"
    return f'<div class="vl-table-wrap"><table class="vl-table"><tr>{head}</tr>{body}</table></div>'


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def configure_page() -> None:
    st.set_page_config(page_title="Vision Lab | Object Detection", page_icon="👁️", layout="wide")
    st.markdown(THEME_CSS, unsafe_allow_html=True)


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


def render_empty_state() -> None:
    st.markdown(
        '<div class="vl-panel">Upload a photo or take one with your camera. '
        "<br><br><b>1.</b> Adjust the thresholds in the sidebar. "
        "<br><b>2.</b> Results update automatically. "
        "<br><b>3.</b> Explore the detections, privacy blur, density map and analysis tabs.</div>",
        unsafe_allow_html=True,
    )


def render_metrics(table: pd.DataFrame, elapsed_ms: float) -> None:
    mean_confidence = f"{table['Confidence'].mean():.0%}" if not table.empty else "n/a"
    st.markdown(
        cards_html(
            [
                ("Objects detected", len(table)),
                ("Categories", table["Object"].nunique()),
                ("Mean confidence", mean_confidence),
                ("Inference time", f"{elapsed_ms:.0f} ms"),
            ]
        ),
        unsafe_allow_html=True,
    )


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
        "Brighter lilac areas mark where detected objects are concentrated, "
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
        st.markdown(class_bars_html(table["Object"].value_counts()), unsafe_allow_html=True)
    with grid_col:
        st.subheader("Where the objects are")
        st.markdown(grid_html(spatial_grid(table, rgb.shape)), unsafe_allow_html=True)
        st.caption("Count of object centres in each part of the frame.")

    st.subheader("Detection table")
    st.markdown(table_html(table), unsafe_allow_html=True)

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
    st.markdown(hero_html(), unsafe_allow_html=True)

    model = load_model()
    settings = sidebar_settings(model.names)
    data = read_source()

    if data is None:
        render_empty_state()
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
