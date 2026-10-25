"""Task 4 - Object detection and tracking (YOLOv8 + SORT) for images, video and webcam snapshots."""
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.detection import (YoloDetector, count_classes, detect_image, load_yolo, process_video,
                            video_info)
from core.ui import header, inject_css, page_config

page_config(page_title="Object Detection & Tracking", page_icon="🎯", layout="wide")
inject_css(".block-container{padding-top:2rem}")

MODELS = {
    "YOLOv8n · fastest (bundled)": "yolov8n.pt",
    "YOLOv8s · balanced (downloads ~22 MB)": "yolov8s.pt",
    "YOLOv8m · most accurate (downloads ~50 MB)": "yolov8m.pt",
}
ss = st.session_state


@st.cache_resource(show_spinner="Loading YOLO model...")
def get_model(weights: str):
    return load_yolo(weights)


# ------------------------------ sidebar ------------------------------------ #
with st.sidebar:
    st.subheader("⚙️ Detection")
    model_label = st.selectbox("Model", list(MODELS))
    try:
        model = get_model(MODELS[model_label])
    except Exception as exc:
        st.error(f"Could not load {MODELS[model_label]}: {exc}")
        st.stop()
    class_names = [model.names[i] for i in sorted(model.names)]
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    iou = st.slider("NMS IoU threshold", 0.10, 0.95, 0.50, 0.05)
    imgsz = st.select_slider("Inference size", [320, 480, 640, 960], value=640)
    chosen = st.multiselect("Only detect these classes (empty = all 80)", class_names)
    class_ids = [i for i, n in model.names.items() if n in chosen] or None

    st.subheader("🧭 Tracker (SORT)")
    max_age = st.slider("Max age (frames a lost track is kept)", 1, 60, 15)
    min_hits = st.slider("Min hits (frames before an ID is shown)", 1, 10, 3)
    sort_iou = st.slider("Match IoU threshold", 0.05, 0.80, 0.30, 0.05)


def make_detector() -> YoloDetector:
    return YoloDetector(conf=conf, iou=iou, classes=class_ids, imgsz=imgsz, model=model)


def to_png_bytes(bgr: np.ndarray) -> bytes:
    return cv2.imencode(".png", bgr)[1].tobytes()


def show_image_results(img_rgb: np.ndarray, key: str) -> None:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    with st.spinner("Detecting..."):
        annotated, dets = detect_image(make_detector(), bgr)
    counts = count_classes(dets, model.names)
    c1, c2 = st.columns(2)
    c1.markdown("**Original**")
    c1.image(img_rgb)
    c2.markdown("**Detections**")
    c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    m1, m2 = st.columns(2)
    m1.metric("Objects found", len(dets))
    m2.metric("Distinct classes", len(counts))
    if len(dets):
        table = pd.DataFrame({
            "class": [model.names[int(c)] for c in dets[:, 5]],
            "confidence": dets[:, 4].round(3),
            "x1": dets[:, 0].round(0), "y1": dets[:, 1].round(0),
            "x2": dets[:, 2].round(0), "y2": dets[:, 3].round(0)})
        left, right = st.columns([3, 2])
        left.dataframe(table, hide_index=True)
        right.bar_chart(pd.Series(counts, name="count"))
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Annotated image (PNG)", to_png_bytes(annotated), "detections.png",
                           "image/png", key=f"png_{key}")
        d2.download_button("⬇️ Detections (CSV)", table.to_csv(index=False), "detections.csv",
                           "text/csv", key=f"csv_{key}")
    else:
        st.info("No objects found. Try lowering the confidence threshold in the sidebar.")


# ------------------------------ page --------------------------------------- #
header("🎯 Object Detection & Tracking",
       "CodeAlpha AI Internship · Task 4 · YOLOv8 detection + SORT tracking")
tab_img, tab_vid, tab_cam, tab_about = st.tabs(["📷 Image", "🎬 Video + tracking", "🤳 Webcam snapshot", "ℹ️ How it works"])

# ---- image ----
with tab_img:
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"], key="img_up")
    use_sample = st.checkbox("Use the sample image (bus.jpg)")
    if up is not None:
        show_image_results(np.array(Image.open(up).convert("RGB")), "img")
    elif use_sample:
        try:
            from ultralytics.utils import ASSETS
            show_image_results(np.array(Image.open(ASSETS / "bus.jpg").convert("RGB")), "sample")
        except Exception as exc:
            st.warning(f"Sample image unavailable: {exc}")
    else:
        st.info("Upload a photo (street, room, desk...) or tick the sample image.")

# ---- video ----
with tab_vid:
    st.write("Upload a short clip. Every object gets a **persistent ID** and a motion trail; "
             "optionally count objects crossing a line.")
    vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv", "webm"], key="vid_up")
    o1, o2, o3 = st.columns(3)
    stride = o1.slider("Process every Nth frame", 1, 6, 2,
                       help="Higher = faster, but fast-moving objects are tracked less smoothly.")
    max_frames = o2.slider("Max frames to process", 30, 600, 240, 30,
                           help="Keeps free cloud CPUs responsive.")
    max_width = o3.select_slider("Max width (px)", [480, 640, 960, 1280], value=640)

    use_line = st.toggle("Count objects crossing a line")
    line = None
    if use_line:
        l1, l2 = st.columns(2)
        orient = l1.radio("Line", ["Horizontal", "Vertical"], horizontal=True)
        pos = l2.slider("Position (%)", 5, 95, 50) / 100
        line = ((0.0, pos), (1.0, pos)) if orient == "Horizontal" else ((pos, 0.0), (pos, 1.0))

    if vid is not None:
        key = (vid.name, vid.size)
        if ss.get("vid_key") != key:  # new upload: write it once, drop the previous temp file + result
            old_path = ss.get("vid_path")
            if old_path and os.path.exists(old_path):
                os.remove(old_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vid.name)[1] or ".mp4") as f:
                f.write(vid.getvalue())
                ss.vid_path = f.name
            ss.vid_key, ss.vid_result = key, None
        src_path = ss.vid_path
        try:
            n, fps, w, h = video_info(src_path)
            st.caption(f"{vid.name} · {w}×{h} · {fps:.0f} fps · {n} frames (~{n / fps:.1f}s)")
        except Exception:
            pass

        if st.button("▶️ Run detection + tracking", type="primary", key="run_vid"):
            dst_path = src_path + "_tracked.mp4"
            bar, preview, status = st.progress(0.0), st.empty(), st.empty()

            def on_frame(i, total, frame_bgr, live):
                if total:
                    bar.progress(min(i / total, 1.0))
                if i % 2 == 0 or i == 1:
                    preview.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                status.caption(f"Frame {i}/{total or '?'} · tracking {live['tracked']} objects")

            try:
                res = process_video(src_path, dst_path, make_detector(), max_frames=max_frames,
                                    stride=stride, max_width=max_width, line=line,
                                    tracker_kwargs={"max_age": max_age, "min_hits": min_hits,
                                                    "iou_threshold": sort_iou},
                                    on_frame=on_frame)
                with open(dst_path, "rb") as fh:
                    ss.vid_result = {
                        "video": fh.read(), "csv": res.stats.to_csv(), "counts": res.stats.unique_counts(),
                        "frames": res.stats.frames, "in": res.line_in, "out": res.line_out,
                        "line": bool(line), "rows": len(res.stats.rows)}
                bar.empty(), preview.empty(), status.empty()
            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
            finally:
                for p in (dst_path,):
                    if os.path.exists(p):
                        os.remove(p)

    r = ss.get("vid_result")
    if r:
        st.success(f"Processed {r['frames']} frames.")
        st.video(r["video"])
        cols = st.columns(3 if r["line"] else 2)
        cols[0].metric("Unique objects tracked", sum(r["counts"].values()))
        cols[1].metric("Classes", len(r["counts"]))
        if r["line"]:
            cols[2].metric("Line crossings (in / out)", f"{r['in']} / {r['out']}")
        if r["counts"]:
            st.markdown("**Unique objects per class** (each tracking ID counted once)")
            st.bar_chart(pd.Series(r["counts"], name="unique objects"))
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Tracked video (MP4)", r["video"], "tracked.mp4", "video/mp4", key="dl_vid")
        d2.download_button("⬇️ Tracking log (CSV)", r["csv"], "tracking_log.csv", "text/csv", key="dl_vid_csv")

# ---- webcam snapshot ----
with tab_cam:
    st.write("Take a photo with your webcam and detect objects in it. For **continuous real-time "
             "tracking**, run `scripts/webcam_tracker.py` locally (see *How it works*).")
    shot = st.camera_input("Webcam")
    if shot is not None:
        show_image_results(np.array(Image.open(shot).convert("RGB")), "cam")

# ---- about ----
with tab_about:
    st.markdown("""
**Pipeline** (per frame)

1. **Detect** - a pre-trained YOLOv8 network returns boxes, classes and confidences.
2. **Track** - our own implementation of **SORT** (Bewley et al., 2016) in `core/sort_tracker.py`:
   a constant-velocity **Kalman filter** predicts each object's next box; the **Hungarian algorithm**
   matches predictions to new detections by IoU (class-aware); unmatched detections start new tracks;
   tracks unseen for `max age` frames are dropped; an ID is only shown after `min hits` confirmations.
3. **Draw** - labelled boxes with tracking IDs, motion trails, and an optional line-crossing counter.

**Real-time webcam / video on your own machine**
```bash
pip install -r requirements.txt
python scripts/webcam_tracker.py                      # webcam, press q to quit
python scripts/webcam_tracker.py --source clip.mp4 --save out.mp4
python scripts/webcam_tracker.py --classes person car --conf 0.4
```
""")
