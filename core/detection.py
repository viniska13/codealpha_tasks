"""Detection -> tracking -> drawing pipeline for Task 4.

The YOLO model is isolated behind :class:`YoloDetector`; everything else
(tracker, drawing, line counting, video loop, statistics) is plain NumPy/OpenCV
and is unit-tested with a fake detector, so it works the same for any model
that returns ``[x1, y1, x2, y2, conf, cls]`` rows.
"""
from __future__ import annotations

import colorsys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol, Sequence

import cv2
import numpy as np

from .sort_tracker import Sort


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class Detector(Protocol):
    names: dict[int, str]

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return an (N, 6) array ``[x1, y1, x2, y2, conf, cls]``."""


def load_yolo(weights: str = "yolov8n.pt"):
    """Load (and download, if needed) an Ultralytics YOLO model. Cache this in the UI."""
    from ultralytics import YOLO  # imported lazily: heavy, optional for tests

    return YOLO(weights)


class YoloDetector:
    """Thin wrapper around an Ultralytics YOLO model.

    Pass an already-loaded ``model`` to share one set of weights between runs while
    keeping per-run settings (confidence, classes, ...) private to each detector.
    """

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35, iou: float = 0.5,
                 classes: Optional[Sequence[int]] = None, imgsz: int = 640, model=None):
        self.model = model if model is not None else load_yolo(weights)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.classes = list(classes) if classes else None
        self.names: dict[int, str] = dict(self.model.names)

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        res = self.model.predict(frame_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                 classes=self.classes, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 6))
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()[:, None]
        cls = res.boxes.cls.cpu().numpy()[:, None]
        return np.hstack([xyxy, conf, cls]).astype(float)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def color_for(track_id: int) -> tuple[int, int, int]:
    """Stable, well-separated BGR colour for a track id (golden-ratio hue walk)."""
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_tracks(frame: np.ndarray, tracks: np.ndarray, names: dict[int, str],
                trails: Optional[dict[int, deque]] = None,
                line: Optional[tuple[tuple[int, int], tuple[int, int]]] = None,
                counts: Optional[dict[str, int]] = None) -> np.ndarray:
    """Return a copy of ``frame`` annotated with boxes, ids, labels, trails, line."""
    out = frame.copy()
    thick = max(2, round(min(out.shape[:2]) / 300))
    scale = max(0.5, min(out.shape[:2]) / 900)

    if trails:
        for tid, pts in trails.items():
            if len(pts) > 1:
                cv2.polylines(out, [np.array(pts, np.int32)], False, color_for(tid), thick)

    for x1, y1, x2, y2, tid, conf, cls in tracks:
        tid, cls = int(tid), int(cls)
        color = color_for(tid)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, thick)
        label = f"{names.get(cls, cls)} #{tid} {conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        top = max(p1[1] - th - base - 4, 0)
        cv2.rectangle(out, (p1[0], top), (p1[0] + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(out, label, (p1[0] + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 1, cv2.LINE_AA)

    if line:
        cv2.line(out, line[0], line[1], (0, 255, 255), thick + 1)
    if counts:
        text = "  ".join(f"{k}: {v}" for k, v in counts.items())
        cv2.rectangle(out, (0, 0), (out.shape[1], int(34 * scale) + 8), (0, 0, 0), -1)
        cv2.putText(out, text, (8, int(26 * scale) + 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 255, 255), 2, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Line crossing counter
# --------------------------------------------------------------------------- #
class LineCounter:
    """Counts tracks crossing a line, split by direction.

    ``in``  = crossing from the left of the A->B vector to its right
    ``out`` = the opposite way.  Each track id is counted at most once.
    """

    def __init__(self, a: tuple[int, int], b: tuple[int, int]):
        self.a, self.b = a, b
        self.in_count = 0
        self.out_count = 0
        self._last_side: dict[int, float] = {}
        self._counted: set[int] = set()

    def _side(self, p: tuple[float, float]) -> float:
        (ax, ay), (bx, by) = self.a, self.b
        return (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)

    def update(self, tracks: np.ndarray) -> None:
        for x1, y1, x2, y2, tid, *_ in tracks:
            tid = int(tid)
            side = self._side(((x1 + x2) / 2, (y1 + y2) / 2))
            prev = self._last_side.get(tid)
            if prev is not None and prev * side < 0 and tid not in self._counted:
                self._counted.add(tid)
                if prev < 0 < side:
                    self.in_count += 1
                else:
                    self.out_count += 1
            self._last_side[tid] = side


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
@dataclass
class RunStats:
    """Aggregates everything we want to show / export after a run."""
    names: dict[int, str]
    frames: int = 0
    unique_ids: dict[int, set] = field(default_factory=lambda: defaultdict(set))
    rows: list[dict] = field(default_factory=list)  # one row per tracked box per frame

    def add(self, frame_idx: int, tracks: np.ndarray, fps: float = 0.0) -> None:
        self.frames += 1
        for x1, y1, x2, y2, tid, conf, cls in tracks:
            cls = int(cls)
            self.unique_ids[cls].add(int(tid))
            self.rows.append({
                "frame": frame_idx, "time_s": round(frame_idx / fps, 3) if fps else None,
                "track_id": int(tid), "class": self.names.get(cls, str(cls)),
                "confidence": round(float(conf), 3),
                "x1": round(float(x1), 1), "y1": round(float(y1), 1),
                "x2": round(float(x2), 1), "y2": round(float(y2), 1),
            })

    def unique_counts(self) -> dict[str, int]:
        return {self.names.get(c, str(c)): len(ids) for c, ids in
                sorted(self.unique_ids.items(), key=lambda kv: -len(kv[1]))}

    def to_csv(self) -> str:
        import csv, io

        buf = io.StringIO()
        cols = ["frame", "time_s", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"]
        w = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        w.writerows(self.rows)
        return buf.getvalue()


def count_classes(dets: np.ndarray, names: dict[int, str]) -> dict[str, int]:
    """Class histogram for a single image (no tracking)."""
    c = Counter(names.get(int(k), str(int(k))) for k in dets[:, 5]) if len(dets) else Counter()
    return dict(c.most_common())


# --------------------------------------------------------------------------- #
# Image + video drivers
# --------------------------------------------------------------------------- #
def detect_image(detector: Detector, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run detection on one image. Returns (annotated_bgr, detections)."""
    dets = detector.detect(frame_bgr)
    # draw_tracks expects a track id column; use 1..N as display ids
    as_tracks = np.column_stack([dets[:, :4], np.arange(1, len(dets) + 1), dets[:, 4], dets[:, 5]]) \
        if len(dets) else np.zeros((0, 7))
    return draw_tracks(frame_bgr, as_tracks, detector.names), dets


@dataclass
class VideoResult:
    output_path: Path
    stats: RunStats
    fps: float
    size: tuple[int, int]
    line_in: int = 0
    line_out: int = 0


def video_info(path: str) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return n, fps, w, h


def process_video(
    src: str,
    dst: str,
    detector: Detector,
    *,
    max_frames: Optional[int] = None,
    stride: int = 1,
    max_width: Optional[int] = 960,
    tracker_kwargs: Optional[dict] = None,
    line: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
    trail_len: int = 30,
    on_frame: Optional[Callable[[int, int, np.ndarray, dict], None]] = None,
) -> VideoResult:
    """Detect + track every ``stride``-th frame of ``src`` and write an H.264 mp4 to ``dst``.

    ``line`` is given in *relative* coordinates ((x1,y1),(x2,y2)) in 0..1 so it works
    after down-scaling. ``on_frame(i, total, annotated_bgr, live_counts)`` is called per
    processed frame (used by the UI for the live preview / progress bar).
    """
    import imageio.v2 as imageio

    stride = max(1, int(stride))
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {src}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_fps = max(1.0, src_fps / stride)
    available = -(-total // stride) if total else 0  # ceil(total / stride)
    n_planned = min(available, max_frames) if (max_frames and available) else (max_frames or available)

    tracker = Sort(**(tracker_kwargs or {}))
    stats = RunStats(detector.names)
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_len))
    counter: Optional[LineCounter] = None
    writer = None
    size: tuple[int, int] = (0, 0)
    processed = read = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            read += 1
            if (read - 1) % stride:
                continue
            if max_width and frame.shape[1] > max_width:
                s = max_width / frame.shape[1]
                frame = cv2.resize(frame, (max_width, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
            if writer is None:
                size = (w - w % 2, h - h % 2)  # H.264 needs even dimensions
                writer = imageio.get_writer(str(dst), fps=out_fps, codec="libx264",
                                            macro_block_size=1, quality=7)
                if line:
                    px = ((int(line[0][0] * w), int(line[0][1] * h)), (int(line[1][0] * w), int(line[1][1] * h)))
                    counter = LineCounter(*px)

            tracks = tracker.update(detector.detect(frame))
            for x1, y1, x2, y2, tid, *_ in tracks:
                trails[int(tid)].append((int((x1 + x2) / 2), int(y2)))  # foot point
            if counter:
                counter.update(tracks)
            live = {"tracked": len(tracks)}
            if counter:
                live.update({"in": counter.in_count, "out": counter.out_count})
            annotated = draw_tracks(frame, tracks, detector.names, trails,
                                    counter and (counter.a, counter.b), live if counter else None)
            writer.append_data(cv2.cvtColor(annotated[:size[1], :size[0]], cv2.COLOR_BGR2RGB))
            processed += 1
            stats.add(read - 1, tracks, src_fps)
            if on_frame:
                on_frame(processed, n_planned, annotated, live)
            if max_frames and processed >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.close()

    if processed == 0:
        raise ValueError("The video has no readable frames")
    return VideoResult(Path(dst), stats, out_fps, size,
                       counter.in_count if counter else 0, counter.out_count if counter else 0)
