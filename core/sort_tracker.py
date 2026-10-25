"""SORT (Simple Online and Realtime Tracking) implemented from scratch.

Bewley et al., 2016 - https://arxiv.org/abs/1602.00763

* Each track has a constant-velocity Kalman filter over the box state
  ``[cx, cy, area, aspect_ratio, vcx, vcy, v_area]``.
* Every frame: predict -> associate detections to predictions with the
  Hungarian algorithm on IoU (class-aware) -> update matched tracks ->
  start new tracks for unmatched detections -> drop tracks unseen for
  ``max_age`` frames.
* A track is reported only after ``min_hits`` consecutive-ish confirmations so
  one-frame false positives never get an ID.

Detections are rows ``[x1, y1, x2, y2, conf, cls]``; outputs are rows
``[x1, y1, x2, y2, track_id, conf, cls]``.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes ``a`` (N,4) and ``b`` (M,4) in xyxy format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a[:, None, :]
    b = b[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _xyxy_to_z(box: np.ndarray) -> np.ndarray:
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / max(h, 1e-6)])


def _x_to_xyxy(x: np.ndarray) -> np.ndarray:
    area, ratio = max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(area * ratio)
    h = area / max(w, 1e-6)
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class KalmanBoxTracker:
    _next_id = 1

    def __init__(self, box: np.ndarray, conf: float, cls: int):
        # State x = [cx, cy, s, r, vcx, vcy, vs]; measurement z = [cx, cy, s, r]
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4])
        self.x = np.zeros(7)
        self.x[:4] = _xyxy_to_z(box)

        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1
        self.conf, self.cls = float(conf), int(cls)
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def predict(self) -> np.ndarray:
        if self.x[2] + self.x[6] <= 0:  # keep area positive
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _x_to_xyxy(self.x)

    def update(self, box: np.ndarray, conf: float, cls: int) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.conf, self.cls = float(conf), int(cls)
        z = _xyxy_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        return _x_to_xyxy(self.x)


class Sort:
    def __init__(self, max_age: int = 15, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age, self.min_hits, self.iou_threshold = max_age, min_hits, iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        KalmanBoxTracker._next_id = 1  # fresh ids for every new Sort instance

    def update(self, dets: np.ndarray | None = None) -> np.ndarray:
        """Advance one frame. ``dets``: (N, 6) ``[x1,y1,x2,y2,conf,cls]``."""
        self.frame_count += 1
        dets = np.zeros((0, 6)) if dets is None or len(dets) == 0 else np.asarray(dets, float)

        preds = np.array([t.predict() for t in self.trackers]).reshape(-1, 4)
        bad = np.isnan(preds).any(axis=1)
        if bad.any():  # drop trackers whose filter diverged
            keep = ~bad
            self.trackers = [t for t, k in zip(self.trackers, keep) if k]
            preds = preds[keep]

        matches, unmatched_dets = self._associate(dets, preds)
        for d_idx, t_idx in matches:
            self.trackers[t_idx].update(dets[d_idx, :4], dets[d_idx, 4], dets[d_idx, 5])
        for d_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[d_idx, :4], dets[d_idx, 4], dets[d_idx, 5]))

        out = []
        for t in self.trackers:
            confirmed = t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            if t.time_since_update == 0 and confirmed:
                out.append([*t.box, t.id, t.conf, t.cls])
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return np.array(out) if out else np.zeros((0, 7))

    def _associate(self, dets: np.ndarray, preds: np.ndarray):
        if len(preds) == 0:
            return [], list(range(len(dets)))
        if len(dets) == 0:
            return [], []
        iou = iou_matrix(dets[:, :4], preds)
        # class-aware: a "car" detection can't continue a "person" track
        t_cls = np.array([t.cls for t in self.trackers])
        iou[dets[:, 5][:, None] != t_cls[None, :]] = 0.0

        rows, cols = linear_sum_assignment(-iou)
        matches, matched_d = [], set()
        for r, c in zip(rows, cols):
            if iou[r, c] >= self.iou_threshold:
                matches.append((r, c))
                matched_d.add(r)
        return matches, [d for d in range(len(dets)) if d not in matched_d]
