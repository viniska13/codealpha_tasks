"""Real-time object detection + SORT tracking from a webcam or a video file.

    python scripts/webcam_tracker.py                          # default webcam, press q to quit
    python scripts/webcam_tracker.py --source clip.mp4 --save out.mp4
    python scripts/webcam_tracker.py --classes person car --conf 0.4
    python scripts/webcam_tracker.py --source clip.mp4 --no-display --max-frames 300 --save out.mp4
"""
import argparse
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.detection import RunStats, YoloDetector, draw_tracks  # noqa: E402
from core.sort_tracker import Sort  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="webcam index (0, 1, ...) or path to a video file")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--classes", nargs="*", help="class names to keep, e.g. person car")
    ap.add_argument("--save", help="write the annotated video to this .mp4 path")
    ap.add_argument("--csv", help="write a per-frame tracking log to this CSV path")
    ap.add_argument("--no-display", action="store_true", help="don't open a window (servers / CI)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    args = ap.parse_args()

    detector = YoloDetector(args.weights, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    if args.classes:
        wanted = {n.lower() for n in args.classes}
        ids = [i for i, n in detector.names.items() if n.lower() in wanted]
        if not ids:
            sys.exit(f"None of {args.classes} are valid class names. Examples: person, car, dog, cell phone")
        detector.classes = ids

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open source: {args.source}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)
    stats = RunStats(detector.names)
    trails = defaultdict(lambda: deque(maxlen=30))
    writer, frame_idx, t_prev, fps = None, 0, time.time(), 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tracks = tracker.update(detector.detect(frame))
            for x1, y1, x2, y2, tid, *_ in tracks:
                trails[int(tid)].append((int((x1 + x2) / 2), int(y2)))
            out = draw_tracks(frame, tracks, detector.names, trails)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if fps else 1 / max(now - t_prev, 1e-6)
            t_prev = now
            cv2.putText(out, f"FPS {fps:.1f}  tracked {len(tracks)}", (10, out.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            stats.add(frame_idx, tracks, fps_in)
            frame_idx += 1

            if args.save:
                if writer is None:
                    h, w = out.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
                writer.write(out)
            if not args.no_display:
                cv2.imshow("CodeAlpha - detection + SORT tracking (q to quit)", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames. Unique objects per class: {stats.unique_counts()}")
    if args.csv:
        Path(args.csv).write_text(stats.to_csv(), encoding="utf-8")
        print(f"Tracking log -> {args.csv}")
    if args.save:
        print(f"Annotated video -> {args.save}")


if __name__ == "__main__":
    main()
