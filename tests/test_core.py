"""Unit tests for Task 2 (FAQ engine) and Task 4 (SORT + video pipeline).

Run:  python -m unittest discover -s tests -v
No network, no Streamlit and no YOLO weights are needed.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.faq_engine import FAQEngine, preprocess  # noqa: E402
from core.sort_tracker import Sort, iou_matrix  # noqa: E402


# --------------------------------------------------------------------------- #
class FAQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FAQEngine()

    def test_preprocess_removes_stopwords_and_stems(self):
        self.assertEqual(preprocess("What are the submission deadlines?").split()[-1], preprocess("deadlines"))
        self.assertNotIn("the", preprocess("the deadline").split())

    def test_exact_question_returns_that_faq(self):
        r = self.engine.ask("What is the submission deadline?")
        self.assertEqual(r.kind, "answer")
        self.assertEqual(r.best.faq_id, "deadline")

    def test_typo_tolerance(self):
        self.assertEqual(self.engine.rank("what is the submision deadlne")[0].faq_id, "deadline")

    def test_out_of_scope_is_not_answered(self):
        for q in ("explain quantum computing", "what is the capital of france", "asdfgh qwerty"):
            self.assertNotEqual(self.engine.ask(q).kind, "answer", q)

    def test_stopword_only_query(self):
        self.assertEqual(self.engine.ask("what can you do").best.faq_id, "help")

    def test_empty_query(self):
        self.assertEqual(self.engine.rank("   "), [])

    def test_all_faqs_have_required_fields(self):
        for f in self.engine.faqs:
            for k in ("id", "category", "question", "answer", "variants"):
                self.assertIn(k, f)
        self.assertEqual(len({f["id"] for f in self.engine.faqs}), len(self.engine.faqs))


# --------------------------------------------------------------------------- #
class SortTests(unittest.TestCase):
    def test_iou(self):
        a = np.array([[0, 0, 10, 10]], float)
        self.assertAlmostEqual(iou_matrix(a, a)[0, 0], 1.0)
        self.assertEqual(iou_matrix(a, np.array([[20, 20, 30, 30]], float))[0, 0], 0.0)
        self.assertAlmostEqual(iou_matrix(a, np.array([[5, 0, 15, 10]], float))[0, 0], 1 / 3, places=5)

    def test_identity_persists_for_moving_object(self):
        sort, ids = Sort(min_hits=2), set()
        for i in range(30):
            x = 10 + i * 6
            out = sort.update(np.array([[x, 50, x + 40, 130, 0.9, 0]], float))
            ids.update(out[:, 4].astype(int).tolist())
        self.assertEqual(len(ids), 1)

    def test_two_objects_get_two_ids_and_keep_them_when_crossing(self):
        sort, seen = Sort(min_hits=2), {0: set(), 1: set()}
        for i in range(40):
            a = np.array([10 + i * 5, 20, 50 + i * 5, 80, 0.9, 0])        # moves right
            b = np.array([210 - i * 5, 120, 250 - i * 5, 180, 0.9, 0])    # moves left (different row)
            out = sort.update(np.array([a, b], float))
            for row in out:
                cy = (row[1] + row[3]) / 2
                seen[0 if cy < 100 else 1].add(int(row[4]))
        self.assertEqual(len(seen[0]), 1)
        self.assertEqual(len(seen[1]), 1)
        self.assertNotEqual(seen[0], seen[1])

    def test_track_survives_short_occlusion(self):
        sort, ids = Sort(max_age=10, min_hits=2), set()
        for i in range(40):
            dets = np.zeros((0, 6)) if 15 <= i < 20 else np.array([[100 + i * 3, 100, 140 + i * 3, 180, 0.9, 0]], float)
            out = sort.update(dets)
            ids.update(out[:, 4].astype(int).tolist())
        self.assertEqual(len(ids), 1)

    def test_track_dropped_after_max_age(self):
        sort = Sort(max_age=3, min_hits=1)
        sort.update(np.array([[0, 0, 50, 50, 0.9, 0]], float))
        for _ in range(6):
            sort.update(None)
        self.assertEqual(len(sort.trackers), 0)

    def test_single_frame_false_positive_gets_no_id(self):
        sort = Sort(min_hits=3)
        for i in range(5):
            sort.update(np.array([[10 + i, 10, 60 + i, 90, 0.9, 0]], float))
        out = sort.update(np.array([[10, 10, 60, 90, 0.9, 0], [300, 300, 340, 360, 0.5, 2]], float))
        self.assertEqual(len(out), 1)

    def test_class_aware_association(self):
        sort = Sort(min_hits=1)
        for _ in range(3):
            sort.update(np.array([[0, 0, 50, 50, 0.9, 0]], float))
        out = sort.update(np.array([[0, 0, 50, 50, 0.9, 5]], float))  # same place, other class
        self.assertEqual(int(out[0, 6]), 5)
        self.assertGreater(int(out[0, 4]), 1)  # new id, not the person's id


# --------------------------------------------------------------------------- #
class FakeDetector:
    """Detects a moving white square on a black background - no ML needed."""
    names = {0: "box"}

    def detect(self, frame):
        gray = frame.mean(axis=2)
        ys, xs = np.where(gray > 200)
        if len(xs) == 0:
            return np.zeros((0, 6))
        return np.array([[xs.min(), ys.min(), xs.max(), ys.max(), 0.95, 0]], float)


class VideoPipelineTests(unittest.TestCase):
    def _make_video(self, path, frames=40, w=320, h=240):
        import imageio.v2 as imageio
        wr = imageio.get_writer(str(path), fps=20, codec="libx264", macro_block_size=1)
        for i in range(frames):
            f = np.zeros((h, w, 3), np.uint8)
            x = 10 + i * 6
            f[100:140, x:x + 40] = 255
            wr.append_data(f)
        wr.close()

    def test_end_to_end_tracking_and_line_counting(self):
        from core.detection import process_video, video_info
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d, "in.mp4"), Path(d, "out.mp4")
            self._make_video(src)
            seen = []
            res = process_video(str(src), str(dst), FakeDetector(), tracker_kwargs={"min_hits": 2},
                                line=((0.5, 0.0), (0.5, 1.0)), max_width=None,
                                on_frame=lambda i, n, f, live: seen.append(i))
            self.assertTrue(dst.exists() and dst.stat().st_size > 1000)
            n, fps, w, h = video_info(str(dst))          # output is a valid, readable video
            self.assertEqual((w, h), (320, 240))
            self.assertGreaterEqual(n, 38)
            self.assertEqual(res.stats.unique_counts(), {"box": 1})   # one object, one id
            self.assertEqual(res.line_in + res.line_out, 1)           # crossed the line once
            self.assertEqual(len(seen), res.stats.frames)
            self.assertIn("track_id", res.stats.to_csv().splitlines()[0])

    def test_stride_and_max_frames(self):
        from core.detection import process_video
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d, "in.mp4"), Path(d, "out.mp4")
            self._make_video(src, frames=40)
            res = process_video(str(src), str(dst), FakeDetector(), stride=2, max_frames=10, max_width=None)
            self.assertEqual(res.stats.frames, 10)

    def test_missing_video_raises(self):
        from core.detection import process_video
        with self.assertRaises(ValueError):
            process_video("/nonexistent.mp4", "/tmp/x.mp4", FakeDetector())

    def test_detect_image_draws_and_counts(self):
        from core.detection import count_classes, detect_image
        img = np.zeros((240, 320, 3), np.uint8)
        img[50:100, 50:100] = 255
        annotated, dets = detect_image(FakeDetector(), img)
        self.assertEqual(count_classes(dets, FakeDetector.names), {"box": 1})
        self.assertFalse(np.array_equal(annotated, img))  # something was drawn


if __name__ == "__main__":
    unittest.main()
