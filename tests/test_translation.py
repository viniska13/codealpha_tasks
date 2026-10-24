"""Unit tests for the Lexicon translation backend (core/translation.py).

Run:  python -m unittest discover -s tests -v
No network needed: the HTTP layer and engines are replaced with fakes, so the tests are
fast, deterministic, and cover failure modes that are hard to trigger on demand.
"""
import sys
import time
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import translation as tr  # noqa: E402


class TranslationTests(unittest.TestCase):
    """Backend logic. HTTP is replaced by fakes, so no network is needed."""

    @staticmethod
    def eng(name, fn, max_chars=4500, auto=True):
        return tr.Engine(name, fn, max_chars, auto)

    def test_chunking_respects_limit_and_preserves_content(self):
        text = ("Hello world. " * 600).strip() + "\n\nSecond paragraph here."
        chunks = tr.chunk_text(text, limit=500)
        self.assertTrue(all(len(c) <= 500 for c in chunks))
        joined = " ".join(c for c in chunks if c)
        self.assertEqual(joined.count("Hello world."), 600)
        self.assertIn("Second paragraph here.", joined)

    def test_chunking_handles_unbroken_giant_word(self):
        self.assertEqual([len(c) for c in tr.chunk_text("x" * 1234, limit=500)], [500, 500, 234])

    def test_each_engine_gets_chunks_within_its_own_limit(self):
        seen = []
        def spy(t, s, d):
            seen.append(len(t))
            return t.upper()
        text = "word " * 400  # 2000 chars
        r = tr.translate(text, "en", "fr", engines=[self.eng("small", spy, max_chars=450)])
        self.assertTrue(seen and max(seen) <= 450)
        self.assertGreater(r.chunks, 4)

    def test_fallback_engine_used_when_first_fails(self):
        def bad(*_): raise tr.EngineError("blocked")
        r = tr.translate("hi there", "en", "fr",
                         engines=[self.eng("A", bad), self.eng("B", lambda t, s, d: f"[{d}] {t}")])
        self.assertEqual((r.engine, r.text), ("B", "[fr] hi there"))
        self.assertIn("A: blocked", r.errors[0])

    def test_all_engines_fail_raises_with_every_reason(self):
        def a(*_): raise tr.EngineError("quota")
        def b(*_): raise tr.EngineError("captcha")
        with self.assertRaises(tr.TranslationError) as cm:
            tr.translate("hi", "en", "de", engines=[self.eng("A", a), self.eng("B", b)])
        self.assertIn("A: quota", str(cm.exception))
        self.assertIn("B: captcha", str(cm.exception))

    def test_empty_engine_output_counts_as_failure(self):
        r = tr.translate("hi", "en", "fr", engines=[self.eng("A", lambda *a: ""), self.eng("B", lambda *a: "salut")])
        self.assertEqual((r.engine, r.text), ("B", "salut"))

    def test_engine_reported_language_overrides_guess(self):
        r = tr.translate("namaste dost", "auto", "en",
                         engines=[self.eng("G", lambda t, s, d: ("hello friend", "hi"))])
        self.assertEqual((r.source_code, r.detected, r.source_name), ("hi", True, "Hindi"))

    def test_engine_without_auto_support_is_skipped_when_language_unknown(self):
        orig = tr.detect_language
        tr.detect_language = lambda t: None
        try:
            r = tr.translate("zzzz", "auto", "en", engines=[self.eng("M", lambda *a: "x", auto=False),
                                                           self.eng("G", lambda *a: "ok")])
        finally:
            tr.detect_language = orig
        self.assertEqual(r.engine, "G")
        self.assertIn("M:", r.errors[0])

    def test_same_language_short_circuits(self):
        r = tr.translate("hello", "en", "en", engines=[self.eng("A", lambda *a: 1 / 0)])
        self.assertEqual(r.text, "hello")

    def test_multiline_structure_preserved(self):
        r = tr.translate("one\n\ntwo", "en", "fr", engines=[self.eng("A", lambda t, s, d: t.upper())])
        self.assertEqual(r.text, "ONE\n\nTWO")

    def test_result_reports_timing_and_detection(self):
        orig = tr.detect_with_confidence
        tr.detect_with_confidence = lambda text: ("fr", 0.97)
        try:
            r = tr.translate("bonjour tout le monde", "auto", "en",
                             engines=[self.eng("G", lambda t, s, d: (time.sleep(0.02), "hello everyone")[1])])
        finally:
            tr.detect_with_confidence = orig
        self.assertEqual((r.source_code, r.detected, r.text), ("fr", True, "hello everyone"))
        self.assertGreaterEqual(r.elapsed_ms, 15)

    def test_empty_text_rejected(self):
        with self.assertRaises(ValueError):
            tr.translate("   ", "en", "fr")

    def test_language_table_is_consistent(self):
        self.assertEqual(len(set(tr.LANGUAGES.values())), len(tr.LANGUAGES))
        self.assertEqual(tr.LANGUAGES["Hindi"], "hi")

    # ---- the real engines, with the HTTP layer faked ----
    def _patch_http(self, responder):
        original = tr._http_json  # captured per call so a second patch can't clobber the restore
        tr._http_json = responder
        self.addCleanup(lambda: setattr(tr, "_http_json", original))

    def test_google_web_parses_segments_and_detected_language(self):
        self._patch_http(lambda *a, **k: [[["Bonjour. ", "Hello. ", None], ["Ça va ?", "How are you?", None]],
                                          None, "en"])
        self.assertEqual(tr._google_web("Hello. How are you?", "auto", "fr"), ("Bonjour. Ça va ?", "en"))

    def test_google_web_rejects_garbage_and_empty(self):
        self._patch_http(lambda *a, **k: {"unexpected": True})
        with self.assertRaises(tr.EngineError):
            tr._google_web("x", "en", "fr")
        self._patch_http(lambda *a, **k: [[[None, "x"]], None, "en"])
        with self.assertRaises(tr.EngineError):
            tr._google_web("x", "en", "fr")

    def test_mymemory_success_quota_and_codes(self):
        calls = []
        def ok(method, url, params=None, **k):
            calls.append(params["langpair"])
            return {"responseStatus": 200, "responseData": {"translatedText": "Bonjour &amp; salut"}}
        self._patch_http(ok)
        self.assertEqual(tr._mymemory("Hello", "iw", "fr")[0], "Bonjour & salut")
        self.assertEqual(calls, ["he|fr"])  # Hebrew code mapped for MyMemory
        self._patch_http(lambda *a, **k: {"responseStatus": 200, "responseData": {
            "translatedText": "MYMEMORY WARNING: YOU USED ALL AVAILABLE FREE TRANSLATIONS FOR TODAY"}})
        with self.assertRaises(tr.EngineError):
            tr._mymemory("Hello", "en", "fr")
        with self.assertRaises(tr.EngineError):
            tr._mymemory("Hello", "auto", "fr")

    def test_google_cloud_engine_parses_and_unescapes(self):
        self._patch_http(lambda *a, **k: {"data": {"translations": [
            {"translatedText": "Caf&#39;e", "detectedSourceLanguage": "en"}]}})
        fn = tr._make_google_cloud("KEY")
        self.assertEqual(fn("Cafe", "auto", "fr"), ("Caf'e", "en"))

    def test_default_engine_order(self):
        self.assertEqual([e.name for e in tr.default_engines()], ["Google Translate", "MyMemory"])
        names = [e.name for e in tr.default_engines("k")]
        self.assertEqual(names[0], "Google Cloud Translation API")
        self.assertLessEqual(tr.default_engines()[-1].max_chars, 500)

    def test_http_layer_retries_then_succeeds_and_hides_secrets(self):
        import requests
        orig_req, orig_sleep = requests.request, tr.time.sleep
        seq = iter([requests.ConnectionError("https://x/?key=SECRET"), 429, 200])
        class R:
            def __init__(s, code): s.status_code, s.text = code, "t"
            def json(s): return {"ok": 1}
        def fake(*a, **k):
            v = next(seq)
            if isinstance(v, Exception): raise v
            return R(v)
        requests.request, tr.time.sleep = fake, lambda s: None
        try:
            self.assertEqual(tr._http_json("GET", "u"), {"ok": 1})
            requests.request = lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("https://x/?key=SECRET"))
            with self.assertRaises(tr.EngineError) as cm:
                tr._http_json("GET", "u")
            self.assertNotIn("SECRET", str(cm.exception))
            requests.request = lambda *a, **k: types.SimpleNamespace(
                status_code=200, text="<html>", json=lambda: (_ for _ in ()).throw(ValueError("x")))
            with self.assertRaises(tr.EngineError) as cm:
                tr._http_json("GET", "u")
            self.assertIn("not JSON", str(cm.exception))
        finally:
            requests.request, tr.time.sleep = orig_req, orig_sleep

    def test_health_check_reports_ok_and_failed(self):
        def good(*a): return "Bonjour"
        def bad(*a): raise tr.EngineError("blocked")
        rows = tr.check_health([self.eng("Good", good), self.eng("Bad", bad)], include_tts=False)
        by = {r["component"]: r for r in rows}
        self.assertEqual(by["Good"]["status"], "OK")
        self.assertEqual((by["Bad"]["status"], by["Bad"]["detail"]), ("FAILED", "blocked"))
        self.assertIn("Language detection (offline)", by)


if __name__ == "__main__":
    unittest.main()
