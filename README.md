# CodeAlpha AI Internship: Three Instruments

[![tests](https://github.com/viniska13/codealpha_tasks/actions/workflows/ci.yml/badge.svg)](https://github.com/viniska13/codealpha_tasks/actions/workflows/ci.yml)

Three small apps built to **fail gracefully**: a translator that survives a blocked API, a chatbot that survives an exhausted quota, and a vision tool that turns raw detections into something a person can act on.

| 47 | 3 | 20 | 80 | 5 |
|:--:|:--:|:--:|:--:|:--:|
| languages | translation engines in a fallback chain | unit tests, no network needed | COCO classes detected | chatbot model fallbacks |

| Task | Instrument | Live |
|---|---|---|
| 1 | **Lexicon**, a translation desk | [open](https://codealpha-translator.streamlit.app/) |
| 2 | **Assistant**, a conversational chatbot | [open](https://codealpha-chatbotai.streamlit.app/) |
| 4 | **Vision Lab**, object detection and scene analysis | [open](https://codealpha-objectdetection.streamlit.app/) |

<!-- Add screenshots to docs/ and uncomment:
![Lexicon](docs/lexicon.png) ![Assistant](docs/assistant.png) ![Vision Lab](docs/vision.png)
-->

**Reading this in two minutes?** Start with [`core/translation.py`](core/translation.py) (the engine chain), [`tests/test_translation.py`](tests/test_translation.py) (what is proven), `call_gemini` in [`chatbot.py`](chatbot.py) (model fallback), and `build_table` / `draw_detections` in [`object_detection.py`](object_detection.py) (one table, many views).

---

## 1 · Lexicon: translation that does not break

**Problem.** Free translation endpoints get rate-limited, return captcha pages from cloud servers, or silently return nothing. A translator that just calls one of them looks fine until it is shown to someone.

**Built.** A backend that treats failure as the normal case.
- Direct JSON calls with timeouts and retries (no HTML scraping).
- An ordered engine chain: Google Cloud API (if a key is set) → Google web JSON → MyMemory.
- Each engine gets chunks sized for its own limit (about 4,500 characters vs about 450).
- A health check that probes every engine, the language detector and speech, and reports OK or FAILED with the reason.
- 47 languages, offline detection with a confidence score, side-by-side comparison across four languages, a phrasebook with audio, document translation, and a session ledger with latency charts and CSV export.

**Proof.** 20 unit tests cover chunking, fallback order, empty and malformed responses, MyMemory quota warnings, retry with backoff, and the guarantee that API keys never appear in error messages. CI runs them on every push.

## 2 · Assistant: a chatbot that explains its failures

**Problem.** Free-tier AI APIs run out of quota or go busy exactly when you demo them.

**Built.** Language models are tried in order from a fallback list; a missing model, exhausted quota or busy server moves to the next one, and the model that worked is remembered for the session. The conversation keeps the last 20 turns of context. Keys come from Streamlit secrets or environment variables only. Errors become plain sentences ("free limit used up, wait a few minutes") rather than stack traces.

## 3 · Vision Lab: detections as decisions

**Problem.** Bounding boxes alone do not help anyone decide anything.

**Built.** YOLOv8 output becomes one tidy table (class, confidence, box, area %), and every view derives from it.
- **HUD overlay** with adjustable confidence, IoU, input size and class filters.
- **Privacy blur** that blurs chosen classes (people by default) before a photo is shared.
- **Density map** that weights object centres by confidence.
- **Scene summary** in plain language, with a rule-based scene guess that says it is an inference.
- **Position grid** and CSV / PNG export.

Uploads honour EXIF orientation, are downscaled to 1,280 px on the long side, and results are cached per image and settings. Inference
