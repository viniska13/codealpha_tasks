"""Translation backend for Task 1.

Engines are tried in order until one succeeds:

1. Google Cloud Translation API   - official, used only if an API key is configured
2. Google Translate (web JSON)    - free endpoint, returns the detected source language
3. MyMemory                       - free REST API, independent of Google

Every engine is a plain HTTP call with timeouts and retries (no HTML scraping), long text
is split per engine limit, and :func:`check_health` reports exactly which parts work.
"""
from __future__ import annotations

import html
import io
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

# Display name -> Google Translate language code
LANGUAGES: dict[str, str] = {
    "Afrikaans": "af", "Arabic": "ar", "Bengali": "bn", "Bulgarian": "bg",
    "Catalan": "ca", "Chinese (Simplified)": "zh-CN", "Chinese (Traditional)": "zh-TW",
    "Croatian": "hr", "Czech": "cs", "Danish": "da", "Dutch": "nl", "English": "en",
    "Finnish": "fi", "French": "fr", "German": "de", "Greek": "el", "Gujarati": "gu",
    "Hebrew": "iw", "Hindi": "hi", "Hungarian": "hu", "Indonesian": "id",
    "Italian": "it", "Japanese": "ja", "Kannada": "kn", "Korean": "ko",
    "Malayalam": "ml", "Marathi": "mr", "Nepali": "ne", "Norwegian": "no",
    "Persian": "fa", "Polish": "pl", "Portuguese": "pt", "Punjabi": "pa",
    "Romanian": "ro", "Russian": "ru", "Serbian": "sr", "Slovak": "sk",
    "Spanish": "es", "Swahili": "sw", "Swedish": "sv", "Tamil": "ta", "Telugu": "te",
    "Thai": "th", "Turkish": "tr", "Ukrainian": "uk", "Urdu": "ur",
    "Vietnamese": "vi",
}
CODE_TO_NAME = {code.lower(): name for name, code in LANGUAGES.items()}

_LANGDETECT_ALIASES = {"zh-cn": "zh-CN", "zh-tw": "zh-TW", "he": "iw"}
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LexiconTranslator/1.0)"}


class TranslationError(RuntimeError):
    """Raised when every configured engine failed."""


class EngineError(RuntimeError):
    """One engine failed (network, quota, bad response ...)."""


@dataclass
class TranslationResult:
    text: str
    source_code: str
    target_code: str
    engine: str
    chunks: int = 1
    detected: bool = False
    errors: list[str] = field(default_factory=list)
    elapsed_ms: int = 0

    @property
    def source_name(self) -> str:
        return CODE_TO_NAME.get(self.source_code.lower(), self.source_code)

    @property
    def target_name(self) -> str:
        return CODE_TO_NAME.get(self.target_code.lower(), self.target_code)


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
_SENTENCE_END = re.compile(r"(?<=[.!?।。！？])\s+")


def chunk_text(text: str, limit: int = 4500) -> list[str]:
    """Split text into pieces <= limit chars (paragraph -> sentence -> word).

    Every input line becomes at least one chunk (blank lines stay as empty chunks), and
    :func:`translate` re-joins chunks with newlines, so line structure survives.
    """
    if limit < 10:
        raise ValueError("limit too small")
    chunks: list[str] = []
    for paragraph in text.split("\n"):
        if len(paragraph) <= limit:
            chunks.append(paragraph)
            continue
        current = ""
        for sentence in _SENTENCE_END.split(paragraph):
            for piece in _hard_split(sentence, limit):
                if current and len(current) + 1 + len(piece) > limit:
                    chunks.append(current)
                    current = piece
                else:
                    current = f"{current} {piece}".strip() if current else piece
        if current:
            chunks.append(current)
    return chunks


def _hard_split(s: str, limit: int) -> list[str]:
    if len(s) <= limit:
        return [s]
    out, cur = [], ""
    for word in s.split(" "):
        while len(word) > limit:
            if cur:
                out.append(cur)
                cur = ""
            out.append(word[:limit])
            word = word[limit:]
        if cur and len(cur) + 1 + len(word) > limit:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        out.append(cur)
    return out


# --------------------------------------------------------------------------- #
# Language detection (offline)
# --------------------------------------------------------------------------- #
def _normalise_code(code: Optional[str]) -> Optional[str]:
    """Map any engine/detector code to one of our Google-style codes, else None."""
    if not code:
        return None
    code = _LANGDETECT_ALIASES.get(code.lower(), code)
    return next((c for c in LANGUAGES.values() if c.lower() == code.lower()), None)


def detect_with_confidence(text: str) -> Optional[tuple[str, float]]:
    """Best guess ``(google_code, probability)``, or None if unavailable / too short."""
    sample = text.strip()
    if len(sample) < 8:
        return None
    try:
        from langdetect import DetectorFactory, detect_langs

        DetectorFactory.seed = 0
        best = detect_langs(sample)[0]
    except Exception:
        return None
    code = _normalise_code(best.lang)
    return (code, float(best.prob)) if code else None


def detect_language(text: str) -> Optional[str]:
    guess = detect_with_confidence(text)
    return guess[0] if guess else None


# --------------------------------------------------------------------------- #
# HTTP layer (single place for timeouts, retries and error messages)
# --------------------------------------------------------------------------- #
def _http_json(method: str, url: str, *, params: Optional[dict] = None, data: Optional[dict] = None,
               timeout: tuple[float, float] = (5, 20), attempts: int = 3):
    """Perform an HTTP request and return parsed JSON, retrying transient failures."""
    import requests

    last: Optional[EngineError] = None
    for attempt in range(attempts):
        try:
            resp = requests.request(method, url, params=params, data=data, headers=_HEADERS, timeout=timeout)
        except requests.RequestException as exc:  # DNS, timeout, connection reset ...
            # only the class name: exception text can contain the full URL (and API keys)
            last = EngineError(f"network error ({exc.__class__.__name__})")
        else:
            if resp.status_code in (429, 500, 502, 503, 504):
                last = EngineError(f"HTTP {resp.status_code} (rate-limited or server busy)")
            elif resp.status_code >= 400:
                raise EngineError(f"HTTP {resp.status_code}: {resp.text[:140].strip()}")
            else:
                try:
                    return resp.json()
                except ValueError:
                    raise EngineError("response was not JSON (blocked, captcha or consent page?)")
        if attempt < attempts - 1:
            time.sleep(0.6 * (2 ** attempt))
    raise last or EngineError("request failed")


# --------------------------------------------------------------------------- #
# Engines.  fn(text, source, target) -> str  or  (str, detected_source_code | None)
# --------------------------------------------------------------------------- #
EngineFn = Callable[[str, str, str], Union[str, tuple]]


@dataclass(frozen=True)
class Engine:
    name: str
    fn: EngineFn
    max_chars: int = 4500        # per request
    supports_auto: bool = True   # can it detect the source language itself?


def _google_web(text: str, source: str, target: str):
    data = _http_json("POST", "https://translate.googleapis.com/translate_a/single",
                      params={"client": "gtx", "sl": source, "tl": target, "dt": "t"},
                      data={"q": text})
    try:
        out = "".join(seg[0] for seg in data[0] if seg and seg[0])
        detected = data[2] if len(data) > 2 and isinstance(data[2], str) else None
    except (TypeError, IndexError, KeyError):
        raise EngineError("unexpected response format")
    if not out.strip():
        raise EngineError("empty translation returned")
    return out, detected


def _make_google_cloud(api_key: str) -> EngineFn:
    def fn(text: str, source: str, target: str):
        form = {"q": text, "target": target, "format": "text"}
        if source != "auto":
            form["source"] = source
        data = _http_json("POST", "https://translation.googleapis.com/language/translate/v2",
                          params={"key": api_key}, data=form)
        try:
            item = data["data"]["translations"][0]
            return html.unescape(item["translatedText"]), item.get("detectedSourceLanguage")
        except (TypeError, IndexError, KeyError):
            raise EngineError("unexpected response format")
    return fn


_MYMEMORY_CODES = {"iw": "he"}


def _mymemory(text: str, source: str, target: str):
    if source == "auto":
        raise EngineError("needs an explicit source language")
    pair = f"{_MYMEMORY_CODES.get(source, source)}|{_MYMEMORY_CODES.get(target, target)}"
    data = _http_json("GET", "https://api.mymemory.translated.net/get", params={"q": text, "langpair": pair})
    try:
        status = int(data.get("responseStatus", 200))
        out = (data.get("responseData") or {}).get("translatedText") or ""
    except (AttributeError, TypeError, ValueError):
        raise EngineError("unexpected response format")
    if status != 200:
        raise EngineError(f"MyMemory error {status}: {data.get('responseDetails', '')}"[:160])
    if not out.strip() or "MYMEMORY WARNING" in out.upper():
        raise EngineError(out[:160] or "empty translation returned")
    return html.unescape(out), None


def default_engines(google_api_key: Optional[str] = None) -> list[Engine]:
    engines: list[Engine] = []
    if google_api_key:
        engines.append(Engine("Google Cloud Translation API", _make_google_cloud(google_api_key), 4500, True))
    engines.append(Engine("Google Translate", _google_web, 4500, True))
    engines.append(Engine("MyMemory", _mymemory, 450, False))  # MyMemory limit: 500 bytes
    return engines


# --------------------------------------------------------------------------- #
# Translate
# --------------------------------------------------------------------------- #
def translate(text: str, source: str, target: str, engines: Optional[Sequence[Engine]] = None) -> TranslationResult:
    """Translate ``text``. ``source`` may be ``"auto"``.

    The whole text goes through one engine (output never mixes engines); if that engine
    fails on any chunk the next engine retries the entire text.
    """
    if not text or not text.strip():
        raise ValueError("Nothing to translate")
    engines = list(engines) if engines else default_engines()

    resolved, detected = source, False
    if source == "auto":
        guess = detect_language(text)
        if guess:
            resolved, detected = guess, True

    if resolved == target:
        return TranslationResult(text, resolved, target, "none (same language)", chunks=0, detected=detected)

    errors: list[str] = []
    started = time.perf_counter()
    for eng in engines:
        src = "auto" if (source == "auto" and eng.supports_auto) else resolved
        if src == "auto" and not eng.supports_auto:
            errors.append(f"{eng.name}: could not detect the source language, please choose it manually")
            continue
        try:
            parts, engine_detected = [], None
            chunks = chunk_text(text, eng.max_chars)
            for c in chunks:
                if not c.strip():
                    parts.append(c)
                    continue
                result = eng.fn(c, src, target)
                out, det = result if isinstance(result, tuple) else (result, None)
                if not out:
                    raise EngineError("empty translation returned")
                parts.append(out)
                engine_detected = engine_detected or det
            final_src, was_detected = resolved, detected
            if source == "auto" and _normalise_code(engine_detected):
                final_src, was_detected = _normalise_code(engine_detected), True
            return TranslationResult(
                "\n".join(parts), final_src, target, eng.name,
                chunks=len([c for c in chunks if c.strip()]), detected=was_detected, errors=errors,
                elapsed_ms=int((time.perf_counter() - started) * 1000))
        except Exception as exc:
            errors.append(f"{eng.name}: {exc}")
    raise TranslationError(" | ".join(errors) or "no engines configured")


# --------------------------------------------------------------------------- #
# Text to speech
# --------------------------------------------------------------------------- #
def speak(text: str, lang_code: str, max_chars: int = 1500) -> bytes:
    """MP3 bytes of ``text`` spoken in ``lang_code`` (needs gTTS + internet)."""
    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text=text[:max_chars], lang=lang_code).write_to_fp(buf)
    data = buf.getvalue()
    if not data:
        raise RuntimeError("speech service returned no audio")
    return data


# --------------------------------------------------------------------------- #
# Health check: shows exactly which backend piece works
# --------------------------------------------------------------------------- #
def check_health(engines: Optional[Sequence[Engine]] = None, include_tts: bool = True) -> list[dict]:
    """Probe every engine plus detection and speech. Returns rows of name/status/ms/detail."""
    rows: list[dict] = []

    def timed(name: str, fn: Callable[[], str]) -> None:
        t0 = time.perf_counter()
        try:
            detail = fn()
            rows.append({"component": name, "status": "OK", "ms": int((time.perf_counter() - t0) * 1000),
                         "detail": detail})
        except Exception as exc:
            rows.append({"component": name, "status": "FAILED", "ms": int((time.perf_counter() - t0) * 1000),
                         "detail": str(exc)[:200]})

    for eng in (engines or default_engines()):
        def probe(e=eng):
            out = e.fn("Hello, how are you?", "en", "fr")
            return (out[0] if isinstance(out, tuple) else out)[:60]
        timed(eng.name, probe)

    def detection():
        guess = detect_with_confidence("This is a simple English sentence for testing.")
        if not guess:
            raise RuntimeError("langdetect is not installed or returned nothing")
        return f"{CODE_TO_NAME[guess[0].lower()]} ({guess[1]:.0%})"
    timed("Language detection (offline)", detection)

    if include_tts:
        timed("Text-to-speech (gTTS)", lambda: f"{len(speak('Hello', 'en'))} bytes of audio")
    return rows
