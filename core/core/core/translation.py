"""Translation engine for Task 1: chunking, auto-detect, engine fallback, text-to-speech."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

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
MAX_CHUNK = 4500  # Google limit is 5000 chars per request


class TranslationError(RuntimeError):
    """Raised when every configured engine failed."""


@dataclass
class TranslationResult:
    text: str
    source_code: str
    target_code: str
    engine: str
    chunks: int = 1
    detected: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def source_name(self) -> str:
        return CODE_TO_NAME.get(self.source_code.lower(), self.source_code)

    @property
    def target_name(self) -> str:
        return CODE_TO_NAME.get(self.target_code.lower(), self.target_code)


# ---------------------------- chunking ------------------------------------- #
_SENTENCE_END = re.compile(r"(?<=[.!?।。！？])\s+")


def chunk_text(text: str, limit: int = MAX_CHUNK) -> list[str]:
    """Split text into pieces <= limit chars (paragraph -> sentence -> word)."""
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


# ------------------------ language detection ------------------------------- #
def detect_language(text: str) -> Optional[str]:
    """Google-style language code, or None if unavailable / too short."""
    sample = text.strip()
    if len(sample) < 3:
        return None
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        code = detect(sample).lower()
    except Exception:
        return None
    code = _LANGDETECT_ALIASES.get(code, code)
    return code if code.lower() in CODE_TO_NAME else None


# ------------------------------ engines ------------------------------------ #
Engine = Callable[[str, str, str], str]


def google_engine(text: str, source: str, target: str) -> str:
    from deep_translator import GoogleTranslator

    return GoogleTranslator(source=source, target=target).translate(text)


def mymemory_engine(text: str, source: str, target: str) -> str:
    from deep_translator import MyMemoryTranslator

    if source == "auto":
        raise ValueError("MyMemory needs an explicit source language")
    return MyMemoryTranslator(source=source, target=target).translate(text)


DEFAULT_ENGINES: list[tuple[str, Engine]] = [
    ("Google Translate", google_engine),
    ("MyMemory", mymemory_engine),
]


def translate(text: str, source: str, target: str,
              engines: Optional[Sequence[tuple[str, Engine]]] = None) -> TranslationResult:
    """Translate text. `source` may be "auto". Falls back to the next engine on failure."""
    if not text or not text.strip():
        raise ValueError("Nothing to translate")
    engines = list(engines or DEFAULT_ENGINES)

    detected = False
    resolved_source = source
    if source == "auto":
        guess = detect_language(text)
        if guess:
            resolved_source, detected = guess, True

    if resolved_source == target:
        return TranslationResult(text, resolved_source, target, "none (same language)",
                                 chunks=0, detected=detected)

    chunks = chunk_text(text)
    errors: list[str] = []
    for name, engine in engines:
        src_for_engine = source if name == "Google Translate" else resolved_source
        try:
            out = []
            for c in chunks:
                out.append(engine(c, src_for_engine, target) if c.strip() else c)
            return TranslationResult(
                "\n".join(o or "" for o in out), resolved_source, target, name,
                chunks=len([c for c in chunks if c.strip()]), detected=detected, errors=errors)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise TranslationError("All translation engines failed -> " + " | ".join(errors))


# --------------------------- text to speech -------------------------------- #
def speak(text: str, lang_code: str, max_chars: int = 1500) -> bytes:
    """MP3 bytes of `text` spoken in `lang_code` (needs gTTS + internet)."""
    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text=text[:max_chars], lang=lang_code).write_to_fp(buf)
    return buf.getvalue()
