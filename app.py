"""Task 1 - Polyglot AI Translator."""
import html
from datetime import datetime

import streamlit as st

from core import translation as tr
from core.ui import header, inject_css, page_config

page_config(page_title="Polyglot AI Translator", page_icon="🌐", layout="wide")
inject_css("""
.result-box{padding:1rem 1.2rem;border-radius:12px;border-left:5px solid #6c5ce7;
            background:rgba(108,92,231,.08);white-space:pre-wrap;font-size:1.05rem;line-height:1.6}
""")

NAMES = sorted(tr.LANGUAGES)
AUTO = "Auto-detect"

ss = st.session_state
ss.setdefault("src_lang", AUTO)
ss.setdefault("tgt_lang", "Hindi")
ss.setdefault("history", [])
ss.setdefault("result", None)
ss.setdefault("audio", None)


def swap_languages() -> None:
    """Swap source/target; when source is Auto-detect use what was detected last time."""
    src, tgt = ss.src_lang, ss.tgt_lang
    if src == AUTO:
        res = ss.result
        if not res:
            return
        src = res.source_name
    ss.src_lang, ss.tgt_lang = tgt, src
    res = ss.result
    if res:  # carry the translation over as the new input text
        ss.input_text = res.text
        ss.result, ss.audio = None, None


def run_translation(text: str, src_name: str, tgt_name: str):
    src = "auto" if src_name == AUTO else tr.LANGUAGES[src_name]
    result = tr.translate(text, src, tr.LANGUAGES[tgt_name])
    ss.result, ss.audio = result, None
    ss.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"), "src": result.source_name,
        "tgt": result.target_name, "in": text, "out": result.text, "engine": result.engine})
    del ss.history[50:]
    return result


def show_result(res: tr.TranslationResult, key: str) -> None:
    detected = " (auto-detected)" if res.detected else ""
    st.caption(f"{res.source_name}{detected} → **{res.target_name}** · engine: {res.engine}"
               + (f" · {res.chunks} chunks" if res.chunks > 1 else ""))
    st.markdown(f'<div class="result-box">{html.escape(res.text)}</div>', unsafe_allow_html=True)
    with st.expander("📋 Copy text"):
        st.code(res.text, language=None)  # st.code has a built-in copy button
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download .txt", res.text, file_name=f"translation_{res.target_code}.txt",
                       key=f"dl_{key}")
    if c2.button("🔊 Listen", key=f"tts_{key}"):
        try:
            with st.spinner("Generating speech..."):
                ss.audio = tr.speak(res.text, res.target_code)
        except Exception as exc:
            ss.audio = None
            st.warning(f"Text-to-speech isn't available for this language right now ({exc}).")
    if ss.audio:
        st.audio(ss.audio, format="audio/mp3")


# --------------------------------------------------------------------------- #
header("🌐 Polyglot AI Translator",
       f"CodeAlpha AI Internship · Task 1 · {len(NAMES)} languages · auto-detect · speech · files")

with st.sidebar:
    st.subheader("About")
    st.write("Text is split into safe-sized chunks, sent to **Google Translate**, and "
             "automatically retried on **MyMemory** if the first engine fails.")
    st.markdown("**Features**\n- Auto language detection\n- Swap languages\n- Long-text & file support\n"
                "- Copy, download, text-to-speech\n- Session history")
    st.caption("Built with Python, Streamlit, deep-translator, langdetect, gTTS.")

tab_text, tab_file, tab_hist = st.tabs(["✍️ Text", "📄 File", f"🕘 History ({len(ss.history)})"])

with tab_text:
    c1, c2, c3 = st.columns([5, 1, 5])
    c1.selectbox("From", [AUTO] + NAMES, key="src_lang")
    c2.write("")
    c2.button("⇄", on_click=swap_languages, help="Swap languages", key="swap")
    c3.selectbox("To", NAMES, key="tgt_lang")

    text = st.text_area("Text to translate", key="input_text", height=180,
                        placeholder="Type or paste text here (paragraphs and long text are fine)...")
    st.caption(f"{len(text):,} characters")

    if st.button("Translate", type="primary", key="go"):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            try:
                with st.spinner("Translating..."):
                    run_translation(text, ss.src_lang, ss.tgt_lang)
            except tr.TranslationError as exc:
                st.error(f"Translation failed on every engine. Check your connection and retry.\n\n{exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
    if ss.result:
        show_result(ss.result, "main")

with tab_file:
    st.write("Upload a **.txt** or **.md** file; the whole document is translated chunk by chunk.")
    up = st.file_uploader("Document", type=["txt", "md"], key="file_up")
    tgt_file = st.selectbox("Translate document to", NAMES, index=NAMES.index("English"), key="file_tgt")
    if up is not None:
        raw = up.getvalue().decode("utf-8", errors="replace")
        st.caption(f"{len(raw):,} characters")
        with st.expander("Preview original"):
            st.text(raw[:1500] + ("…" if len(raw) > 1500 else ""))
        if st.button("Translate document", type="primary", key="go_file"):
            try:
                with st.spinner("Translating document..."):
                    res = tr.translate(raw, "auto", tr.LANGUAGES[tgt_file])
                ss.file_result = res
            except Exception as exc:
                st.error(f"Could not translate the file: {exc}")
    if ss.get("file_result"):
        show_result(ss.file_result, "file")

with tab_hist:
    if not ss.history:
        st.info("Your translations in this session will appear here.")
    else:
        if st.button("Clear history"):
            ss.history = []
            st.rerun()
        for item in ss.history:
            with st.expander(f"{item['time']} · {item['src']} → {item['tgt']} · {item['in'][:40]}"):
                st.markdown(f"**Original**\n\n{item['in']}")
                st.markdown(f"**Translation** ({item['engine']})\n\n{item['out']}")
