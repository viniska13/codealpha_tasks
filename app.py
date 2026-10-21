"""Lexicon - a translation desk. CodeAlpha AI Internship, Task 1."""
import html
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import streamlit as st

from core import translation as tr

st.set_page_config(page_title="Lexicon · Translation Desk", page_icon="文", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap');
:root{--ink:#1b1a17;--paper:#f6f1e7;--card:#fbf8f1;--rule:#d9d0be;--accent:#c8452d;--mute:#6f685a}
html,body,.stApp{font-family:'IBM Plex Sans',sans-serif}
h1,h2,h3{font-family:'Fraunces',serif!important;font-weight:600!important;letter-spacing:-.01em}
footer{visibility:hidden}
.block-container{padding-top:2.2rem;max-width:1180px}

.lx-mast{display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;flex-wrap:wrap;
         border-bottom:2px solid var(--ink);padding-bottom:.7rem;margin-bottom:1.4rem}
.lx-word{font-family:'Fraunces',serif;font-size:2.7rem;line-height:1;font-weight:600}
.lx-word i{color:var(--accent);font-style:normal}
.lx-tag{font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:var(--mute);
        text-transform:uppercase;letter-spacing:.14em;margin-top:.45rem}
.lx-stats{display:flex;gap:1.8rem}
.lx-stat b{display:block;font-family:'Fraunces',serif;font-size:1.6rem;font-weight:600;line-height:1}
.lx-stat span{font-family:'IBM Plex Mono',monospace;font-size:.62rem;text-transform:uppercase;
              letter-spacing:.12em;color:var(--mute)}

.lx-label{font-family:'IBM Plex Mono',monospace;font-size:.68rem;text-transform:uppercase;
          letter-spacing:.16em;color:var(--mute);border-bottom:1px solid var(--rule);
          padding-bottom:.35rem;margin-bottom:.6rem}
.lx-out{background:var(--card);border:1px solid var(--rule);border-left:3px solid var(--accent);
        padding:1rem 1.2rem;min-height:262px;font-family:'Fraunces',serif;font-size:1.22rem;
        line-height:1.55;white-space:pre-wrap;word-break:break-word}
.lx-out.empty{color:var(--mute);font-family:'IBM Plex Sans',sans-serif;font-size:.95rem;font-style:italic}
.lx-meta{font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:var(--mute);margin:.5rem 0 .2rem}
.lx-inst{border-top:1px solid var(--ink);padding-top:.5rem;margin-top:.4rem}
.lx-inst span{font-family:'IBM Plex Mono',monospace;font-size:.62rem;text-transform:uppercase;
              letter-spacing:.14em;color:var(--mute)}
.lx-inst b{display:block;font-family:'Fraunces',serif;font-size:1.35rem;font-weight:600;margin-top:.15rem}
.lx-foot{font-family:'IBM Plex Mono',monospace;font-size:.68rem;color:var(--mute);
         border-top:1px solid var(--rule);margin-top:2.2rem;padding-top:.7rem;letter-spacing:.04em}

.stButton>button,.stDownloadButton>button{border-radius:2px;font-family:'IBM Plex Mono',monospace;
        font-size:.76rem;letter-spacing:.06em;text-transform:uppercase}
.stTabs [data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace;text-transform:uppercase;
        letter-spacing:.1em;font-size:.76rem}
.stTextArea textarea{background:var(--card);font-size:1rem;line-height:1.55}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

NAMES = sorted(tr.LANGUAGES)
AUTO = "Auto-detect"

PHRASES = {
    "Greetings": ["Hello, how are you?", "Nice to meet you.", "Good morning!", "Thank you very much.",
                  "See you tomorrow.", "Please excuse me."],
    "Travel": ["Where is the railway station?", "How much does this ticket cost?",
               "Could you take me to this address?", "Is this seat taken?",
               "What time does the bus leave?", "I have a reservation."],
    "Food": ["A table for two, please.", "What do you recommend?", "I am vegetarian.",
             "Could I have the bill, please?", "Is this dish spicy?", "Water, please."],
    "Help": ["I need help.", "Please call a doctor.", "I lost my passport.",
             "Where is the nearest hospital?", "I don't understand.", "Could you speak more slowly?"],
    "Study & work": ["What is the deadline?", "Could you share the notes?",
                     "I would like to schedule a meeting.", "Can you explain this again?",
                     "I will send the report today.", "Let us review the results together."],
}

ss = st.session_state
ss.setdefault("src_lang", AUTO)
ss.setdefault("tgt_lang", "Hindi")
ss.setdefault("input_text", "")
ss.setdefault("history", [])
ss.setdefault("result", None)
ss.setdefault("file_result", None)
ss.setdefault("cmp_results", None)
ss.setdefault("audio", None)


@st.cache_resource
def phrase_cache() -> dict:
    return {}


# ----------------------------- helpers ------------------------------------- #
def log(res: tr.TranslationResult, text: str, conf=None) -> None:
    ss.history.append({"time": datetime.now(), "from": res.source_name, "to": res.target_name,
                       "chars_in": len(text), "chars_out": len(res.text), "ms": res.elapsed_ms,
                       "engine": res.engine, "confidence": conf, "text": text[:60]})
    del ss.history[:-200]


def swap_languages() -> None:
    src, tgt = ss.src_lang, ss.tgt_lang
    if src == AUTO:
        if not ss.result:
            return
        src = ss.result.source_name
    ss.src_lang, ss.tgt_lang = tgt, src
    if ss.result:
        ss.input_text = ss.result.text
        ss.result, ss.audio = None, None


def run_translation(text: str) -> None:
    src = "auto" if ss.src_lang == AUTO else tr.LANGUAGES[ss.src_lang]
    guess = tr.detect_with_confidence(text) if src == "auto" else None
    res = tr.translate(text, src, tr.LANGUAGES[ss.tgt_lang])
    ss.result, ss.audio = res, None
    log(res, text, guess[1] if guess else None)


def stat_html(label: str, value: str) -> str:
    return f'<div class="lx-inst"><span>{html.escape(label)}</span><b>{html.escape(str(value))}</b></div>'


def render_output(res: tr.TranslationResult, key: str) -> None:
    arrow = f"{res.source_name}{' (detected)' if res.detected else ''} → {res.target_name}"
    st.markdown(f'<div class="lx-out">{html.escape(res.text)}</div>'
                f'<div class="lx-meta">{html.escape(arrow)} · {res.engine} · {res.elapsed_ms} ms</div>',
                unsafe_allow_html=True)
    a, b = st.columns(2)
    a.download_button("Download .txt", res.text, file_name=f"lexicon_{res.target_code}.txt", key=f"dl_{key}")
    if b.button("Listen", key=f"tts_{key}"):
        try:
            with st.spinner("Generating speech"):
                ss.audio = tr.speak(res.text, res.target_code)
        except Exception as exc:
            ss.audio = None
            st.warning(f"Speech isn't available for this language right now ({exc}).")
    if ss.audio:
        st.audio(ss.audio, format="audio/mp3")
    with st.expander("Copy text"):
        st.code(res.text, language=None)


def cached_phrase(cache: dict, text: str, code: str):
    if (text, code) not in cache:
        try:
            cache[(text, code)] = tr.translate(text, "en", code).text
        except Exception:
            return None
    return cache[(text, code)]


# ----------------------------- masthead ------------------------------------ #
n_req = len(ss.history)
n_chars = sum(h["chars_in"] for h in ss.history)
n_langs = len({h["to"] for h in ss.history})
st.markdown(
    f'<div class="lx-mast"><div><div class="lx-word">Lex<i>i</i>con</div>'
    f'<div class="lx-tag">Translation desk · {len(NAMES)} languages · CodeAlpha AI Internship · Task 1</div></div>'
    f'<div class="lx-stats"><div class="lx-stat"><b>{n_req}</b><span>Translations</span></div>'
    f'<div class="lx-stat"><b>{n_chars:,}</b><span>Characters</span></div>'
    f'<div class="lx-stat"><b>{n_langs}</b><span>Target languages</span></div></div></div>',
    unsafe_allow_html=True)

tab_tr, tab_cmp, tab_pb, tab_log = st.tabs(["Translate", "Compare", "Phrasebook", "Session"])

# ----------------------------- translate ----------------------------------- #
with tab_tr:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="lx-label">Source</div>', unsafe_allow_html=True)
        st.selectbox("From", [AUTO] + NAMES, key="src_lang", label_visibility="collapsed")
        text = st.text_area("Source text", key="input_text", height=262, label_visibility="collapsed",
                            placeholder="Type or paste text. Paragraphs and long passages are fine.")
        words = len(text.split())
        info = f"{len(text):,} chars · {words} words" + (f" · ~{max(1, round(words / 200))} min read" if words else "")
        guess = tr.detect_with_confidence(text) if (ss.src_lang == AUTO and text.strip()) else None
        if guess:
            info += f" · looks like {tr.CODE_TO_NAME[guess[0].lower()]} ({guess[1]:.0%})"
        st.markdown(f'<div class="lx-meta">{info}</div>', unsafe_allow_html=True)

        b1, b2, _ = st.columns([1, 1, 1])
        go = b1.button("Translate", type="primary", key="go")
        b2.button("Swap ⇄", on_click=swap_languages, key="swap")
        if go:
            if not text.strip():
                st.warning("Enter some text first.")
            else:
                try:
                    with st.spinner("Translating"):
                        run_translation(text)
                except tr.TranslationError as exc:
                    st.error(f"Every translation engine failed. Check your connection and retry.\n\n{exc}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
    with right:
        st.markdown('<div class="lx-label">Translation</div>', unsafe_allow_html=True)
        st.selectbox("To", NAMES, key="tgt_lang", label_visibility="collapsed")
        if ss.result:
            render_output(ss.result, "main")
        else:
            st.markdown('<div class="lx-out empty">The translation appears here.</div>', unsafe_allow_html=True)

    if ss.result:
        r = ss.result
        last = ss.history[-1] if ss.history else {}
        conf = last.get("confidence")
        cols = st.columns(5)
        cols[0].markdown(stat_html("Source language", r.source_name), unsafe_allow_html=True)
        cols[1].markdown(stat_html("Detection confidence", f"{conf:.0%}" if conf else "n/a"), unsafe_allow_html=True)
        cols[2].markdown(stat_html("Length ratio", f"{len(r.text) / max(len(text or r.text), 1):.2f}×"),
                         unsafe_allow_html=True)
        cols[3].markdown(stat_html("Latency", f"{r.elapsed_ms} ms"), unsafe_allow_html=True)
        cols[4].markdown(stat_html("Chunks", r.chunks), unsafe_allow_html=True)

    with st.expander("Translate a document (.txt / .md)"):
        up = st.file_uploader("Document", type=["txt", "md"], key="file_up", label_visibility="collapsed")
        tgt_file = st.selectbox("Document language", NAMES, index=NAMES.index("English"), key="file_tgt")
        if up is not None:
            raw = up.getvalue().decode("utf-8", errors="replace")
            st.caption(f"{len(raw):,} characters. Long files are split into safe-sized chunks.")
            if st.button("Translate document", key="go_file"):
                try:
                    with st.spinner("Translating document"):
                        res = tr.translate(raw, "auto", tr.LANGUAGES[tgt_file])
                    ss.file_result = res
                    log(res, raw)
                except Exception as exc:
                    st.error(f"Could not translate the file: {exc}")
        if ss.file_result:
            render_output(ss.file_result, "file")

# ----------------------------- compare ------------------------------------- #
with tab_cmp:
    st.markdown('<div class="lx-label">One text, several languages</div>', unsafe_allow_html=True)
    ctext = st.text_area("Compare text", key="cmp_text", height=120, label_visibility="collapsed",
                         placeholder="Type a sentence to see it in up to four languages at once.")
    targets = st.multiselect("Translate into (up to four)", NAMES, default=["French", "Spanish", "German"],
                             max_selections=4, key="cmp_targets")
    if st.button("Compare", type="primary", key="go_cmp"):
        if not ctext.strip() or not targets:
            st.warning("Enter text and choose at least one language.")
        else:
            def one(name):
                try:
                    return name, tr.translate(ctext, "auto", tr.LANGUAGES[name]), None
                except Exception as exc:
                    return name, None, str(exc)
            with st.spinner("Translating"):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    ss.cmp_results = list(pool.map(one, targets))
            for _, res, _err in ss.cmp_results:
                if res:
                    log(res, ctext)
    if ss.cmp_results:
        cols = st.columns(len(ss.cmp_results))
        for col, (name, res, err) in zip(cols, ss.cmp_results):
            body = html.escape(res.text) if res else f"Failed: {html.escape(err or '')}"
            col.markdown(f'<div class="lx-label">{html.escape(name)}</div>'
                         f'<div class="lx-out" style="min-height:120px;font-size:1.05rem">{body}</div>',
                         unsafe_allow_html=True)

# ----------------------------- phrasebook ---------------------------------- #
with tab_pb:
    st.markdown('<div class="lx-label">Everyday phrases, translated on demand</div>', unsafe_allow_html=True)
    p1, p2 = st.columns([3, 2])
    cat = p1.radio("Category", list(PHRASES), horizontal=True, key="pb_cat")
    pb_lang = p2.selectbox("Language", NAMES, index=NAMES.index("Kannada"), key="pb_lang")
    code = tr.LANGUAGES[pb_lang]
    phrases = PHRASES[cat]
    pcache = phrase_cache()  # fetched on the main thread; workers only touch the dict
    with st.spinner("Loading phrases"):
        with ThreadPoolExecutor(max_workers=6) as pool:
            outs = list(pool.map(lambda p: cached_phrase(pcache, p, code), phrases))
    st.dataframe(pd.DataFrame({"English": phrases, pb_lang: [o or "-" for o in outs]}),
                 hide_index=True)
    pick = st.selectbox("Hear a phrase", range(len(phrases)), format_func=lambda i: phrases[i], key="pb_pick")
    if st.button("Play", key="pb_play"):
        if outs[pick]:
            try:
                st.audio(tr.speak(outs[pick], code), format="audio/mp3")
            except Exception as exc:
                st.warning(f"Speech isn't available right now ({exc}).")
        else:
            st.warning("That phrase could not be translated.")

# ----------------------------- session ------------------------------------- #
with tab_log:
    if not ss.history:
        st.info("Everything you translate in this session is logged here, with timings.")
    else:
        df = pd.DataFrame(ss.history)
        df["time"] = df["time"].dt.strftime("%H:%M:%S")
        m = st.columns(4)
        m[0].markdown(stat_html("Requests", len(df)), unsafe_allow_html=True)
        m[1].markdown(stat_html("Median latency", f"{int(df['ms'].median())} ms"), unsafe_allow_html=True)
        m[2].markdown(stat_html("Characters in", f"{int(df['chars_in'].sum()):,}"), unsafe_allow_html=True)
        m[3].markdown(stat_html("Most used target", df["to"].mode().iat[0]), unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")
        c1.markdown('<div class="lx-label">Requests by target language</div>', unsafe_allow_html=True)
        c1.bar_chart(df.groupby("to").size().rename("requests"), color="#c8452d")
        c2.markdown('<div class="lx-label">Latency per request (ms)</div>', unsafe_allow_html=True)
        c2.line_chart(df["ms"].rename("ms"), color="#1f3d36")

        st.markdown('<div class="lx-label">Ledger</div>', unsafe_allow_html=True)
        ledger = df.rename(columns={"chars_in": "in", "chars_out": "out"})[
            ["time", "from", "to", "in", "out", "ms", "engine", "text"]].iloc[::-1]
        st.dataframe(ledger, hide_index=True)
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("Export CSV", ledger.to_csv(index=False), "lexicon_session.csv", "text/csv")
        if d2.button("Clear session", key="clear"):
            ss.history, ss.result, ss.file_result, ss.cmp_results = [], None, None, None
            st.rerun()

st.markdown('<div class="lx-foot">Engines: Google Translate, falling back to MyMemory · long text is split '
            'at paragraph and sentence boundaries · language detection runs offline (langdetect)</div>',
            unsafe_allow_html=True)
