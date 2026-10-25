"""Docket - an explainable FAQ desk. CodeAlpha AI Internship, Task 2."""
import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core.faq_engine import STEMMER, FAQEngine, evaluate

st.set_page_config(page_title="Docket · FAQ Desk", page_icon="§", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap');
:root{--ink:#1b1a17;--card:#fbf8f1;--rule:#d9d0be;--accent:#1f3d36;--hot:#c8452d;--mute:#6f685a;--ok:#2f6b4f;--warn:#b7791f}
html,body,.stApp{font-family:'IBM Plex Sans',sans-serif}
h1,h2,h3{font-family:'Fraunces',serif!important;font-weight:600!important}
footer{visibility:hidden}
.block-container{padding-top:2.2rem;max-width:1280px}

.dk-mast{display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;flex-wrap:wrap;
         border-bottom:2px solid var(--ink);padding-bottom:.7rem;margin-bottom:1.3rem}
.dk-word{font-family:'Fraunces',serif;font-size:2.7rem;line-height:1;font-weight:600}
.dk-word i{color:var(--hot);font-style:normal}
.dk-tag{font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:var(--mute);text-transform:uppercase;
        letter-spacing:.14em;margin-top:.45rem}
.dk-stats{display:flex;gap:1.8rem}
.dk-stat b{display:block;font-family:'Fraunces',serif;font-size:1.6rem;font-weight:600;line-height:1}
.dk-stat span{font-family:'IBM Plex Mono',monospace;font-size:.62rem;text-transform:uppercase;
              letter-spacing:.12em;color:var(--mute)}

.dk-label{font-family:'IBM Plex Mono',monospace;font-size:.68rem;text-transform:uppercase;letter-spacing:.16em;
          color:var(--mute);border-bottom:1px solid var(--rule);padding-bottom:.35rem;margin-bottom:.7rem}
.dk-mono{font-family:'IBM Plex Mono',monospace;font-size:.74rem;color:var(--mute)}
.dk-chip{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:.76rem;background:var(--card);
         border:1px solid var(--rule);padding:.12rem .5rem;margin:0 .3rem .35rem 0}
.dk-pill{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:.7rem;letter-spacing:.12em;
         text-transform:uppercase;color:#fff;padding:.18rem .6rem}
.dk-pill.answer{background:var(--ok)} .dk-pill.suggest{background:var(--warn)} .dk-pill.fallback{background:var(--hot)}
.dk-gauge{position:relative;height:10px;background:var(--rule);margin:.9rem 0 .35rem}
.dk-gauge .fill{position:absolute;left:0;top:0;bottom:0;background:var(--accent)}
.dk-gauge .tick{position:absolute;top:-5px;bottom:-5px;width:2px;background:var(--hot)}
.dk-match{border-top:1px solid var(--rule);padding:.6rem 0}
.dk-match .q{font-family:'Fraunces',serif;font-size:1rem;line-height:1.3}
.dk-bar{display:flex;align-items:center;gap:.5rem;margin-top:.25rem}
.dk-bar span{font-family:'IBM Plex Mono',monospace;font-size:.64rem;color:var(--mute);width:3.4rem;
             text-transform:uppercase;letter-spacing:.08em}
.dk-bar div{height:6px;background:var(--accent)} .dk-bar em{font-family:'IBM Plex Mono',monospace;
             font-style:normal;font-size:.68rem;color:var(--ink)}
.dk-foot{font-family:'IBM Plex Mono',monospace;font-size:.68rem;color:var(--mute);border-top:1px solid var(--rule);
         margin-top:2rem;padding-top:.7rem}

.stButton>button,.stDownloadButton>button{border-radius:2px;font-family:'IBM Plex Mono',monospace;
        font-size:.74rem;letter-spacing:.04em;text-align:left}
.stTabs [data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace;text-transform:uppercase;
        letter-spacing:.1em;font-size:.76rem}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
GREETING = ("This is the CodeAlpha internship desk. Ask about the deadline, the four tasks, GitHub and "
            "LinkedIn steps, the submission form, the certificate or the LOR. Pick a question from the "
            "index on the left, or type your own.")


@st.cache_resource(show_spinner="Indexing the knowledge base")
def get_engine() -> FAQEngine:
    return FAQEngine()


engine = get_engine()
ss = st.session_state
ss.setdefault("messages", [{"role": "assistant", "content": GREETING, "kind": "greeting", "suggestions": []}])
ss.setdefault("pending", None)
ss.setdefault("last", None)
ss.setdefault("threshold", float(engine.threshold))
ss.setdefault("log", [])        # (kind, confidence) per question
ss.setdefault("eval", None)


def queue(q: str) -> None:
    ss.pending = q


def reset_chat() -> None:
    ss.messages = [{"role": "assistant", "content": GREETING, "kind": "greeting", "suggestions": []}]
    ss.last, ss.log = None, []


# ---------------- handle a new question BEFORE drawing, so it shows up this run ---------------- #
typed = st.chat_input("Ask about the CodeAlpha internship")
prompt = typed or ss.pending
ss.pending = None
if prompt:
    resp = engine.ask(prompt, k=3, threshold=ss.threshold)
    sugg = [] if resp.kind == "answer" else [
        {"id": m.faq_id, "question": m.question} for m in resp.analysis.matches
        if m.category != "Small talk"][:3]
    ss.messages.append({"role": "user", "content": prompt})
    ss.messages.append({"role": "assistant", "content": resp.answer, "kind": resp.kind, "suggestions": sugg})
    ss.last = {"analysis": resp.analysis, "kind": resp.kind, "threshold": ss.threshold}
    ss.log.append((resp.kind, resp.confidence))

# ----------------------------------- masthead ---------------------------------------------- #
asked = len(ss.log)
answered = sum(1 for k, _ in ss.log if k == "answer")
mean_conf = (sum(c for _, c in ss.log) / asked) if asked else 0
st.markdown(
    f'<div class="dk-mast"><div><div class="dk-word">Dock<i>e</i>t</div>'
    f'<div class="dk-tag">FAQ desk · {len(engine.faqs)} entries · CodeAlpha AI Internship · Task 2</div></div>'
    f'<div class="dk-stats"><div class="dk-stat"><b>{asked}</b><span>Questions</span></div>'
    f'<div class="dk-stat"><b>{answered}/{asked}</b><span>Answered</span></div>'
    f'<div class="dk-stat"><b>{mean_conf:.0%}</b><span>Mean confidence</span></div></div></div>',
    unsafe_allow_html=True)

tab_ask, tab_eval = st.tabs(["Ask", "Evaluation"])

# ------------------------------------------ ask -------------------------------------------- #
with tab_ask:
    col_kb, col_chat, col_insp = st.columns([1.05, 2.2, 1.4], gap="large")

    with col_kb:
        st.markdown('<div class="dk-label">Index</div>', unsafe_allow_html=True)
        for cat in engine.categories():
            if cat == "Small talk":
                continue
            with st.expander(cat, expanded=(cat == "Deadline")):
                for f in engine.questions_in(cat):
                    st.button(f["question"], key=f"kb_{f['id']}", on_click=queue, args=(f["question"],))
        st.button("Clear conversation", on_click=reset_chat, key="reset")

    with col_chat:
        st.markdown('<div class="dk-label">Conversation</div>', unsafe_allow_html=True)
        last_idx = len(ss.messages) - 1
        for i, m in enumerate(ss.messages):
            avatar = ":material/support_agent:" if m["role"] == "assistant" else ":material/person:"
            with st.chat_message(m["role"], avatar=avatar):
                st.markdown(m["content"])
                if m["role"] == "assistant" and i == last_idx and m.get("suggestions"):
                    for s in m["suggestions"]:
                        st.button(s["question"], key=f"sg_{i}_{s['id']}", on_click=queue, args=(s["question"],))
        transcript = "\n\n".join(f"**{m['role'].title()}:** {m['content']}" for m in ss.messages)
        st.download_button("Download transcript (.md)", transcript, "docket_transcript.md", "text/markdown")

    with col_insp:
        st.markdown('<div class="dk-label">Inspector</div>', unsafe_allow_html=True)
        last = ss.last
        if not last:
            st.markdown('<div class="dk-mono">Ask a question and this panel shows how it was understood: '
                        'the cleaned tokens, the cosine-similarity scores of the closest entries, and why '
                        'the desk answered, asked for clarification, or declined.</div>', unsafe_allow_html=True)
        else:
            a, thr = last["analysis"], last["threshold"]
            top = a.matches[0].score if a.matches else 0.0
            chips = "".join(f'<span class="dk-chip">{html.escape(t)}</span>' for t in a.tokens) or "-"
            st.markdown(f'<div class="dk-mono">1 · Cleaned tokens (stop-words removed, stemmed)</div>{chips}',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="dk-mono" style="margin-top:.5rem">2 · Decision</div>'
                f'<span class="dk-pill {last["kind"]}">{last["kind"]}</span> '
                f'<span class="dk-mono">score {top:.2f} · threshold {thr:.2f} · vocabulary coverage {a.coverage:.0%}</span>'
                f'<div class="dk-gauge"><div class="fill" style="width:{min(top, 1) * 100:.0f}%"></div>'
                f'<div class="tick" style="left:{min(thr, 1) * 100:.0f}%"></div></div>'
                f'<div class="dk-mono">red tick = answer threshold</div>', unsafe_allow_html=True)
            st.markdown('<div class="dk-mono" style="margin:.8rem 0 .2rem">3 · Closest entries (cosine similarity)</div>',
                        unsafe_allow_html=True)
            for rank, m in enumerate(a.matches, 1):
                def bar(label, v):
                    return (f'<div class="dk-bar"><span>{label}</span><div style="width:{max(v, 0.01) * 120:.0f}px"></div>'
                            f'<em>{v:.2f}</em></div>')
                st.markdown(
                    f'<div class="dk-match"><div class="q">{rank}. {html.escape(m.question)}</div>'
                    f'<div class="dk-mono">matched phrasing: “{html.escape(m.phrase)}”</div>'
                    f'{bar("word", m.word)}{bar("char", m.char)}{bar("final", m.score)}</div>',
                    unsafe_allow_html=True)
        st.slider("Answer threshold", 0.20, 0.80, key="threshold", step=0.01,
                  help="Below this score the desk offers suggestions instead of answering.")

# --------------------------------------- evaluation ---------------------------------------- #
with tab_eval:
    st.markdown('<div class="dk-label">How accurate is it?</div>', unsafe_allow_html=True)
    st.write("Two query sets are bundled. The **tuning set** was used while choosing the threshold. The "
             "**held-out set** was written afterwards and never used for tuning, so it is the honest number.")
    if st.button("Run evaluation", type="primary", key="run_eval"):
        out = {}
        for name in ("eval_queries", "holdout_queries"):
            path = ROOT / "data" / f"{name}.json"
            if path.exists():
                out[name] = evaluate(engine, json.loads(path.read_text(encoding="utf-8")))
        ss.eval = out
    if ss.eval:
        labels = {"eval_queries": "Tuning set", "holdout_queries": "Held-out set"}
        rows = [{"Set": labels[n], "Questions": r["n_in_scope"], "Top-1": f"{r['top1']:.0%}",
                 "Top-3": f"{r['top3']:.0%}", "Confident & correct": f"{r['confident_correct']:.0%}",
                 "Off-topic declined": f"{r['out_of_scope_rejected']:.0%}"} for n, r in ss.eval.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        for n, r in ss.eval.items():
            if r["misses"] or r["leaks"]:
                with st.expander(f"{labels[n]}: what it got wrong"):
                    if r["misses"]:
                        st.dataframe(pd.DataFrame(r["misses"]), hide_index=True)
                    if r["leaks"]:
                        st.dataframe(pd.DataFrame(r["leaks"]), hide_index=True)
    st.markdown(f'<div class="dk-mono" style="margin-top:1rem">Method: {STEMMER} · TF-IDF over words (1-2 grams) '
                f'and characters (3-5 grams) · cosine similarity · vocabulary-coverage penalty for off-topic input.</div>',
                unsafe_allow_html=True)

st.markdown('<div class="dk-foot">Retrieval-based: every answer is a curated entry, nothing is generated. '
            'Typos are tolerated through character n-grams.</div>', unsafe_allow_html=True)
