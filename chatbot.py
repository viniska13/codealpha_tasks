import os

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Theme: dark canvas with neon pink (#ff9fbd) and neon aqua (#9fdfe5)
# All styling lives in this file, so Task 1 and Task 3 are unaffected.
# ---------------------------------------------------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0a0a12;
    --panel: #12121c;
    --text: #ececf4;
    --muted: #9a9ab0;
    --pink: #ff9fbd;
    --aqua: #9fdfe5;
    --pink-glow: 0 0 8px rgba(255, 159, 189, 0.85), 0 0 24px rgba(255, 159, 189, 0.45);
    --aqua-glow: 0 0 8px rgba(159, 223, 229, 0.85), 0 0 24px rgba(159, 223, 229, 0.45);
}

:root, .stApp { color-scheme: dark; }

.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: var(--bg) !important; }
[data-testid="stHeader"] * { color: var(--muted) !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #0a0a12 100%) !important;
    border-right: 1px solid rgba(159, 223, 229, 0.25);
}

/* Typography */
.stApp, .stApp p, .stApp li, .stApp label, .stApp textarea, .stApp input, .stApp button,
.stApp [data-testid="stMarkdownContainer"] {
    font-family: 'Poppins', sans-serif;
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"] { color: var(--text); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: 'Poppins', sans-serif; font-weight: 600; color: var(--aqua) !important; text-shadow: var(--aqua-glow);
}
.stApp a { color: var(--aqua) !important; }
.stApp strong { color: #ffffff; font-weight: 600; }
code, pre, kbd { font-family: 'JetBrains Mono', monospace !important; }
p code, li code {
    color: var(--pink) !important; background: rgba(255, 159, 189, 0.1) !important;
    border-radius: 6px; padding: 0.1rem 0.4rem;
}
pre, [data-testid="stCode"] {
    background: #0c0c15 !important; border: 1px solid rgba(159, 223, 229, 0.35) !important; border-radius: 12px !important;
}
pre code { color: var(--aqua) !important; }
th { color: var(--aqua) !important; }
td, th { border-color: rgba(159, 223, 229, 0.2) !important; }
hr { border-color: rgba(159, 223, 229, 0.2) !important; }

/* Header */
.nb-hero { padding: 0.5rem 0 0.2rem 0; }
.nb-title {
    font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 2.6rem; line-height: 1.15; margin: 0;
    background: linear-gradient(90deg, var(--pink), var(--aqua));
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 10px rgba(255, 159, 189, 0.45)) drop-shadow(0 0 18px rgba(159, 223, 229, 0.3));
}
.nb-sub { color: var(--muted) !important; font-size: 1.05rem; margin: 0.2rem 0 1rem 0; }
.nb-line {
    height: 2px; border-radius: 2px; margin-bottom: 1.2rem;
    background: linear-gradient(90deg, var(--pink), var(--aqua), transparent);
    box-shadow: 0 0 12px rgba(159, 223, 229, 0.5);
}

/* Sidebar brand */
.nb-brand { display: flex; align-items: center; gap: 0.8rem; margin: 0.2rem 0 1rem 0; }
.nb-orb {
    width: 42px; height: 42px; border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, #ffffff 0%, var(--pink) 35%, var(--aqua) 100%);
    box-shadow: 0 0 14px rgba(255, 159, 189, 0.8), 0 0 28px rgba(159, 223, 229, 0.55);
    animation: nb-pulse 3s ease-in-out infinite;
}
.nb-brand-name { margin: 0; font-weight: 600; font-size: 1.05rem; color: #ffffff !important; }
.nb-status { margin: 0; font-size: 0.8rem; color: var(--aqua) !important; }
.nb-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px;
    background: var(--aqua); box-shadow: var(--aqua-glow);
}
@keyframes nb-pulse {
    0%, 100% { box-shadow: 0 0 12px rgba(255, 159, 189, 0.7), 0 0 24px rgba(159, 223, 229, 0.4); }
    50% { box-shadow: 0 0 20px rgba(255, 159, 189, 1), 0 0 38px rgba(159, 223, 229, 0.75); }
}

/* Chat bubbles: assistant = aqua, user = pink */
[data-testid="stChatMessage"] {
    background: rgba(159, 223, 229, 0.05);
    border: 1px solid rgba(159, 223, 229, 0.5);
    border-radius: 18px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 0 14px rgba(159, 223, 229, 0.16);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(255, 159, 189, 0.06);
    border-color: rgba(255, 159, 189, 0.55);
    box-shadow: 0 0 14px rgba(255, 159, 189, 0.18);
}
[data-testid^="stChatMessageAvatar"] { background: var(--panel) !important; border: 1px solid rgba(159, 223, 229, 0.4); }

/* Chat input */
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: var(--bg) !important; }
[data-testid="stChatInput"] {
    background: var(--panel) !important;
    border: 1px solid var(--aqua) !important;
    border-radius: 28px !important;
    box-shadow: 0 0 10px rgba(159, 223, 229, 0.35);
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--pink) !important;
    box-shadow: 0 0 14px rgba(255, 159, 189, 0.65), 0 0 30px rgba(255, 159, 189, 0.25);
}
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea { background: transparent !important; }
[data-testid="stChatInput"] textarea { color: #ffffff !important; caret-color: var(--pink); }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; }
[data-testid="stChatInputSubmitButton"] { color: var(--pink) !important; background: transparent !important; }
[data-testid="stChatInputSubmitButton"]:hover { text-shadow: var(--pink-glow); }

/* Buttons */
.stApp .stButton button, .stApp [data-testid^="stBaseButton"] {
    background: var(--panel) !important;
    color: var(--text) !important;
    border: 1px solid rgba(159, 223, 229, 0.65) !important;
    border-radius: 12px !important;
    font-family: 'Poppins', sans-serif; font-weight: 500;
    transition: box-shadow 0.2s ease, border-color 0.2s ease, color 0.2s ease;
}
.stApp .stButton button:hover, .stApp [data-testid^="stBaseButton"]:hover {
    border-color: var(--pink) !important; color: var(--pink) !important;
    box-shadow: 0 0 12px rgba(255, 159, 189, 0.6), 0 0 26px rgba(255, 159, 189, 0.25);
}
.stApp .stButton button *, .stApp [data-testid^="stBaseButton"] * { color: inherit !important; }
.st-key-new_chat button {
    border-color: var(--pink) !important;
    box-shadow: 0 0 10px rgba(255, 159, 189, 0.45);
    font-weight: 600;
}
.st-key-new_chat button:hover {
    border-color: var(--aqua) !important; color: var(--aqua) !important;
    box-shadow: 0 0 12px rgba(159, 223, 229, 0.65), 0 0 26px rgba(159, 223, 229, 0.3);
}

/* Misc */
[data-testid="stCaptionContainer"] { color: var(--muted) !important; }
.stSpinner, .stSpinner * { color: var(--aqua) !important; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, var(--pink), var(--aqua)); border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }
</style>
"""

st.set_page_config(page_title="AI Assistant", page_icon="✨", layout="centered")

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Instructions given to the AI (background facts about the internship are
# included so it can still answer those questions accurately)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful, friendly AI assistant. Answer any question the user asks, on any topic, clearly and accurately. Use Markdown when it helps (lists, code blocks, bold). Keep answers focused and to the point.

If the user asks about the CodeAlpha AI internship, use these facts:
- Final deadline to submit all tasks via the official Google Form: November 10, 2026.
- Required: any 2 or 3 tasks in the chosen domain (Artificial Intelligence) for the certificate and LOR.
- Certificates and Letters of Recommendation are distributed from November 11, 2026, after the Google Form submission is verified.
- Submission: the GitHub repository link with the completed code goes into the official CodeAlpha Google Form.
- A LinkedIn video explanation is optional, not mandatory.
- The portfolio has three apps: Task 1 Polyglot AI Translator, Task 2 this chatbot, Task 3 Real-Time Object Detection (YOLOv8 + OpenCV)."""

QUICK_PROMPTS = [
    "Explain machine learning in simple words",
    "Write a Python function to reverse a string",
    "What is the deadline for my internship tasks?",
    "Give me tips for a good LinkedIn post",
]

GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
CLAUDE_MODEL = "claude-haiku-5-5"
MAX_HISTORY = 20


class ChatError(Exception):
    pass


def _error_detail(response):
    try:
        return str(response.json().get("error", {}).get("message", ""))[:200]
    except Exception:
        return ""


def get_secret(name):
    try:
        value = st.secrets[name]
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(name)


def call_gemini(history, api_key):
    contents = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
        for m in history
    ]
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
    }
    custom = get_secret("GEMINI_MODEL")
    models = [custom] if custom else list(GEMINI_MODELS)
    cached = st.session_state.get("gemini_model")
    if cached in models:
        models.remove(cached)
        models.insert(0, cached)

    last_status = None
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            r = requests.post(
                url,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=body,
                timeout=60,
            )
        except requests.RequestException:
            raise ChatError("Network problem. Please try again.")

        if r.status_code in (404, 429, 500, 503):
            # Model missing, out of free quota, or overloaded: try the next model.
            last_status = r.status_code
            continue
        if r.status_code in (400, 401, 403):
            detail = _error_detail(r)
            raise ChatError(f"The API key or request was rejected. {detail}")
        if r.status_code != 200:
            raise ChatError(f"The AI service returned an error ({r.status_code}). {_error_detail(r)}")

        data = r.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise ChatError("I couldn't answer that one. Please try rephrasing.")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise ChatError("I couldn't answer that one. Please try rephrasing.")
        st.session_state["gemini_model"] = model
        return text.strip()

    if last_status == 429:
        raise ChatError(
            "The free daily/minute limit is used up on all models. "
            "Wait a few minutes (or until tomorrow), or create a new key with a different Google account."
        )
    if last_status in (500, 503):
        raise ChatError("The AI service is busy right now. Please try again in a moment.")
    raise ChatError("No available Gemini model was found. Set GEMINI_MODEL in your secrets.")


def call_claude(history, api_key):
    body = {
        "model": get_secret("CLAUDE_MODEL") or CLAUDE_MODEL,
        "max_tokens": 1500,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": m["role"], "content": m["content"]} for m in history],
    }
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
            timeout=60,
        )
    except requests.RequestException:
        raise ChatError("Network problem. Please try again.")

    if r.status_code in (401, 403):
        raise ChatError("The API key looks invalid. Please check ANTHROPIC_API_KEY in your secrets.")
    if r.status_code == 429:
        raise ChatError("Too many requests right now. Please wait a moment and try again.")
    if r.status_code != 200:
        raise ChatError(f"The AI service returned an error ({r.status_code}). Please try again.")

    blocks = r.json().get("content", [])
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    if not text.strip():
        raise ChatError("I couldn't answer that one. Please try rephrasing.")
    return text.strip()


def get_bot_response(messages):
    history = messages[-MAX_HISTORY:]
    while history and history[0]["role"] != "user":
        history = history[1:]

    gemini_key = get_secret("GEMINI_API_KEY")
    claude_key = get_secret("ANTHROPIC_API_KEY")
    if gemini_key:
        return call_gemini(history, gemini_key)
    if claude_key:
        return call_claude(history, claude_key)
    raise ChatError(
        "The chatbot is not configured yet. Add `GEMINI_API_KEY` in the app's Secrets."
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


def reset_chat():
    st.session_state.messages = []
    st.session_state.pending_prompt = None


def queue_prompt(text):
    st.session_state.pending_prompt = text


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="nb-brand"><div class="nb-orb"></div><div>'
        '<p class="nb-brand-name">AI Assistant</p>'
        '<p class="nb-status"><span class="nb-dot"></span>Online</p></div></div>',
        unsafe_allow_html=True,
    )
    st.button("➕ New Chat", key="new_chat", use_container_width=True, on_click=reset_chat)
    st.markdown("---")
    for i, text in enumerate(QUICK_PROMPTS):
        st.button(text, key=f"quick_{i}", use_container_width=True, on_click=queue_prompt, args=(text,))

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="nb-hero"><p class="nb-title">Hello, Viniska</p>'
    '<p class="nb-sub">How can I help you today?</p></div><div class="nb-line"></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="✨" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
prompt = st.chat_input("Ask me anything...")

if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    st.chat_message("user", avatar="👤").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Thinking..."):
            try:
                response = get_bot_response(st.session_state.messages)
            except ChatError as error:
                response = f"⚠️ {error}"
            except Exception:
                response = "⚠️ Something went wrong. Please try again."
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
