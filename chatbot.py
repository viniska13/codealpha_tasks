import os

import requests
import streamlit as st

st.set_page_config(page_title="AI Assistant", page_icon="✨", layout="centered")

st.markdown(
    """
<style>
.stChatMessage {
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 0.8rem;
    border: 1px solid rgba(128, 128, 128, 0.25);
}
.gemini-header {
    font-family: 'Google Sans', sans-serif;
    background: linear-gradient(90deg, #4285F4, #9B72CB, #D96570);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
    font-size: 2.2rem;
    margin-bottom: 0;
}
.gemini-sub {
    opacity: 0.7;
    font-size: 1.1rem;
    margin-top: -8px;
}
</style>
""",
    unsafe_allow_html=True,
)

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
    st.button("➕ New Chat", use_container_width=True, on_click=reset_chat)
    st.markdown("---")
    for i, text in enumerate(QUICK_PROMPTS):
        st.button(text, key=f"quick_{i}", use_container_width=True, on_click=queue_prompt, args=(text,))

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="gemini-header">Hello, Viniska</p>', unsafe_allow_html=True)
st.markdown('<p class="gemini-sub">How can I help you today?</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

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
