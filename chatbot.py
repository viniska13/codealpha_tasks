"""
CodeAlpha AI Internship - Task 2: Gemini-Style FAQ Chatbot
-----------------------------------------------------------
Upgrades over the first version:
  * Smarter matching (typo tolerant, handles several topics in one question)
  * Works in both light and dark mode
  * Typing effect on bot replies
  * More quick-prompt buttons + clear-chat button + chat counter
  * Only uses streamlit + Python standard library, so requirements.txt
    does NOT need to change and Task 1 / Task 3 are unaffected.
"""

import difflib
import re
import time

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Gemini-Style AI Assistant",
    page_icon="✨",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS (theme-neutral: no hard-coded white background, so it looks
# right in both light and dark mode)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
.stChatMessage {
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 0.8rem;
    border: 1px solid rgba(128, 128, 128, 0.25);
}
.stChatInputContainer, [data-testid="stChatInput"] {
    border-radius: 24px;
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
# Knowledge base
# Each entry: a topic name, trigger keywords/phrases, and the answer.
# ---------------------------------------------------------------------------
FAQ_ENTRIES = [
    {
        "topic": "deadline",
        "keywords": ["deadline", "due", "last date", "submit by", "cutoff", "cut off", "november 10", "nov 10"],
        "answer": "📅 The final deadline to submit all your tasks via the official Google Form is **November 10, 2026**.",
    },
    {
        "topic": "tasks",
        "keywords": ["task", "tasks", "how many", "requirements", "requirement", "complete"],
        "answer": "📋 You are required to complete any **2 or 3 tasks** within your chosen domain (Artificial Intelligence) to be eligible for your certificate and LOR.",
    },
    {
        "topic": "certificate",
        "keywords": ["certificate", "certification", "lor", "letter of recommendation", "recommendation"],
        "answer": "🎓 Certificates and Letters of Recommendation (LOR) will be distributed starting **November 11, 2026**, upon verifying your Google Form submission.",
    },
    {
        "topic": "submission",
        "keywords": ["submission", "submit", "google form", "form", "github link", "upload"],
        "answer": "📤 You must submit your GitHub repository link containing your completed code inside the official CodeAlpha Google Form.",
    },
    {
        "topic": "linkedin",
        "keywords": ["linkedin", "video", "post", "explanation video"],
        "answer": "🎥 Sharing a video explanation on LinkedIn is completely **optional**! It gives extra visibility, but it is not mandatory for your certificate.",
    },
    {
        "topic": "domain",
        "keywords": ["domain", "track", "internship track", "field"],
        "answer": "🧠 You are registered under the **Artificial Intelligence** domain for this session.",
    },
    {
        "topic": "projects",
        "keywords": ["project", "projects", "portfolio", "about", "apps", "applications", "my work"],
        "answer": (
            "🚀 Your portfolio has three deployed apps:\n\n"
            "- **Task 1:** Polyglot AI Translator (multi-language translation)\n"
            "- **Task 2:** Gemini-Style FAQ Chatbot (this app!)\n"
            "- **Task 3:** Real-Time Object Detection Suite (YOLOv8 + OpenCV)"
        ),
    },
    {
        "topic": "techstack",
        "keywords": ["tech", "stack", "technology", "technologies", "tools", "libraries", "python", "streamlit", "yolo"],
        "answer": (
            "🛠️ **Tech stack used across the portfolio:**\n\n"
            "- Python 3.11 and Streamlit Community Cloud\n"
            "- Ultralytics YOLOv8, OpenCV (headless) and Pillow for computer vision\n"
            "- `translators` and `deep-translator` for translation\n"
            "- Git and GitHub for version control"
        ),
    },
    {
        "topic": "greeting",
        "keywords": ["hello", "hi", "hey", "hii", "hola", "good morning", "good evening"],
        "answer": "👋 Hello! Welcome back. How can I assist you with your CodeAlpha tasks today?",
    },
    {
        "topic": "help",
        "keywords": ["help", "assist", "support", "what can you do", "options", "menu"],
        "answer": (
            "💡 I can answer questions about:\n\n"
            "- 📅 Deadlines\n"
            "- 📋 Task requirements\n"
            "- 📤 Submission steps\n"
            "- 🎓 Certificate and LOR dates\n"
            "- 🎥 LinkedIn video\n"
            "- 🛠️ Tech stack and your projects"
        ),
    },
]

FALLBACK_MESSAGE = (
    "🤔 I'm not entirely sure about that one. Try asking me about:\n\n"
    "- **deadlines** 📅\n"
    "- **task requirements** 📋\n"
    "- **certificate dates** 🎓\n"
    "- **submission steps** 📤\n\n"
    "Or type **help** to see everything I can do!"
)

WELCOME_MESSAGE = (
    "Hello! I'm your Gemini-styled AI assistant. "
    "Ask me anything about your CodeAlpha tasks or deadlines!"
)

QUICK_PROMPTS = [
    ("📅 What is the deadline?", "What is the deadline?"),
    ("📋 How many tasks needed?", "How many tasks do I need to complete?"),
    ("🎓 When is my certificate?", "When will I get my certificate?"),
    ("📤 How do I submit?", "How do I submit my tasks?"),
    ("🎥 Is the LinkedIn video required?", "Is the LinkedIn video mandatory?"),
    ("🛠️ What tech stack is used?", "What tech stack is used?"),
]


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------
def _normalize(text: str) -> list:
    """Lowercase, strip punctuation, return a list of word tokens."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def _score_entry(entry: dict, tokens: list, padded: str) -> int:
    """Score how well one FAQ entry matches the user's query."""
    score = 0
    for keyword in entry["keywords"]:
        if " " in keyword:
            # Multi-word phrase: must appear as whole words.
            if f" {keyword} " in padded:
                score = max(score, 3)
        elif keyword in tokens:
            # Exact whole-word match.
            score = max(score, 3)
        elif len(keyword) >= 4:
            # Typo tolerance (e.g. "deadlne" -> "deadline").
            for token in tokens:
                if len(token) >= 4 and difflib.SequenceMatcher(None, keyword, token).ratio() >= 0.8:
                    score = max(score, 2)
                    break
    return score


def get_bot_response(user_query: str) -> str:
    """Return the best FAQ answer (or two answers if the question covers two topics)."""
    tokens = _normalize(user_query)
    if not tokens:
        return FALLBACK_MESSAGE

    padded = " " + " ".join(tokens) + " "
    scored = [(_score_entry(entry, tokens, padded), entry) for entry in FAQ_ENTRIES]
    scored = [item for item in scored if item[0] >= 2]
    scored.sort(key=lambda item: item[0], reverse=True)

    if not scored:
        return FALLBACK_MESSAGE

    best_score, best_entry = scored[0]
    answers = [best_entry["answer"]]

    # If the question clearly covers a second topic, answer that too.
    if len(scored) > 1 and scored[1][0] >= 3 and best_score >= 3:
        answers.append(scored[1][1]["answer"])

    return "\n\n".join(answers)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def reset_chat() -> None:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.session_state.pending_prompt = None


def queue_prompt(text: str) -> None:
    """Callback for sidebar quick-prompt buttons."""
    st.session_state.pending_prompt = text


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ---------------------------------------------------------------------------
# UI: Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ✨ Gemini Clone")
    st.write("**CodeAlpha Task 2: FAQ Chatbot**")
    st.markdown("---")

    st.button("➕ New Chat", use_container_width=True, on_click=reset_chat)

    st.markdown("---")
    st.markdown("**Quick Prompt Ideas:**")
    for index, (label, prompt_text) in enumerate(QUICK_PROMPTS):
        st.button(
            label,
            key=f"quick_{index}",
            use_container_width=True,
            on_click=queue_prompt,
            args=(prompt_text,),
        )

    st.markdown("---")
    user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
    st.caption(f"💬 Questions asked this chat: {user_msgs}")

# ---------------------------------------------------------------------------
# UI: Main header
# ---------------------------------------------------------------------------
st.markdown('<p class="gemini-header">Hello, Viniska</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="gemini-sub">What would you like to know about your internship?</p>',
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# UI: Chat history
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    avatar = "✨" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Handle input (typed prompt OR sidebar quick prompt)
# ---------------------------------------------------------------------------
prompt = st.chat_input("Enter a prompt here...")

if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    # Show user message
    st.chat_message("user", avatar="👤").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    response = get_bot_response(prompt)

    # Show assistant response with a typing effect
    with st.chat_message("assistant", avatar="✨"):
        placeholder = st.empty()
        shown = ""
        for word in response.split(" "):
            shown += word + " "
            placeholder.markdown(shown + "▌")
            time.sleep(0.02)
        placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
