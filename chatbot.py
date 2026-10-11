import streamlit as st

# Page config
st.set_page_config(page_title="Gemini-Style AI Assistant", page_icon="✨", layout="centered")

# Custom CSS for Gemini-like styling (minimalist dark/light accents, rounded chat bubbles)
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border: 1px solid #f0f0f0;
    }
    .stChatInputContainer {
        border-radius: 24px;
        border: 1px solid #dfe1e5;
        box-shadow: 0 1px 6px rgba(32,33,36,.28);
    }
    .gemini-header {
        font-family: 'Google Sans', sans-serif;
        background: linear-gradient(90deg, #4285F4, #9B72CB, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        font-size: 2.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar (Gemini Style Sidebar Controls)
with st.sidebar:
    st.markdown("### ✨ Gemini Clone")
    st.write("**CodeAlpha Task 2: FAQ Chatbot**")
    st.markdown("---")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Gemini-styled AI assistant. How can I help you today?"}
        ]
        st.rerun()
    st.markdown("---")
    st.markdown("**Quick Prompt Ideas:**")
    if st.button("📅 What is the deadline?"):
        st.session_state.user_quick_prompt = "What is the deadline?"
    if st.button("📋 How many tasks needed?"):
        st.session_state.user_quick_prompt = "How many tasks do I need to complete?"

# Main Title Header
st.markdown('<p class="gemini-header">Hello, Viniska</p>', unsafe_allow_html=True)
st.markdown("<p style='color: #5f6368; font-size: 1.1rem; margin-top: -10px;'>What would you like to know about your internship?</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Predefined Knowledge Base (FAQs)
faq_database = {
    "deadline": "The final deadline to submit all your tasks via the official Google Form is **November 10, 2026**.",
    "tasks": "You are required to complete any **2 or 3 tasks** within your chosen domain (Artificial Intelligence) to be eligible for your certificate and LOR.",
    "certificate": "Certificates and Letters of Recommendation (LOR) will be distributed starting **November 11, 2026**, upon verifying your Google Form submission.",
    "submission": "You must submit your GitHub repository link containing your completed code inside the official CodeAlpha Google Form.",
    "linkedin": "Sharing a video explanation on LinkedIn is completely **optional**! It gives extra visibility, but it is not mandatory for your certificate.",
    "domain": "You are registered under the **Artificial Intelligence** domain for this session.",
    "hello": "Hello! Welcome back. How can I assist you with your CodeAlpha tasks today?",
    "help": "I can answer questions about deadlines, task requirements, submission steps, and certificate release dates!"
}

# Response logic
def get_bot_response(user_query):
    query = user_query.lower()
    for keyword, answer in faq_database.items():
        if keyword in query:
            return answer
    return "I'm not entirely sure about that specific query. You can ask me about deadlines, task requirements, certificate dates, or submissions!"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your AI assistant. Ask me anything about your CodeAlpha tasks or deadlines!"}
    ]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="✨" if message["role"]=="assistant" else "👤"):
        st.markdown(message["content"])

# Handle sidebar quick prompt clicks if triggered
prompt = st.chat_input("Enter a prompt here...")
if "user_quick_prompt" in st.session_state and st.session_state.user_quick_prompt:
    prompt = st.session_state.user_quick_prompt
    st.session_state.user_quick_prompt = None

# Process user input
if prompt:
    # Display user message
    st.chat_message("user", avatar="👤").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    response = get_bot_response(prompt)
    
    # Display assistant response
    with st.chat_message("assistant", avatar="✨"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})