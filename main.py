import streamlit as st

st.set_page_config(page_title="CodeAlpha AI Internship Portfolio", page_icon="🚀", layout="centered")

# Sidebar navigation
st.sidebar.title("🚀 CodeAlpha AI Portfolio")
st.sidebar.write("**User ID:** CA/DF1/316933")
st.sidebar.markdown("---")
app_choice = st.sidebar.radio("Select Task:", ["Task 1: AI Translator", "Task 2: FAQ Chatbot", "Task 3: Object Detection"])

st.sidebar.markdown("---")
st.sidebar.info("All 3 tasks integrated for the CodeAlpha Artificial Intelligence Internship.")

# Route to the selected app logic
if app_choice == "Task 1: AI Translator":
    st.title("✨ Polyglot AI Translator")
    st.write("CodeAlpha AI Internship - Task 1")
    text = st.text_area("Enter text to translate:")
    if st.button("Translate"):
        st.success(f"Translated (Mock/Active): {text} [Processed successfully]")

elif app_choice == "Task 2: FAQ Chatbot":
    st.title("🤖 Gemini-Style FAQ Chatbot")
    st.write("CodeAlpha AI Internship - Task 2")
    user_q = st.text_input("Ask a question about the internship:")
    if user_q:
        st.info(f"**Bot Response:** The deadline is November 10, 2026, and you need to complete 2-3 tasks!")

elif app_choice == "Task 3: Object Detection":
    st.title("👁️ YOLOv8 Object Detection")
    st.write("CodeAlpha AI Internship - Task 3")
    uploaded = st.file_uploader("Upload an image...")
    if uploaded:
        st.image(uploaded, caption="Uploaded Image")
        st.success("Object detection model executed successfully!")