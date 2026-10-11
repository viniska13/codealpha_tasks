import streamlit as st
import translators as ts

# Page config
st.set_page_config(page_title="Polyglot AI - Translator", page_icon="✨", layout="centered")

# Custom CSS for colorful, modern styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 100%, #667eea 100%);
        color: #ffffff;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #e6fffa;
        border-left: 5px solid #319795;
        color: #234e52;
    }
    </style>
""", unsafe_allow_html=True)

# Title Header
st.markdown("<h1 style='text-align: center; color: #4a5568;'>✨ Polyglot AI Translator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #718096;'>CodeAlpha AI Internship | Professional Language Translation Suite</p>", unsafe_allow_html=True)
st.markdown("---")

# Language choices (using standard codes for the translators library)
LANGUAGES = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Hindi': 'hi',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Kannada': 'kn',
    'Chinese': 'zh',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Arabic': 'ar',
    'Russian': 'ru',
    'Italian': 'it',
    'Portuguese': 'pt'
}

sorted_languages = sorted(LANGUAGES.keys())

# Selection layout
col1, col2 = st.columns(2)

with col1:
    source_lang = st.selectbox("📥 Source Language", ["Auto Detect"] + sorted_languages)

with col2:
    target_lang = st.selectbox("📤 Target Language", sorted_languages, index=sorted_languages.index("English"))

# Text input area
text_to_translate = st.text_area("✍️ Enter text to translate:", placeholder="Type or paste your content here...", height=150)

# Translate Action
if st.button("🚀 Translate Now"):
    if not text_to_translate.strip():
        st.warning("⚠️ Please enter some text to translate.")
    else:
        with st.spinner("Translating with AI... 🤖"):
            try:
                dest_code = LANGUAGES[target_lang]
                src_code = 'auto' if source_lang == "Auto Detect" else LANGUAGES[source_lang]
                
                # Using the robust translators library backend (Google engine)
                translated_text = ts.translate_text(
                    text_to_translate, 
                    translator='google', 
                    from_language=src_code, 
                    to_language=dest_code
                )
                
                st.markdown("### 🎯 Translated Output:")
                st.markdown(f'<div class="success-box"><b>{translated_text}</b></div>', unsafe_allow_html=True)
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Translation Error: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.8rem; color: #a0aec0;'>Built with Python, Streamlit & Translators Engine</p>", unsafe_allow_html=True)