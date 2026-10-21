"""Small Streamlit helpers shared by all three apps."""
import streamlit as st


def page_config(**kwargs) -> None:
    """Call st.set_page_config, ignoring the error raised when main.py (the hub)
    has already configured the page."""
    try:
        st.set_page_config(**kwargs)
    except Exception:
        pass


def inject_css(css: str) -> None:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def header(title: str, subtitle: str) -> None:
    st.markdown(f"<h1 style='margin-bottom:0'>{title}</h1>", unsafe_allow_html=True)
    st.caption(subtitle)
