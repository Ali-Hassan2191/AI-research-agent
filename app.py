"""
app.py  -  Streamlit user interface
Run locally with:  streamlit run app.py
"""
import os
import streamlit as st
from research_agent import run_research

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")

st.title("🔎 AI Research Agent")
st.caption("Powered by CrewAI + Groq (gpt-oss-120b) + DuckDuckGo search")


# ---- Get the Groq API key: secrets first, then env var, then sidebar box ----
def get_api_key() -> str:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass  # no secrets file locally - that's fine
    return os.getenv("GROQ_API_KEY", "")


api_key = get_api_key()

with st.sidebar:
    st.header("Settings")
    if not api_key:
        api_key = st.text_input("Groq API key", type="password",
                                help="Get a free key at console.groq.com")
    else:
        st.success("API key loaded ✅")
    st.markdown("---")
    st.markdown("**How it works**\n1. Enter a topic\n2. The agent searches the web\n3. It writes a report")

topic = st.text_input("Research topic", placeholder="e.g. Latest trends in solar energy")

if st.button("Generate report", type="primary"):
    if not api_key:
        st.error("Please add your Groq API key in the sidebar.")
    elif not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        try:
            with st.spinner("Agent is researching... this can take 30-90 seconds"):
                report = run_research(topic.strip(), api_key)
            st.session_state["report"] = report
            st.session_state["topic"] = topic.strip()
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# Show the report (kept in session_state so it survives reruns, e.g. download click)
if "report" in st.session_state:
    st.markdown("---")
    st.markdown(st.session_state["report"])
    st.download_button(
        "⬇️ Download report (.md)",
        data=st.session_state["report"],
        file_name="research_report.md",
        mime="text/markdown",
    )
