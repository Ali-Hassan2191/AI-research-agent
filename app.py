"""
app.py  -  Streamlit user interface ("Workspace" design)
Run locally with:  streamlit run app.py

The report text is produced by research_agent.py - this file only handles the look.
"""
import os
import re
import time
from urllib.parse import urlparse

import streamlit as st
from research_agent import run_research, MODEL_NAME

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="wide")

# =====================================================================
# 1. STYLE (CSS)
# =====================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700;9..144,800&display=swap');

/* --- Streamlit's top bar: make it transparent and push content below it,
       so the banner never hides behind the white header --- */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"], .block-container {
    padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1100px;
}
footer { visibility: hidden; }

/* --- Top banner --- */
.hero {
    display: flex; align-items: center; gap: 22px;
    background: linear-gradient(120deg, #0E3B40 0%, #1E7A6B 100%);
    border-radius: 22px; padding: 26px 32px; margin-bottom: 1.4rem;
    box-shadow: 0 12px 30px rgba(15, 59, 64, .18);
}
.hero-icon {
    width: 68px; height: 68px; flex-shrink: 0; border-radius: 18px;
    background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28);
    display: flex; align-items: center; justify-content: center;
}
.hero-title {
    font-family: 'Fraunces', Georgia, serif; font-weight: 800;
    font-size: 2.3rem; line-height: 1.15; color: #fff;
}
.hero-sub { color: #D3EEE8; font-size: 1.02rem; margin-top: 4px; line-height: 1.45; }

/* --- Input box + Generate button --- */
[data-testid="stForm"] { padding: 0; border: none; }
[data-baseweb="input"] { border-radius: 10px; border: 1px solid #CBD8D4; background: #fff; }
[data-baseweb="input"] { min-height: 3.1rem; }
[data-baseweb="input"] input { font-size: 1.05rem; padding: .85rem 1rem; }
[data-testid="InputInstructions"] { display: none; }
[data-testid="stForm"] { margin-bottom: 1rem; }
[data-testid="stFormSubmitButton"] button {
    height: 3.1rem; border-radius: 10px; font-weight: 700; font-size: 1.05rem;
}

/* --- Stat cards --- */
.stat { background: #fff; border: 1px solid #DDE5E2; border-radius: 14px; padding: 16px 20px; }
.stat b { display: block; font-size: 1.9rem; line-height: 1.25; color: #0F766E; }
.stat span { font-size: .85rem; color: #7A918C; }

/* --- Report card --- */
.st-key-report_card {
    background: #fff; border: 1px solid #DDE5E2; border-radius: 14px; padding: .5rem 1.2rem 1rem;
}
.st-key-report_card [data-testid="stVerticalBlockBorderWrapper"] { background: transparent; border: none; padding: 0; }
.empty { text-align: center; color: #7A918C; padding: 3rem 1rem; }
.empty b { color: #17302C; font-size: 1.15rem; }

/* --- Sidebar --- */
.side-logo { font-size: 1.35rem; font-weight: 700; color: #0F766E; margin: .2rem 0 1rem; }
.side-label { font-size: .8rem; color: #7A918C; margin: 1.1rem 0 .3rem; }
[data-testid="stSidebar"] { border-right: 1px solid #DDE5E2; }
[data-testid="stSidebar"] button {
    justify-content: flex-start; text-align: left; border: none; box-shadow: none;
    background: transparent; border-radius: 8px; padding: .45rem .75rem;
}
[data-testid="stSidebar"] button > div,
[data-testid="stSidebar"] button [data-testid="stMarkdownContainer"] { width: 100%; justify-content: flex-start; text-align: left; }
[data-testid="stSidebar"] button p { text-align: left; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
[data-testid="stSidebar"] button:hover { background: #F1F4F3; }
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: #E2F2EF; color: #0F766E; font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================================
# 2. HELPERS
# =====================================================================
def get_api_key() -> str:
    """Looks for the Groq key in Streamlit secrets, then in environment variables."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass  # no secrets file locally - that's fine
    return os.getenv("GROQ_API_KEY", "")


def extract_urls(text: str) -> list:
    """All unique links found in the report (in order)."""
    urls = [u.rstrip(".,;:") for u in re.findall(r"https?://[^\s)\]>\"']+", text)]
    return list(dict.fromkeys(urls))


def safe_markdown(text: str) -> str:
    """Streamlit treats $...$ as a math formula. Escape '$' so prices like $135,000 show normally.
    (Only used for display - the downloaded file keeps the original text.)"""
    return re.sub(r"(?<!\\)\$", r"\\$", text)


def select_report(report_id):
    st.session_state["current"] = report_id


# =====================================================================
# 3. SESSION STATE
# =====================================================================
if "history" not in st.session_state:
    st.session_state["history"] = []      # list of past reports (newest first)
    st.session_state["current"] = None    # id of the report being shown
    st.session_state["next_id"] = 1

api_key = get_api_key()

# =====================================================================
# 4. SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown('<div class="side-logo">Research Agent</div>', unsafe_allow_html=True)

    st.button(
        "＋  New report", key="new_report", width="stretch",
        type="primary" if st.session_state["current"] is None else "secondary",
        on_click=select_report, args=(None,),
    )

    if st.session_state["history"]:
        st.markdown('<div class="side-label">Past reports</div>', unsafe_allow_html=True)
        for item in st.session_state["history"][:10]:
            st.button(
                item["topic"], key=f"hist_{item['id']}", width="stretch",
                type="primary" if st.session_state["current"] == item["id"] else "secondary",
                on_click=select_report, args=(item["id"],),
            )
        st.caption("Past reports are kept only until you refresh the page.")

    st.markdown('<div class="side-label">Settings</div>', unsafe_allow_html=True)
    if api_key:
        st.success("API key loaded ✅")
    else:
        api_key = st.text_input("Groq API key", type="password", help="Get a free key at console.groq.com")
    st.caption(f"Model: {MODEL_NAME.split('/', 1)[1]}")

# =====================================================================
# 5. BANNER
# =====================================================================
st.markdown(
    """
<div class="hero">
  <div class="hero-icon">
    <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8"
         stroke-linecap="round" stroke-linejoin="round">
      <circle cx="10.5" cy="10.5" r="6.5"/><line x1="15.5" y1="15.5" x2="21" y2="21"/>
      <path d="M10.5 7.5v6M7.5 10.5h6" stroke-width="1.4"/>
    </svg>
  </div>
  <div>
    <div class="hero-title">AI Research Agent</div>
    <div class="hero-sub">Enter any topic. The agent searches the web and writes a structured report with its sources.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# =====================================================================
# 6. TOPIC FORM  (pressing Enter also submits)
# =====================================================================
with st.form("topic_form", border=False):
    col_input, col_btn = st.columns([5, 1], vertical_alignment="bottom")
    topic = col_input.text_input(
        "Research topic", placeholder="e.g. Solar energy storage trends in 2026",
        label_visibility="collapsed",
    )
    submitted = col_btn.form_submit_button("Generate", type="primary", width="stretch")

if submitted:
    if not api_key:
        st.error("Please add your Groq API key in the sidebar.")
    elif not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        try:
            search_log, started = [], time.time()
            with st.spinner("Agent is researching... this can take 30-90 seconds"):
                report = run_research(topic.strip(), api_key, search_log)
            cited = extract_urls(report)
            found = list(dict.fromkeys(r["url"] for s in search_log for r in s["results"] if r.get("url")))
            rid = st.session_state["next_id"]
            st.session_state["next_id"] += 1
            st.session_state["history"].insert(0, {
                "id": rid, "topic": topic.strip(), "report": report, "log": search_log,
                "secs": round(time.time() - started), "words": len(report.split()),
                "cited": cited, "found": found,
            })
            st.session_state["current"] = rid
            st.rerun()  # refresh so the sidebar shows the new report
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# =====================================================================
# 7. SHOW THE CURRENT REPORT
# =====================================================================
current = next((h for h in st.session_state["history"] if h["id"] == st.session_state["current"]), None)

if current is None:
    st.markdown(
        '<div class="stat" style="margin-top:1rem"><div class="empty">'
        "<b>Your report will appear here</b><br>"
        "Type a topic above and press Generate (or hit Enter)."
        "</div></div>",
        unsafe_allow_html=True,
    )
else:
    n_sources = len(current["cited"]) or len(current["found"])
    s1, s2, s3 = st.columns(3)
    for col, value, label in (
        (s1, n_sources, "Sources used"),
        (s2, f"{current['secs']}s", "Time taken"),
        (s3, f"{current['words']:,}", "Words"),
    ):
        col.markdown(f'<div class="stat"><b>{value}</b><span>{label}</span></div>', unsafe_allow_html=True)
    st.write("")

    with st.container(key="report_card"):
        _, col_dl = st.columns([4, 1])
        col_dl.download_button(
            "⬇ Download .md", data=current["report"], file_name="research_report.md",
            mime="text/markdown", width="stretch", key=f"dl_{current['id']}",
        )
        tab_report, tab_sources, tab_log = st.tabs(["Report", "Sources", "Search log"])

        with tab_report:
            st.markdown(safe_markdown(current["report"]))

        with tab_sources:
            if current["cited"]:
                st.markdown("**Links cited in the report**")
                for i, url in enumerate(current["cited"], 1):
                    st.markdown(f"{i}. [{urlparse(url).netloc}]({url})  \n&nbsp;&nbsp;&nbsp;&nbsp;<small>{url}</small>",
                                unsafe_allow_html=True)
            if current["found"]:
                with st.expander(f"Links found by the search tool ({len(current['found'])})"):
                    for url in current["found"]:
                        st.markdown(f"- [{url}]({url})")
            if not current["cited"] and not current["found"]:
                st.info("No links were found for this report.")
            if not current["found"]:
                st.warning("The search tool returned no results, so the sources above "
                           "come from the model's memory and may not be accurate. Please verify them.")

        with tab_log:
            if not current["log"]:
                st.warning("The agent did not run any web search for this report.")
            for i, s in enumerate(current["log"], 1):
                title = f"{i}. {s['query']}  ·  {len(s['results'])} results"
                with st.expander(title):
                    if s.get("error"):
                        st.caption(f"Search problem: {s['error']}")
                    for r in s["results"]:
                        st.markdown(f"- [{r.get('title') or r['url']}]({r['url']})")
