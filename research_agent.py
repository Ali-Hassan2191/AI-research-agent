"""
research_agent.py
-----------------
All the CrewAI logic lives here: the search tool, the agent, the task and the crew.
app.py (the Streamlit UI) just calls run_research().
"""
import os

# Turn off CrewAI telemetry (must be set BEFORE importing crewai)
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from ddgs import DDGS

MODEL_NAME = "groq/openai/gpt-oss-120b"  # "groq/" = provider, rest = Groq model id


# ---------- 1. TOOL: free DuckDuckGo search ----------
@tool("Web Search")
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo. Input: a short search query.
    Returns titles, links and snippets of the top results."""
    try:
        results = DDGS().text(query, max_results=6)
    except Exception as e:
        return f"Search failed: {e}. Try a different, shorter query."

    if not results:
        return "No results found. Try a different query."

    lines = []
    for r in results:
        lines.append(f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}\n")
    return "\n".join(lines)


# ---------- 2. MAIN FUNCTION ----------
def run_research(topic: str, api_key: str) -> str:
    """Runs the research agent on `topic` and returns the report as Markdown text."""
    os.environ["GROQ_API_KEY"] = api_key

    llm = LLM(
        model=MODEL_NAME,
        temperature=0.3,    # low = more factual, less creative
        max_tokens=4096,    # keeps us inside Groq's free-tier limits
    )

    # ---- The single agent ----
    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Research the topic '{topic}' thoroughly and write an accurate, well-structured report.",
        backstory=(
            "You are an experienced analyst. You always search the web for up-to-date "
            "information, compare several sources, and never invent facts or links."
        ),
        tools=[web_search],
        llm=llm,
        allow_delegation=False,  # single-agent app
        max_iter=8,              # max reasoning/search steps (prevents endless loops)
        verbose=False,
    )

    # ---- The task ----
    task = Task(
        description=(
            f"Research this topic: {topic}\n\n"
            "Use the Web Search tool 3-5 times with different queries to gather facts. "
            "Then write the report using only information you actually found."
        ),
        expected_output=(
            "A Markdown report with these sections:\n"
            "# Title\n"
            "## Summary (3-4 sentences)\n"
            "## Key Findings (bullet points)\n"
            "## Detailed Analysis\n"
            "## Conclusion\n"
            "## Sources (list of URLs actually used)"
        ),
        agent=researcher,
    )

    # ---- The crew (one agent, one task) ----
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return result.raw  # the final report text
