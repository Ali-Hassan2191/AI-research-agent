"""
research_agent.py
-----------------
All the CrewAI logic lives here. app.py (the Streamlit UI) just calls run_research().

How it works (3 simple steps):
  1. The LLM decides which web searches to make (it returns a list of search queries).
  2. We run those searches on DuckDuckGo (free) and collect the results.
  3. The CrewAI agent reads the results and writes the report, using only real links.

Why not let the model call the search tool itself? gpt-oss on Groq often makes
malformed tool calls (tool_use_failed errors). Running the search in our own code
is 100% reliable and also guarantees that the sources in the report are real.
"""
import os
import json
import re
import time

# Turn off CrewAI telemetry (must be set BEFORE importing crewai)
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew, LLM, Process
from ddgs import DDGS

MODEL_NAME = "groq/openai/gpt-oss-120b"  # "groq/" = provider, rest = Groq model id


# ---------- 0. FIX: Groq rejects CrewAI's internal "cache_breakpoint" flag ----------
class GroqLLM(LLM):
    """Same as CrewAI's LLM, but removes the 'cache_breakpoint' key from every
    message before it is sent to Groq (Groq returns a 400 error if it is present)."""

    def supports_function_calling(self) -> bool:
        return False  # we never use native tool calls with this model

    def _format_messages_for_provider(self, messages):
        formatted = super()._format_messages_for_provider(messages)
        return [{k: v for k, v in m.items() if k != "cache_breakpoint"} for m in formatted]


# ---------- 1. WEB SEARCH (DuckDuckGo, free) ----------
def search_web(query: str, search_log: list) -> list:
    """Runs one DuckDuckGo search, saves it in search_log and returns the results."""
    results, error = [], None
    for attempt in range(2):  # DuckDuckGo sometimes fails once - try twice
        try:
            results = DDGS().text(query, max_results=6) or []
            error = None
            break
        except Exception as e:
            error = str(e)
            time.sleep(1.5)

    clean = [
        {"title": r.get("title") or "", "url": r.get("href") or "", "snippet": (r.get("body") or "")[:350]}
        for r in results if r.get("href")
    ]
    entry = {"query": query, "results": [{"title": r["title"], "url": r["url"]} for r in clean]}
    if error:
        entry["error"] = error
    search_log.append(entry)
    return clean


def plan_search_queries(llm: LLM, topic: str) -> list:
    """Asks the LLM for 4 good search queries. Falls back to simple queries if anything fails."""
    fallback = [topic, f"{topic} statistics", f"{topic} latest news 2026", f"{topic} analysis"]
    try:
        answer = llm.call(
            "You help with web research. Give 4 different, short web search queries "
            f"to research this topic thoroughly: {topic}\n"
            'Reply with ONLY a JSON list of 4 strings, for example ["query one", "query two", "query three", "query four"]'
        )
        match = re.search(r"\[.*?\]", str(answer), re.S)
        queries = [q.strip() for q in json.loads(match.group(0)) if isinstance(q, str) and q.strip()]
        return queries[:4] or fallback
    except Exception:
        return fallback


# ---------- 2. MAIN FUNCTION ----------
def run_research(topic: str, api_key: str, search_log: list | None = None) -> str:
    """Runs the research agent on `topic` and returns the report as Markdown text.
    If you pass an empty list as `search_log`, it will be filled with the searches made."""
    os.environ["GROQ_API_KEY"] = api_key
    if search_log is None:
        search_log = []

    llm = GroqLLM(
        model=MODEL_NAME,
        temperature=0.3,    # low = more factual, less creative
        max_tokens=4096,    # keeps us inside Groq's free-tier limits
    )

    # ---- Step 1 + 2: decide the searches and run them ----
    sources, seen = [], set()
    for query in plan_search_queries(llm, topic):
        for r in search_web(query, search_log):
            if r["url"] not in seen:
                seen.add(r["url"])
                sources.append(r)
        time.sleep(0.5)  # be gentle with DuckDuckGo

    if not sources:
        raise RuntimeError(
            "The web search returned no results (DuckDuckGo may be rate-limiting). "
            "Please wait a minute and try again."
        )

    notes = "\n\n".join(
        f"[{i}] {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}"
        for i, r in enumerate(sources[:20], 1)
    )

    # ---- Step 3: the single agent writes the report ----
    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Research the topic '{topic}' thoroughly and write an accurate, well-structured report.",
        backstory=(
            "You are an experienced analyst. You compare several sources, "
            "and you never invent facts or links."
        ),
        llm=llm,
        allow_delegation=False,  # single-agent app
        max_iter=3,
        verbose=False,
    )

    task = Task(
        description=(
            f"Write a research report on this topic: {topic}\n\n"
            "Below are web search results that were collected for you. "
            "Use ONLY information from these results, and only list URLs that appear in them. "
            "Do not invent facts, numbers or links.\n\n"
            f"SEARCH RESULTS:\n{notes}"
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

    crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential, verbose=False)
    return crew.kickoff().raw  # the final report text
