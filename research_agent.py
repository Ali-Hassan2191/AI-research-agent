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


# ---------- 0. FIX: Groq rejects CrewAI's internal "cache_breakpoint" flag ----------
class GroqLLM(LLM):
    """Same as CrewAI's LLM, but removes the 'cache_breakpoint' key from every
    message before it is sent to Groq (Groq returns a 400 error if it is present)."""

    def supports_function_calling(self) -> bool:
        # gpt-oss has its own built-in browser tools (open, find, ...) and tries to call
        # them, which Groq rejects. Returning False makes CrewAI use plain-text
        # "Action / Action Input" mode, so no native tool calls are sent to Groq at all.
        return False

    def _format_messages_for_provider(self, messages):
        formatted = super()._format_messages_for_provider(messages)
        return [{k: v for k, v in m.items() if k != "cache_breakpoint"} for m in formatted]


# ---------- 1. TOOL: free DuckDuckGo search ----------
# NOTE: the tool is NOT called "web_search" on purpose. gpt-oss models have a built-in
# tool with that name and get confused (they send wrong arguments like "cursor"/"id").
def make_search_tool(search_log: list):
    """Creates the search tool. Every search is also saved into `search_log`
    so the UI can show it in the 'Search log' tab."""

    @tool("search_duckduckgo")
    def search_duckduckgo(search_query: str) -> str:
        """Search the internet with DuckDuckGo.
        Argument 'search_query' (required): the text to search for, e.g. 'solar battery market 2026'.
        Returns titles, links and snippets of the top results."""
        try:
            results = DDGS().text(search_query, max_results=6)
        except Exception as e:
            search_log.append({"query": search_query, "results": [], "error": str(e)})
            return f"Search failed: {e}. Try a different, shorter query."

        if not results:
            search_log.append({"query": search_query, "results": []})
            return "No results found. Try a different query."

        search_log.append({
            "query": search_query,
            "results": [{"title": r.get("title"), "url": r.get("href")} for r in results],
        })
        lines = []
        for r in results:
            lines.append(f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}\n")
        return "\n".join(lines)

    return search_duckduckgo


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

    # ---- The single agent ----
    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Research the topic '{topic}' thoroughly and write an accurate, well-structured report.",
        backstory=(
            "You are an experienced analyst. You always search the web for up-to-date "
            "information, compare several sources, and never invent facts or links."
        ),
        tools=[make_search_tool(search_log)],
        llm=llm,
        allow_delegation=False,  # single-agent app
        max_iter=8,              # max reasoning/search steps (prevents endless loops)
        verbose=False,
    )

    # ---- The task ----
    task = Task(
        description=(
            f"Research this topic: {topic}\n\n"
            "Use the search_duckduckgo tool 3-5 times, always passing a 'search_query' text, to gather facts. "
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

    # gpt-oss sometimes makes a malformed tool call. Retrying once or twice usually fixes it.
    last_error = None
    for attempt in range(3):
        try:
            result = crew.kickoff()
            return result.raw  # the final report text
        except Exception as e:
            last_error = e
            if "tool_use_failed" not in str(e) and "tool call validation" not in str(e).lower():
                raise  # a different error (e.g. wrong API key) - don't retry
    raise last_error
