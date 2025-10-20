"""
Minimal Streamlit monitor for Tavily Deep Research Agent.

Streams real-time LangGraph events (node transitions, tool inputs/outputs,
and final report) directly from the compiled agent.
"""

import streamlit as st
import asyncio
import sys, os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# ===== ENV + PATH SETUP =====
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from deep_research_from_scratch.tavily_deep_research_agent import agent

# ===== STREAMLIT CONFIGURATION =====
st.set_page_config(page_title="Tavily Deep Research Monitor", layout="centered")

# ===== PAGE STYLE =====
st.markdown(
    """
    <style>
        /* Center all main content */
        .block-container {
            max-width: 900px;
            margin: 0 auto;
            padding-top: 2rem;
        }

        /* Center the title and inputs */
        h1, h2, h3, h4, h5, h6, p, label {
            text-align: center;
        }

        /* Style console box */
        .scroll-box {
            height: 400px;
            overflow-y: scroll;
            border: 1px solid #ccc;
            padding: 8px;
            background-color: #f9f9f9;
            font-family: monospace;
            line-height: 1.4;
            text-align: left;
        }

        /* Center button */
        div[data-testid="stButton"] {
            display: flex;
            justify-content: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== TITLE AND INPUT =====
st.title("Tavily Deep Research — Live Graph Stream")
user_input = st.text_input(
    "Enter a research topic:",
    placeholder="e.g., 'AI safety regulation 2024'",
)

# Containers
trace_container = st.container()
final_report_box = st.empty()

from html import escape
import json
from typing import Any

def to_text(x: Any) -> str:
    """Coerce any event payload to a safe display string."""
    # Already a string
    if isinstance(x, str):
        return x

    # LangChain message objects (e.g., AIMessageChunk, BaseMessage, HumanMessage, etc.)
    try:
        from langchain_core.messages import BaseMessage
        if isinstance(x, BaseMessage):
            # Prefer content if available
            content = getattr(x, "content", None)
            if content is not None:
                return str(content)
            return str(x)
    except Exception:
        pass

    # Dicts / lists → JSON (fallback to repr on failure)
    try:
        return json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        return repr(x)


# ===== ASYNC STREAMING =====
async def run_agent_stream():
    """Stream LangGraph events in real-time with aggregated text streaming."""
    console = st.empty()
    log_buffer = []

    def update_console():
        # Safely escape and render last 200 lines
        from html import escape
        lines = [escape(to_text(s)) for s in log_buffer[-200:]]
        html = "<div class='scroll-box'>" + "<br>".join(lines) + "</div>"
        console.markdown(html, unsafe_allow_html=True)

    thread_config = {
        "configurable": {
            "thread_id": "1",
            "recursion_limit": 50,  # set recursion depth here
        }
    }

    async for e in agent.astream_events(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode="events",
        config=thread_config,
    ):
        event = e.get("event")
        data = e.get("data", {})
        name = e.get("name")
        meta = e.get("metadata", {})
        node = meta.get("langgraph_node")

        # ====== STREAMING HANDLING ======
        if event == "on_chat_model_stream":
            chunk = to_text(data.get("chunk", ""))
            if not chunk.strip():
                continue

            # If previous line is a stream, append to it
            if log_buffer and log_buffer[-1].startswith("[stream]"):
                log_buffer[-1] += chunk
            else:
                log_buffer.append("[stream] " + chunk)

        elif event == "on_chat_model_end":
            # When model finishes, remove [stream] tag for readability
            if log_buffer and log_buffer[-1].startswith("[stream]"):
                log_buffer[-1] = log_buffer[-1].replace("[stream] ", "").strip()

        # ====== NODE + TOOL EVENTS ======
        elif event == "on_chain_start":
            log_buffer.append(f"Starting node: {to_text(node or name)}")

        elif event == "on_tool_start":
            log_buffer.append(f"Tool started: {to_text(name)} (node: {to_text(node)})")

        elif event == "on_tool_end":
            output = data.get("output", "")
            summary = to_text(output)
            if len(summary) > 120:
                summary = summary[:120] + " ..."
            log_buffer.append(f"Tool completed: {to_text(name)} → {summary}")

        elif event == "on_chain_end":
            log_buffer.append(f"Completed node: {to_text(node or name)}")

        elif event == "on_graph_end":
            log_buffer.append("Graph execution complete.")

        else:
            log_buffer.append(f"Event: {to_text(event)} (name: {to_text(name)})")

        update_console()

    # ====== FINAL REPORT ======
    final_state = await agent.ainvoke({"messages": [HumanMessage(content=user_input)]})
    report = final_state.get("final_report", "(No final report generated)")
    final_report_box.markdown(f"### Final Report\n\n{to_text(report)}")

# ===== MAIN UI ACTION =====
if st.button("Run Research"):
    st.info("Running research agent and streaming events...")
    asyncio.run(run_agent_stream())
