"""
Enhanced Streamlit Monitor for Tavily Deep Research Agent

Streams real-time LangGraph events (nodes, tools, and models)
in a clean, scrollable console with optional event filtering.
"""

import asyncio
import json
import os, sys
import streamlit as st
from dotenv import load_dotenv
from html import escape
from typing import Any
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

# ===== ENV + PATH SETUP =====
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from deep_research_from_scratch.tavily_deep_research_agent import deep_researcher_builder

# Compile the agent with a checkpointer
checkpointer = InMemorySaver()
agent = deep_researcher_builder.compile(checkpointer=checkpointer)

# ===== STREAMLIT CONFIG =====
st.set_page_config(page_title="Tavily Research Agent", layout="centered")
st.title("Tavily Research Agent")

# ===== SIDEBAR FILTERS =====
st.sidebar.header("Display Filters")
show_models = st.sidebar.checkbox("Show model streaming", value=True)
show_tools = st.sidebar.checkbox("Show tool events", value=True)
show_nodes = st.sidebar.checkbox("Show node transitions", value=True)

# ===== USER INPUT =====
user_input = st.text_input("Enter a research topic:", placeholder="e.g., 'AI safety regulation 2024'")
start_btn = st.button("Run Research")

console_box = st.empty()
final_report_box = st.empty()

# ===== UTILITIES =====
def to_text(x: Any) -> str:
    """Convert any LangGraph payload to safe text."""
    if isinstance(x, str):
        return x
    if isinstance(x, BaseMessage):
        content = getattr(x, "content", None)
        return str(content) if content is not None else str(x)
    try:
        return json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        return repr(x)

def update_console(log_buffer):
    """Render the last 300 lines in a scrollable box."""
    lines = [escape(to_text(s)) for s in log_buffer[-300:]]
    html = """
    <div style='height:420px; overflow-y:scroll; border:1px solid #ccc;
                padding:8px; background-color:#fafafa; font-family:monospace;
                white-space:pre-wrap; text-align:left;'>
    """ + "<br>".join(lines) + "</div>"
    console_box.markdown(html, unsafe_allow_html=True)

# ===== STREAMING FUNCTION =====
async def run_agent_stream():
    log_buffer = []
    last_node_start = None

    config = {
        "recursion_limit": 50,
        "configurable": {"thread_id": "session-1"},
    }

    async for e in agent.astream_events(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode="events",
        config=config,
    ):
        event = e.get("event")
        data = e.get("data", {})
        name = e.get("name")
        meta = e.get("metadata", {})
        node = meta.get("langgraph_node")

        # ===== FILTERS =====
        if event.startswith("on_chat_model") and not show_models:
            continue
        if event.startswith("on_tool") and not show_tools:
            continue
        if event.startswith("on_chain") and not show_nodes:
            continue

        # ===== STREAMING LOGIC =====
        if event == "on_chat_model_stream":
            chunk = to_text(data.get("chunk", ""))
            if not chunk.strip():
                update_console(log_buffer)
                continue
            if log_buffer and log_buffer[-1].startswith("[stream]"):
                log_buffer[-1] += chunk
            else:
                log_buffer.append("[stream] " + chunk)

        elif event == "on_chat_model_end":
            if log_buffer and log_buffer[-1].startswith("[stream]"):
                log_buffer[-1] = log_buffer[-1].replace("[stream] ", "").strip()

        elif event == "on_chain_start":
            msg = f"[NODE START] {to_text(node or name)}"
            if msg != last_node_start:
                log_buffer.append(msg)
                last_node_start = msg

        elif event == "on_chain_end":
            log_buffer.append(f"[NODE END] {to_text(node or name)}")

        elif event == "on_tool_start":
            log_buffer.append(f"[TOOL START] {to_text(name)} input: {to_text(data.get('input', ''))}")

        elif event == "on_tool_end":
            output = data.get("output", "")
            summary = to_text(output)
            if len(summary) > 200:
                summary = summary[:200] + " ..."
            log_buffer.append(f"[TOOL END] {to_text(name)} → {summary}")

        elif event == "on_graph_end":
            log_buffer.append("[GRAPH END] Execution complete.")

        else:
            log_buffer.append(f"[{event}] {to_text(name or '')}")

        update_console(log_buffer)

    # ===== FINAL REPORT =====
    final_state = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    report = final_state.get("final_report", "(No final report generated)")
    final_report_box.markdown(f"### Final Report\n\n{to_text(report)}")

# ===== MAIN ACTION =====
if start_btn and user_input:
    st.info("Running agent and streaming events...")
    asyncio.run(run_agent_stream())
