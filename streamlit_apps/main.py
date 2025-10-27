"""
Enhanced Streamlit Monitor for Tavily Deep Research Agent
(Final optimized version — single run, guaranteed final report)

Streams real-time LangGraph events (nodes and models),
updates research_brief and success_criteria dynamically in sidebar,
and retrieves the final_report safely from memory via get_state().
"""

import asyncio
import json
import os, sys
from html import escape
from typing import Any
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# ===== ENV + PATH SETUP =====
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# ===== AGENT INITIALIZATION =====
from deep_research_from_scratch.tavily_deep_research_agent import deep_researcher_builder
checkpointer = InMemorySaver()
agent = deep_researcher_builder.compile(checkpointer=checkpointer)

# ===== STREAMLIT CONFIG =====
st.set_page_config(page_title="Tavily Deep Research Agent App", layout="centered")
st.title("Tavily Deep Research Agent")

# ===== CONSOLE LAYOUT =====
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Node Console**")
    node_console = st.empty()

with col2:
    st.markdown("**Model Console**")
    model_console = st.empty()

# ===== SIDEBAR INFO =====
sidebar_brief = st.sidebar.empty()
sidebar_criteria = st.sidebar.empty()
sidebar_brief.markdown("**Research Brief:**")
sidebar_criteria.markdown("**Success Criteria:**")

# ===== INPUT UI =====
st.markdown("---")
user_input = st.text_area(
    "Enter a research topic:",
    placeholder="What do you want to research?",
    height=200,
)

start_btn = st.button("Run Research")
final_report_box = st.empty()

# ===== SESSION CONTROL =====
if "is_running" not in st.session_state:
    st.session_state["is_running"] = False


# ===== UTILITY =====
def to_text(x: Any) -> str:
    """Safely convert LangGraph event payloads or message chunks to readable text."""
    if isinstance(x, str):
        return x
    try:
        from langchain_core.messages import BaseMessage
        if isinstance(x, BaseMessage):
            return str(getattr(x, "content", x))
    except Exception:
        pass
    try:
        return json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        return str(x)


def update_console(console_placeholder, buffer, height=600):
    """Render console buffer with auto-scroll."""
    html = f"""
    <div id='console' style='height:{height}px; overflow-y:auto; border:1px solid #ccc;
                padding:8px; background-color:#fafafa; font-family:monospace;
                white-space:pre-wrap;'>
        {"<br>".join(buffer[-400:])}
    </div>
    <script>
        var consoleBox = document.getElementById('console');
        if (consoleBox) {{
            consoleBox.scrollTop = consoleBox.scrollHeight;
        }}
    </script>
    """
    console_placeholder.markdown(html, unsafe_allow_html=True)


# ===== STREAMING FUNCTION =====
async def run_agent_stream(user_input: str):
    node_buffer = []
    model_buffer = []
    thread_config = {"configurable": {"thread_id": "1", "recursion_limit": 50}}

    async for e in agent.astream_events(
        {"messages": [HumanMessage(content=user_input)]},
        stream_mode="events",
        config=thread_config,
    ):
        event = e.get("event")
        data = e.get("data", {})
        name = e.get("name", "")
        meta = e.get("metadata", {})
        node = meta.get("langgraph_node")

        # ===== NODE EVENTS =====
        if event == "on_chain_start":
            node_buffer.append(f"[NODE START] {node or name}")
        elif event == "on_chain_end":
            node_buffer.append(f"[NODE END] {node or name}")
        elif event == "on_graph_end":
            node_buffer.append("[GRAPH END] Execution complete.")

        # ===== MODEL STREAM =====
        elif event == "on_chat_model_stream":
            chunk = to_text(data.get("chunk", ""))
            if chunk.strip():
                if model_buffer and model_buffer[-1].startswith("[stream]"):
                    model_buffer[-1] += chunk
                else:
                    model_buffer.append("[stream] " + chunk)
        elif event == "on_chat_model_end":
            if model_buffer and model_buffer[-1].startswith("[stream]"):
                model_buffer[-1] = model_buffer[-1].replace("[stream] ", "")
            model_buffer.append("[MODEL OUTPUT END]")

        # ===== AGENTSTATE UPDATES =====
        if event == "on_chain_end":
            out = data.get("output")
            if isinstance(out, dict):
                brief = out.get("research_brief")
                if brief:
                    sidebar_brief.markdown(f"**Research Brief:**\n\n{escape(str(brief))}")

                criteria = out.get("success_criteria")
                if isinstance(criteria, dict) and criteria:
                    formatted = "<br>".join(
                        [f"- {escape(k)}: {'Complete' if v else 'Not Complete'}"
                         for k, v in criteria.items()]
                    )
                    sidebar_criteria.markdown(
                        f"**Success Criteria:**<br>{formatted}",
                        unsafe_allow_html=True,
                    )

        # ===== RENDER LIVE =====
        update_console(node_console, node_buffer)
        update_console(model_console, model_buffer)

    # ===== RETRIEVE FINAL STATE SAFELY =====
    final_state = agent.get_state(thread_config["configurable"])

    report = ""
    if isinstance(final_state, dict):
        report = final_state.get("final_report", "")

    final_report_box.markdown(f"### Final Report\n\n{report or '(No final report generated)'}")


# ===== MAIN ACTION =====
if start_btn and user_input and not st.session_state["is_running"]:
    st.session_state["is_running"] = True
    st.info("Running agent and streaming events...")
    try:
        asyncio.run(run_agent_stream(user_input))
    finally:
        st.session_state["is_running"] = False
