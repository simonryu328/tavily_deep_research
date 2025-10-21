"""
Enhanced Streamlit Monitor for Tavily Deep Research Agent

Displays real-time LangGraph events (nodes and models)
and dynamically updates research_brief and success_criteria
from AgentState in the sidebar.
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

from deep_research_from_scratch.tavily_deep_research_agent import agent, deep_researcher_builder

# Compile the agent with checkpointer
# checkpointer = InMemorySaver()
# agent = deep_researcher_builder.compile(checkpointer=checkpointer)
# agent = deep_researcher_builder.compile()


# ===== STREAMLIT CONFIG =====
st.set_page_config(page_title="Tavily Deep Research Agent App", layout="centered")
st.title("Tavily Deep Research Agent")

# ===== CONSOLE LAYOUT (TOP) =====
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Node Console**")
    node_console = st.empty()

with col2:
    st.markdown("**Model Console**")
    model_console = st.empty()

# ===== SIDEBAR INFO =====
st.sidebar.header("Agent State (Live Updates)")
sidebar_brief = st.sidebar.empty()
sidebar_criteria = st.sidebar.empty()

# Initialize placeholders
sidebar_brief.markdown("**Research Brief:** _(awaiting update...)_")
sidebar_criteria.markdown("**Success Criteria:** _(awaiting update...)_")

# ===== INPUT UI =====
st.markdown("---")
user_input = st.text_area(
    "Enter a research topic:",
    placeholder="What do you want to research?",
    height=120,  # adjust as you like (in px)
)

start_btn = st.button("Run Research")

final_report_box = st.empty()

# ===== TEXT CONVERSION UTILITY =====
def to_text(x: Any) -> str:
    """Safely convert LangGraph event payloads or message chunks to readable text."""
    if isinstance(x, str):
        return x
    try:
        from langchain_core.messages import BaseMessage
        if isinstance(x, BaseMessage):
            content = getattr(x, "content", None)
            return str(content or x)
    except Exception:
        pass
    try:
        return json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        return str(x)

# ===== RENDERER =====
def update_console(console_placeholder, buffer, height=600):
    """Render the console buffer with auto-scroll."""
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
async def run_agent_stream():
    node_buffer = []
    model_buffer = []

    thread_config = {
        "configurable": {
            "thread_id": "1",
            "recursion_limit": 50,
        }
    }

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

        # ===== MODEL STREAMING =====
        elif event == "on_chat_model_stream":
            chunk = to_text(data.get("chunk", ""))
            if not chunk.strip():
                continue
            if model_buffer and model_buffer[-1].startswith("[stream]"):
                model_buffer[-1] += chunk
            else:
                model_buffer.append("[stream] " + chunk)
        elif event == "on_chat_model_end":
            if model_buffer and model_buffer[-1].startswith("[stream]"):
                model_buffer[-1] = model_buffer[-1].replace("[stream] ", "")
            model_buffer.append("[MODEL OUTPUT END]")

        # ===== AGENTSTATE EXTRACTION (fix) =====
        if event == "on_chain_end":
            out = data.get("output")

            # We only care about AgentState-like dicts (TypedDict is a dict at runtime)
            if isinstance(out, dict):
                # research_brief
                brief = out.get("research_brief")
                if brief:
                    sidebar_brief.markdown(f"**Research Brief:**\n\n{escape(str(brief))}")

                # success_criteria
                criteria = out.get("success_criteria")
                if isinstance(criteria, dict) and criteria:
                    formatted = "<br>".join(
                        [f"- {escape(k)}: {'✅' if v else '❌'}" for k, v in criteria.items()]
                    )
                    sidebar_criteria.markdown(
                        f"**Success Criteria:**<br>{formatted}", unsafe_allow_html=True
                    )

        # ===== RENDER UPDATES =====
        update_console(node_console, node_buffer)
        update_console(model_console, model_buffer)

    # ===== FINAL REPORT =====
    final_state = await agent.ainvoke({"messages": [HumanMessage(content=user_input)]})
    report = ""
    if isinstance(final_state, dict):
        report = final_state.get("final_report", "")
    final_report_box.markdown(f"### Final Report\n\n{report or '(No final report generated)'}")

# ===== MAIN ACTION =====
if start_btn and user_input:
    st.info("Running agent and streaming events...")
    asyncio.run(run_agent_stream())
