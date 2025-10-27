
"""
Full Multi-Agent Research System

This module integrates all components of the research system:
- User clarification and scoping
- Research brief generation  
- Multi-agent research coordination
- Final report generation

The system orchestrates the complete research workflow from initial user
input through final report delivery.
"""

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from deep_research_from_scratch.utils import get_today_str
from deep_research_from_scratch.prompts import final_report_generation_prompt
from deep_research_from_scratch.state_scope import AgentState, AgentInputState
from deep_research_from_scratch.research_agent_scope import clarify_with_user, write_research_brief, parse_success_criteria
from deep_research_from_scratch.multi_agent_supervisor import supervisor_agent
from deep_research_from_scratch.research_agent import researcher_agent

# ===== Config =====

from langchain.chat_models import init_chat_model
writer_model = init_chat_model(model="openai:gpt-4.1", max_tokens=32000) # model="anthropic:claude-sonnet-4-20250514", max_tokens=64000

# ===== FINAL REPORT GENERATION =====

from deep_research_from_scratch.state_scope import AgentState

def run_researcher_agent(state: AgentState):
    """Bridge node that runs the researcher subgraph after scoping."""

    research_topic = state.get("research_brief", "")
    criteria = state.get("success_criteria", {})
    supervisor_msgs = state.get("supervisor_messages", [])

    researcher_input = {
        "researcher_messages": supervisor_msgs,
        "tool_call_iterations": 0,
        "research_topic": research_topic,
        "compressed_research": "",
        "raw_notes": [],
        "success_criteria": criteria,
    }

    research_output = researcher_agent.invoke(researcher_input)

    return {
        "raw_notes": research_output.get("raw_notes", []),
        "supervisor_messages": research_output.get("researcher_messages", []),
        "final_report": research_output.get("compressed_research", ""),
        "success_criteria": research_output.get("success_criteria", criteria),
    }

async def final_report_generation(state: AgentState):
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """

    notes = state.get("raw_notes", [])

    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str()
    )

    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])

    return {
        "final_report": final_report.content, 
        "messages": ["Here is the final report: " + final_report.content],
    }

# ===== GRAPH CONSTRUCTION =====
# Build the overall workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# ===== Add workflow nodes =====
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("parse_success_criteria", parse_success_criteria)
deep_researcher_builder.add_node("run_researcher_agent", run_researcher_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# ===== Connect the edges =====
deep_researcher_builder.add_edge(START, "clarify_with_user")
# deep_researcher_builder.add_edge("clarify_with_user", "write_research_brief")
deep_researcher_builder.add_edge("write_research_brief", "parse_success_criteria")
deep_researcher_builder.add_edge("parse_success_criteria", "run_researcher_agent")
deep_researcher_builder.add_edge("run_researcher_agent", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# ===== Compile the full workflow =====
agent = deep_researcher_builder.compile()
