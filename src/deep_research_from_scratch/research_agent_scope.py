
"""User Clarification and Research Brief Generation.

This module implements the scoping phase of the research workflow, where we:
1. Assess if the user's request needs clarification
2. Generate a detailed research brief from the conversation

The workflow uses structured output to make deterministic decisions about
whether sufficient context exists to proceed with research.
"""
import re
from datetime import datetime
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research_from_scratch.prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt
from deep_research_from_scratch.state_scope import AgentState, ClarifyWithUser, ResearchQuestion, AgentInputState

# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

# ===== CONFIGURATION =====

# Initialize model
model = init_chat_model(model="openai:gpt-4.1", temperature=0.0)

# ===== WORKFLOW NODES =====

def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with research.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """
    # Set up structured output model
    structured_output_model = model.with_structured_output(ClarifyWithUser)

    # Invoke the model with clarification instructions
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]), 
            date=get_today_str()
        ))
    ])

    # Route based on clarification need
    if response.need_clarification:
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )

def write_research_brief(state: AgentState):
    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """
    # Set up structured output model
    structured_output_model = model.with_structured_output(ResearchQuestion)

    # Generate research brief from conversation history
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    # Convert criteria list → dict with all False initially (not yet evaluated)
    # success_criteria_dict = {criterion: False for criterion in response.success_criteria}

    return {
        "research_brief": response.research_brief,
        # "success_criteria": success_criteria_dict,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")]
    }


def parse_success_criteria(state: AgentState):
    """
    Extracts success criteria from the research brief and updates the AgentState
    with a dictionary mapping each criterion to False (not yet evaluated).
    """
    brief = state.get("research_brief", "")
    criteria_dict = {}

    if not brief:
        return {"success_criteria": {}}

    # --- Extract text after "Success Criteria" section ---
    match = re.search(r"Success Criteria(.*)", brief, flags=re.IGNORECASE | re.DOTALL)
    if match:
        section_text = match.group(1).strip()
        # Capture each bullet (• or -) line as an individual criterion
        lines = re.findall(r"[•\-]\s*(.+)", section_text)
        for line in lines:
            clean_line = re.sub(r"\s+", " ", line).strip()
            if clean_line:
                criteria_dict[clean_line] = False

    return {"success_criteria": criteria_dict}

# ===== GRAPH CONSTRUCTION =====

deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("parse_success_criteria", parse_success_criteria)  # ✅ new node

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("clarify_with_user", "write_research_brief")
deep_researcher_builder.add_edge("write_research_brief", "parse_success_criteria")  # ✅ link new node
deep_researcher_builder.add_edge("parse_success_criteria", END)

# Compile the workflow
scope_research = deep_researcher_builder.compile()
