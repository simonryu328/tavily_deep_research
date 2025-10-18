
"""State Definitions and Pydantic Schemas for Research Scoping.

This defines the state objects and structured schemas used for
the research agent scoping workflow, including researcher state management and output schemas.
"""

import operator
from typing_extensions import Optional, Annotated, List, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing import Dict

# ===== STATE DEFINITIONS =====

class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    pass

class AgentState(MessagesState):
    """
    Main state for the full multi-agent research system.

    Extends MessagesState with additional fields for research coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.
    """

    # Research brief generated from user conversation history
    research_brief: Optional[str]
    # Use dictionary for key–value success criteria tracking
    success_criteria: Annotated[Dict[str, bool], operator.or_] = Field(default_factory=dict)
    # Messages exchanged with the supervisor agent for coordination
    supervisor_messages: Annotated[Sequence[BaseMessage], operator.add]
    # Raw unprocessed research notes collected during the research phase
    raw_notes: Annotated[list[str], operator.add] = []
    # Processed and structured notes ready for report generation
    notes: Annotated[list[str], operator.add] = []
    # Final formatted research report
    final_report: Optional[str]

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decision and questions."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Schema for structured research brief generation.

    Represents a complete research planning output that includes:
    - a detailed research brief describing what and how to research, and
    - a list of success criteria defining what constitutes a high-quality outcome.
    """

    research_brief: str = Field(
        description=(
            "A detailed research brief that interprets the user’s intent, "
            "frames objectives, identifies constraints, and specifies deliverables "
            "to guide the research process."
        ),
    )

    # success_criteria: List[str] = Field(
    #     description=(
    #         "A list of specific, measurable criteria that define what successful "
    #         "research looks like. Each item should describe a clear quality or "
    #         "completeness condition (e.g., 'Includes official sources and verified dates')."
    #     ),
    #)
