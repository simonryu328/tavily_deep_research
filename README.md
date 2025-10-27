# Tavily Deep Research Agent

### Executive Summary

Built with **LangGraph**, **Tavily**, and **OpenAI/watsonx.ai**, this project is a **personal research agent** that autonomously explores, maps, and synthesizes information through a **custom-built chat interface** in **React**.

![demo](/assets/assistant_ui.gif)


### Prerequisites

- **Node.js and npx** (required for MCP server in notebook 3):
```bash
# Install Node.js (includes npx)
# On macOS with Homebrew:
brew install node

# On Ubuntu/Debian:
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation:
node --version
npx --version
```

- Ensure you're using Python 3.11 or later.
- This version is required for optimal compatibility with LangGraph.
```bash
python3 --version
```
- [uv](https://docs.astral.sh/uv/) package manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Update PATH to use the new uv version
export PATH="/Users/$USER/.local/bin:$PATH"
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/langchain-ai/deep_research_from_scratch
cd deep_research_from_scratch
```

2. Install the package and dependencies (this automatically creates and manages the virtual environment):
```bash
uv sync
```

3. Create a `.env` file in the project root with your API keys:
```bash
# Create .env file
touch .env
```

Add your API keys to the `.env` file:
```env
# Required for research agents with external search
TAVILY_API_KEY=your_tavily_api_key_here

# Required for model usage
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: For evaluation and tracing
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=deep_research_from_scratch
```

4. Run notebooks or code using uv:
```bash
# Run Jupyter notebooks directly
uv run jupyter notebook

# Or activate the virtual environment if preferred
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
jupyter notebook
```

5. Run the UI:

- assistant-ui
```bash
cd assistant_ui_app
npm run dev
```

- streamlit
```bash
uv run streamlit run streamlit_apps/main.py
```

---

## **1. Tavily API Expansion**

### **Original Implementation**
```python
# Original: Single search tool
tools = [tavily_search, think_tool]
```

### **Enhanced Implementation**
```python
# Enhanced: Multi-tool strategic workflow
tools = [tavily_search, tavily_extract, tavily_map, think_tool]
```

### **Strategic Research Pattern**

**Before**: `Search → Think → Search → Think` (limited extraction and search depth)

**After**: `Search → Map → Extract → Think` (comprehensive exploration)

#### **New Tool Capabilities**

- **`tavily_extract`**: Extract and summarize full webpage content from specific URLs for deep content analysis
- **`tavily_map`**: Discover all internal pages from authoritative domains for systematic exploration

#### **Research Workflow Evolution**

The expanded toolkit enables a research progression:

```
🔍` Search (broad exploration) 
    ↓
🗺️ Map (systematic domain exploration)
    ↓  
📄 Extract (deep content analysis)
    ↓
🧠 Think (strategic evaluation)
```

This pattern enables:
- **Broad exploration** with comprehensive searches
- **Systematic domain mapping** of authoritative sources  
- **Targeted content extraction** for deep analysis
- **Context continuity** throughout the research process

---

## **2. Research Brief Generation: From Questions to Strategic Briefs**

### **Original Prompt** (Before Enhancement)

```python
transform_messages_into_research_topic_prompt = """
Your job is to translate these messages into a more detailed and concrete research question that will be used to guide the research.

You will return a single research question that will be used to guide the research.
"""
```

**Limitations**: Single question focus, no strategic planning, no success criteria, limited context interpretation

### **Enhanced Strategic Brief Generation** (Commit: `1d3f1f4`)

```python
transform_messages_into_research_topic_prompt = """
<role>
You are an expert research strategist who transforms conversations into comprehensive, actionable research briefs that demonstrate deep understanding of user intent and strategic planning.
</role>

<task>
Generate a strategic research brief from the conversation that interprets underlying needs, frames concrete objectives, identifies constraints, specifies deliverables, and acknowledges uncertainties.
</task>
```

### **Strategic Dimensions**

1. **Intent Understanding**: What is the user trying to accomplish?
2. **Constraint Identification**: Hard requirements vs. soft preferences
3. **Context Synthesis**: Timing, urgency, domain expertise, risk factors
4. **Deliverable Clarity**: Specific, actionable outputs with imperative framing
5. **Source Standards**: What's authoritative? What needs verification?
6. **Ethical Boundaries**: Appropriate disclaimers and professional consultation notes
7. **Uncertainty**: What's confirmed vs. inferred vs. assumed?

### **Example Transformation**

**Before**: 
```
I want to understand the underlying architecture of IBM watsonx Orchestrate, with a specific focus on how its orchestration and agent runtime layer is built.
My goal is to identify exactly which frameworks, technologies, and APIs are integrated into this layer, and to determine whether it uses LangGraph, Langfuse, or Langflow as part of its implementation.
Please provide a detailed breakdown of the orchestration and agent runtime components, listing all known frameworks and APIs involved, and clarify the role (if any) of LangGraph, Langfuse, and Langflow.
If there are proprietary IBM tools or other third- party frameworks in use, include those as well. Unless otherwise specified in public technical documentation, treat any unspecified architectural details as open for investigation.
Prioritize information from official IBM documentation, technical whitepapers, and reputable developer sources.
```

**After**: 
```
I'm investigating the underlying architecture of IBM watsonx Orchestrate, specifically the orchestration and agent runtime layer—not marketing overviews of what the product does. 
I need implementation-level detail: which frameworks are integrated, what APIs are exposed to developers, and definitive answers on whether LangGraph, Langfuse, or Langflow are part of the system.

**Deliver the following:** Definitive integration status for each framework (LangGraph: yes/no with evidence, Langfuse: yes/no with evidence, Langflow: yes/no with evidence).
Document the developer-facing APIs and SDKs available for building with watsonx Orchestrate...

### Success Criteria
- LangGraph integration status confirmed: yes or no with specific evidence from official sources
- Langfuse integration status confirmed: yes or no with specific evidence from official sources
- Langflow integration status confirmed: yes or no with specific evidence from official sources
- Developer-facing APIs documented: specific endpoints, protocols, authentication methods
- Orchestration architecture patterns described: how supervisor/routing/coordination works
- Clear distinction made: official IBM docs vs expert technical blogs vs inferred information
```

---

## **3. Success Criteria Implementation: Research Completion Tracking**

### **State Management Enhancement** (Commit: `f5f60b9`)

```python
class AgentState(MessagesState):
    # Use dictionary for key–value success criteria tracking
    success_criteria: Annotated[Dict[str, bool], operator.or_] = Field(default_factory=dict)
```

### **Success Criteria Parsing** (Commit: `b321655`)

```python
def parse_success_criteria(state: AgentState):
    """
    Extracts success criteria from the research brief and updates the AgentState
    with a dictionary mapping each criterion to False (not yet evaluated).
    """
    brief = state.get("research_brief", "")
    criteria_dict = {}
    
    # Extract text after "Success Criteria" section
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
```

### **Implementation Status**

**Implemented**: Success criteria extraction, dictionary-based tracking, integration with research brief generation

**Next Steps**: Real-time evaluation, dynamic completion logic, progress visualization

---

## **4. Research Agent Prompt Evolution: Strategic Research Guidance**

### **Original Prompt Focus**
```python
research_agent_prompt = """You are a research assistant conducting research on the user's input topic.
Your job is to use tools to gather information about the user's input topic.
```

### **Enhanced Strategic Prompt** (Commit: `beffaa5`)

```python
research_agent_prompt = """You are an expert research assistant conducting strategic research based on the user's research brief and success criteria. Your goal is to deliver findings that directly address the brief's objectives.
```

### **Key Improvements**

1. **Brief-Driven Research**: Research guided by strategic brief rather than simple topic
2. **Success Criteria Integration**: Research decisions informed by completion criteria
3. **Tool Selection Strategy**: Clear guidance on when to use each tool
4. **Transparent Thinking**: Emphasis on making research process auditable

### **Tool Usage Guidance Evolution**

**Before**: Generic tool calling with basic search patterns

**After**: Strategic tool progression with clear decision criteria:

```python
**Research Progression**: Start broad (search) → explore promising domains (map) → extract targeted content (extract) → reflect and decide (think). Adapt this flow based on what you discover.

**CRITICAL**: Always use `think_tool` after each search, map, or extraction to make your research process transparent and auditable.
```

---

## **5. Future Enhancement Roadmap**

### **Next Steps**
1. **Success Criteria Evaluation**: Real-time criteria satisfaction tracking
2. **Dynamic Completion Logic**: Criteria-based research termination
3. **Progress Visualization**: UI indicators for research progress

### **Future Enhancements**
1. **Adaptive Tool Selection**: ML-driven tool choice based on research patterns
2. **Multi-Domain Specialization**: Specialized research strategies for different domains
3. **Collaborative Research**: Multi-user research coordination

---

This documentation focuses on the architectural decisions and improvements made since the fork, highlighting how the expanded Tavily API toolkit improved research quality and strategic planning capabilities.

