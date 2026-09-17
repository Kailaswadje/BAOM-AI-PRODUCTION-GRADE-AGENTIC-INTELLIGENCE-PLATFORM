# =============================================================================
# MODULE 4 — MULTI-AGENT ORCHESTRATION
# CrewAI + LangGraph: 4 specialised agents working together
# Researcher → Reasoner → Validator → Writer
# Professor feedback: orchestration MUST come after agentic engine (M3)
# =============================================================================
 
import os
import warnings

# Suppress LangChain's deprecation warning about the future default of
# `allowed_objects` in langgraph's cache serializer -- this is a
# harmless upstream library notice, not an error in our code.
warnings.filterwarnings(
    "ignore",
    category=Warning,
    message=".*allowed_objects.*"
)

from typing import TypedDict, Annotated, List
import warnings as _warnings
_original_warn = _warnings.warn
_warnings.warn = lambda *a, **k: None  # temporarily disable warn() itself
from langgraph.graph import StateGraph, END
_warnings.warn = _original_warn  # restore normal warning behaviour
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from agent.react_agent import ask as single_agent_ask
from neo4j import GraphDatabase
 
LLM_MODEL      = os.getenv("LLM_MODEL",      "llama3.2")
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Kailas@123")
 
 
# ------------------------------------------------------------------
# Shared state across all agents
# ------------------------------------------------------------------
class AgentState(TypedDict):
    question:        str
    research_result: str
    reasoning:       str
    validation:      str
    final_answer:    str
    confidence:      float
    cypher_queries:  List[str]
    gnn_used:        bool
    messages:        Annotated[list, "messages"]
 
 
# ------------------------------------------------------------------
# Deterministic confidence calculation.
#
# Earlier evaluation showed that asking the Reasoner LLM to self-report
# a numeric confidence (the CONFIDENCE: line below) produces
# non-reproducible values that do not reliably reflect real evidence
# (e.g. a fully correct, schema-verified answer scored only 0.50,
# while a fabricated GDPR-fines hallucination self-reported 0.20 only
# by chance). This function replaces that self-report with a
# deterministic score computed from objective evidence: whether the
# Researcher's Cypher queries actually returned data, and the
# Validator's independently-parsed VERIFIED/PARTIAL/UNVERIFIED status.
# ------------------------------------------------------------------
_NO_EVIDENCE_PHRASES = (
    "empty", "no results", "cannot provide", "cannot answer",
    "does not contain", "no data", "not available", "unable to",
    "cannot be answered", "ungrounded",
)


def evidence_confidence(research_result: str, cypher_queries: list) -> float:
    """Preliminary, deterministic confidence based on retrieval evidence
    alone (used for the re-research routing decision, before the
    Validator has run)."""
    text_lower = (research_result or "").lower()
    has_refusal_language = any(p in text_lower for p in _NO_EVIDENCE_PHRASES)
    has_cypher_evidence   = len(cypher_queries or []) > 0

    if has_refusal_language or not has_cypher_evidence:
        return 0.20
    return 0.65


def finalise_confidence(preliminary: float, validation_text: str) -> float:
    """Combine the preliminary evidence-based confidence with the
    Validator's independently-derived status, deterministically."""
    status = "UNVERIFIED"
    for line in (validation_text or "").split("\n"):
        if line.strip().upper().startswith("STATUS:"):
            status = line.split(":", 1)[1].strip().upper()
            break

    if preliminary <= 0.20:
        return 0.20
    if "VERIFIED" in status and "UNVERIFIED" not in status:
        return 0.90
    if "PARTIAL" in status:
        return 0.55
    return 0.35


# ------------------------------------------------------------------
# Agent 1: Researcher
# Retrieves raw data from Neo4j via the ReAct agent
# ------------------------------------------------------------------
def researcher_agent(state: AgentState) -> AgentState:
    print("[M4] Researcher agent: fetching graph data...")
    question = state["question"]
 
    result = single_agent_ask(question, verbose=False)
 
    state["research_result"] = result["answer"]
    state["cypher_queries"]  = result.get("cypher_used", [])
    state["gnn_used"]        = any(
        "GNN" in str(step) for step in result.get("reasoning", [])
    )
    state["confidence"] = evidence_confidence(state["research_result"], state["cypher_queries"])
    state["messages"].append(
        AIMessage(content=f"[Researcher] Found: {result['answer']}")
    )
    return state
 
 
# ------------------------------------------------------------------
# Agent 2: Reasoner
# Performs multi-hop reasoning using Think-on-Graph beam search pattern
# ------------------------------------------------------------------
def reasoner_agent(state: AgentState) -> AgentState:
    print("[M4] Reasoner agent: multi-hop reasoning...")
    llm      = OllamaLLM(model=LLM_MODEL, temperature=0.1)
 
    prompt = f"""You are an expert reasoning agent for a knowledge graph system.

Question: {state['question']}
Research findings: {state['research_result']}

CRITICAL RULE: Use ONLY the research findings above. Do NOT use your own
prior/parametric knowledge to fill in facts, names, numbers, or sources
that are not present in the research findings -- even if the question
sounds like something you know about from general knowledge (e.g. real
companies, real fines, real events). If the research findings do not
contain the information needed, explicitly say the graph does not
contain this information rather than inventing a plausible-sounding
answer.

Your task:
1. Identify the key entities and relationships involved, using ONLY the research findings.
2. Perform step-by-step multi-hop reasoning (A -> B -> C -> answer), using ONLY the research findings.
3. Identify any gaps in the research that suggest missing links or missing data.

Format your response as:
REASONING: <your step-by-step reasoning, grounded only in the research findings>
MISSING_LINKS: <any gaps detected, or note if the research findings do not cover this question at all>

Do NOT include a confidence score -- confidence is computed separately
by the system from objective retrieval evidence, not from your self-assessment.
"""
    response = llm.invoke(prompt)

    # NOTE: confidence is intentionally NOT parsed from the LLM's response
    # here (see evidence_confidence() / finalise_confidence() above).
    # state["confidence"] was already set by researcher_agent and is
    # refined later by validator_agent.
    state["reasoning"] = response
    state["messages"].append(
        AIMessage(content=f"[Reasoner] Preliminary confidence (evidence-based): {state['confidence']:.2f}")
    )
    return state

 
 
# ------------------------------------------------------------------
# Agent 3: Validator
# Verifies the answer against the graph — cross-checks with Neo4j
# ------------------------------------------------------------------
def validator_agent(state: AgentState) -> AgentState:
    print("[M4] Validator agent: cross-checking with graph...")
    llm    = OllamaLLM(model=LLM_MODEL, temperature=0)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
 
    # Get graph stats for validation context
    with driver.session() as session:
        stats = session.run(
            "MATCH (n) RETURN count(n) AS nodes"
        ).single()
        graph_nodes = stats["nodes"] if stats else 0
    driver.close()
 
    prompt = f"""You are a validation agent for a knowledge graph QA system.
 
Original question: {state['question']}
Research result: {state['research_result']}
Reasoning chain: {state['reasoning']}
GNN was used for missing links: {state['gnn_used']}
Graph size: {graph_nodes} nodes
 
Validate this answer:
1. Is the answer consistent with the question asked?
2. Is the reasoning chain logical and complete?
3. If GNN was used, note that predictions are probabilistic, not certain.
4. Assign a validation status: VERIFIED / PARTIAL / UNVERIFIED
 
Format:
STATUS: <VERIFIED/PARTIAL/UNVERIFIED>
NOTES: <validation notes>
CORRECTED_ANSWER: <if corrections needed, provide here, else write NONE>
"""
    response = llm.invoke(prompt)

    state["validation"]  = response
    state["confidence"]  = finalise_confidence(state["confidence"], response)
    state["messages"].append(
        AIMessage(content=f"[Validator] {response[:100]}...")
    )
    state["messages"].append(
        AIMessage(content=f"[Validator] Final deterministic confidence: {state['confidence']:.2f}")
    )
    return state

 
 
# ------------------------------------------------------------------
# Agent 4: Writer
# Produces the final clean, grounded answer
# ------------------------------------------------------------------
def writer_agent(state: AgentState) -> AgentState:
    print("[M4] Writer agent: composing final answer...")
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.2)
 
    prompt = f"""You are a writer agent. Compose a clear, concise, grounded answer.

Question: {state['question']}
Research: {state['research_result']}
Reasoning: {state['reasoning']}
Validation: {state['validation']}
Confidence: {state['confidence']:.2f}
GNN used: {state['gnn_used']}

CRITICAL RULE: Base your answer ONLY on the Research/Reasoning/Validation
above. NEVER add specific facts, names, numbers, dates, fines, or
sources from your own general knowledge that do not appear in the
Research/Reasoning/Validation text -- even if the question is about a
real-world topic you recognise. If the Research/Reasoning indicates no
relevant data was found, your answer MUST say the graph does not
contain this information, rather than inventing a plausible-sounding
factual answer with fabricated sources.

Rules:
- Answer must be directly responsive to the question.
- If confidence < 0.5, note uncertainty.
- If GNN was used, mention that some links are predicted, not stored.
- Keep the answer under 150 words.
- End with: Source: [Graph Query / GNN Prediction / Both]
"""
    final = llm.invoke(prompt)

    state["final_answer"] = final
    state["messages"].append(AIMessage(content=f"[Writer] {final}"))
    return state

 
 
# ------------------------------------------------------------------
# Routing: decide if re-research is needed
# ------------------------------------------------------------------
def should_re_research(state: AgentState) -> str:
    """Route back to researcher if confidence is very low."""
    if state["confidence"] < 0.2 and len(state["messages"]) < 10:
        return "re_research"
    return "write"
 
 
# ------------------------------------------------------------------
# Build the LangGraph orchestration workflow
# ------------------------------------------------------------------
def build_orchestration_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
 
    # Add nodes
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("reasoner",   reasoner_agent)
    workflow.add_node("validator",  validator_agent)
    workflow.add_node("writer",     writer_agent)
 
    # Define edges (dependency order as per professor's feedback)
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "reasoner")
    workflow.add_conditional_edges(
        "reasoner",
        should_re_research,
        {"re_research": "researcher", "write": "validator"}
    )
    workflow.add_edge("validator", "writer")
    workflow.add_edge("writer", END)
 
    return workflow.compile()
 
 
# ------------------------------------------------------------------
# Public interface: orchestrated ask
# ------------------------------------------------------------------
def orchestrated_ask(question: str) -> dict:
    """
    Ask using the full 4-agent orchestration pipeline.
    Returns structured result with full agent trace.
    """
    app = build_orchestration_graph()
 
    initial_state = AgentState(
        question=question,
        research_result="",
        reasoning="",
        validation="",
        final_answer="",
        confidence=0.0,
        cypher_queries=[],
        gnn_used=False,
        messages=[HumanMessage(content=question)]
    )
 
    result = app.invoke(initial_state)
 
    return {
        "question":      question,
        "final_answer":  result["final_answer"],
        "confidence":    result["confidence"],
        "gnn_used":      result["gnn_used"],
        "cypher_queries": result["cypher_queries"],
        "reasoning":     result["reasoning"],
        "validation":    result["validation"],
    }
 
 
if __name__ == "__main__":
    # One question per loaded dataset, so the multi-agent orchestration
    # pipeline is exercised against FB15k-237, Elliptic Bitcoin, and
    # Companies House UK, not only the general knowledge graph.
    questions = [
        # --- One grounded question per dataset (existing) ---
        "Which entities have the most connections, and what types of relationships do they have?",
        "How many transactions in the graph are flagged as fraudulent, and what does that suggest?",
        "Which directors are linked to which companies, and are any companies dissolved?",

        # --- Data-absent question: references data NOT in any loaded
        # dataset, to re-test the confidence/grounding mechanism under
        # the multi-agent orchestration pipeline specifically (not just
        # the single-agent one from Module 3). Expect a low, honest
        # confidence score and no fabricated specifics. ---
        "Which companies have been fined for GDPR data protection violations?",

        # --- Cross-dataset reasoning: forces the Researcher agent to
        # consider whether the general knowledge graph and the
        # corporate graph are connected at all. ---
        "Is there any relationship between entities in the general knowledge graph and the UK companies dataset?",

        # --- Overview / meta question spanning all three datasets ---
        "How many datasets are loaded in this database, and what does each one contain?",
    ]
    for q in questions:
        print(f"\nOrchestrated question: {q}\n")
        result = orchestrated_ask(q)
        print(f"\nFinal answer:\n{result['final_answer']}")
        print(f"\nConfidence: {result['confidence']:.2f}")
        print(f"GNN used: {result['gnn_used']}")
        print("\n" + "=" * 70)