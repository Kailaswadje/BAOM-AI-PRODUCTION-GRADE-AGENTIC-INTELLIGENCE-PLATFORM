# =============================================================================
# LIMITATION FIX — External Knowledge Agent (Web Search Fallback)
# =============================================================================
# PROBLEM (identified during evaluation):
#   When a question is NOT covered by the local Neo4j graph, the Validator
#   agent correctly assigns LOW CONFIDENCE rather than hallucinating an
#   answer. This is safe behaviour, but the system still cannot answer
#   the question at all.
#
# FIX (agreed with supervisor):
#   Add a 5th agent — the "External Knowledge Agent" — that is triggered
#   ONLY when the Validator's confidence is below a threshold (e.g. 0.4).
#   This agent searches the web for supplementary information, and the
#   Writer agent then combines graph data (if any) + web data into the
#   final answer, clearly labelling which source each part came from.
#
# This plugs directly into your existing orchestration/multi_agent.py
# (Researcher → Reasoner → Validator → [NEW: External Knowledge Agent] → Writer)
# =============================================================================
 
import os
import warnings as _warnings
 
# Suppress the langgraph/langchain_core deprecation warning at its
# source (langchain_core re-registers its own "always" filter during
# import, which overrides standard warnings.filterwarnings/PYTHONWARNINGS
# suppression -- monkey-patching warn() itself during the import is the
# only reliable fix, same approach used in orchestration/multi_agent.py).
_original_warn = _warnings.warn
_warnings.warn = lambda *a, **k: None
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_core.messages import AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
_warnings.warn = _original_warn
 
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
CONFIDENCE_THRESHOLD = 0.6   # below this, trigger web search fallback
# NOTE: the deterministic confidence system (multi_agent.py) only ever
# produces four discrete values: 0.20 (no evidence), 0.35 (evidence but
# unverified), 0.55 (PARTIAL verification), 0.90 (VERIFIED). The
# threshold must sit strictly between 0.55 and 0.90 -- a threshold of
# 0.4 (the original value) let genuinely out-of-dataset questions that
# happened to validate as "PARTIAL" (0.55) skip the web search fallback
# entirely, since 0.55 >= 0.4, leaving the user with no answer at all.
 
 
# ------------------------------------------------------------------
# Extended shared state — adds fields for the new agent
# ------------------------------------------------------------------
class AgentState(TypedDict):
    question:         str
    research_result:  str
    reasoning:        str
    validation:       str
    confidence:       float
    web_search_used:  bool          # NEW — tracks whether fallback fired
    web_search_result: str          # NEW — what the web search found
    final_answer:     str
    cypher_queries:   List[str]
    gnn_used:         bool
    messages:         Annotated[list, "messages"]
 
 
# ------------------------------------------------------------------
# NEW AGENT — External Knowledge Agent
# Triggered only when Validator confidence < CONFIDENCE_THRESHOLD
# ------------------------------------------------------------------
def external_knowledge_agent(state: AgentState) -> AgentState:
    """
    This is the fix for the identified limitation.
    When local graph data is insufficient (low confidence), this agent
    searches the web to supplement the answer instead of the system
    simply returning "low confidence, no answer".
    """
    print(f"[FIX] Confidence {state['confidence']:.2f} is below threshold "
          f"({CONFIDENCE_THRESHOLD}) — triggering external web search...")
 
    search_tool = DuckDuckGoSearchRun()
    llm = OllamaLLM(model=LLM_MODEL, temperature=0)
 
    # Step 1: turn the original question into a good search query.
    # FIX: for questions using relative/current-time language ("latest",
    # "current", "today", "now"), the search query must include the
    # ACTUAL current year explicitly -- otherwise the search engine (and
    # the LLM itself, whose training data may be outdated) resolves
    # "latest" to whatever it last knew about, returning a generic
    # historical results page instead of the specific current answer.
    import datetime
    current_year = datetime.datetime.now().year
    query_prompt = f"""Convert this question into a short, effective web search query.
The current year is {current_year}. If the question uses words like
"latest", "current", "today", "now", or "right now", include the
specific year {current_year} explicitly in the search query, so the
search returns up-to-date results rather than a generic historical
summary page.
Return ONLY the search query, nothing else.
 
Question: {state['question']}
Search query:"""
    search_query = llm.invoke(query_prompt).strip().strip('"')
 
    # Step 2: run the web search
    try:
        web_result = search_tool.run(search_query)
    except Exception as e:
        web_result = f"Web search failed: {e}"
 
    print(f"[FIX] Web search query: '{search_query}'")
    print(f"[FIX] Web search result (first 200 chars): {web_result[:200]}...")
 
    state["web_search_used"]   = True
    state["web_search_result"] = web_result
 
    # Boost confidence when the web search genuinely succeeded (real
    # result text returned, not an error/empty result), since the
    # answer is now grounded in external evidence rather than being an
    # ungrounded "low confidence, no answer" response. Capped at 0.85
    # (not 1.0) because web search results, unlike direct graph
    # queries, are not independently verified against the Neo4j graph
    # -- a deliberately slightly lower ceiling than fully graph-verified
    # answers (which can reach 0.90).
    search_failed = (
        not web_result
        or web_result.startswith("Web search failed")
        or len(web_result.strip()) < 20
    )
    if not search_failed:
        state["confidence"] = 0.85
        print(f"[FIX] Web search succeeded — confidence boosted to {state['confidence']:.2f}")
    else:
        print(f"[FIX] Web search returned no usable result — confidence stays at {state['confidence']:.2f}")
 
    state["messages"].append(
        AIMessage(content=f"[ExternalKnowledgeAgent] Searched web for: {search_query}")
    )
    return state
 
 
# ------------------------------------------------------------------
# UPDATED Writer agent — now combines graph data + web data
# ------------------------------------------------------------------
def writer_agent_with_fallback(state: AgentState) -> AgentState:
    """
    Updated Writer agent: if web search was used, clearly cite both
    sources. If not, behaves exactly as before (graph-only answer).
    """
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.2)
 
    if state.get("web_search_used"):
        # Combine graph + web knowledge
        prompt = f"""You are a writer agent producing a final answer using TWO sources.
 
Question: {state['question']}
 
SOURCE 1 — Local knowledge graph (may be incomplete):
{state['research_result']}
Graph confidence: {state['confidence']:.2f} (low — this is why web search was used)
 
SOURCE 2 — Web search result:
{state['web_search_result']}
 
CRITICAL RULE: Use ONLY facts that are explicitly stated in SOURCE 1 or
SOURCE 2 above. Do NOT fill in the answer using your own training
knowledge, even if you recognise the topic (e.g. sports results, prices,
current events) — your training data may be outdated and your own
"knowledge" of the current answer could be WRONG. If SOURCE 2 is only a
generic description (e.g. a list of past winners, a general topic page)
and does NOT explicitly state the current/specific answer being asked
for, you MUST say the search result did not contain a clear, current
answer — do NOT guess a specific name, number, or date from memory to
fill the gap.
 
IMPORTANT: If SOURCE 2 DOES explicitly contain the specific answer
(e.g. an actual price, date, name, or number), you MUST state that
value clearly and directly in your answer — do not hedge or say
"unable to find" when the value is right there in SOURCE 2.
 
Rules:
- Combine both sources into one clear answer.
- Explicitly state which parts came from the local graph and which came from the web.
- If the web result contradicts or extends the graph result, mention this.
- End with: Source: [Local Graph + Web Search]
- Keep the answer under 180 words.
"""
    else:
        # Original behaviour — graph-only, no fallback needed
        prompt = f"""You are a writer agent. Compose a clear, concise, grounded answer.
 
Question: {state['question']}
Research: {state['research_result']}
Reasoning: {state['reasoning']}
Confidence: {state['confidence']:.2f}
 
Rules:
- Answer must be directly responsive to the question.
- Keep the answer under 150 words.
- End with: Source: [Local Graph]
"""
 
    final = llm.invoke(prompt)
    state["final_answer"] = final
    state["messages"].append(AIMessage(content=f"[Writer] {final}"))
    return state
 
 
# ------------------------------------------------------------------
# UPDATED routing logic
# Decides: re-research OR trigger web search fallback OR go to writer
# ------------------------------------------------------------------
def route_after_validation(state: AgentState) -> str:
    """
    This is the key fix in the workflow routing:
    - confidence >= threshold        -> go straight to writer (as before)
    - confidence < threshold          -> NEW: trigger external knowledge agent
    - (re-research loop still exists for very early low-confidence cases)
    """
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        return "external_knowledge"
    return "write"
 
 
# ------------------------------------------------------------------
# Build the UPDATED orchestration graph (5 agents instead of 4)
# ------------------------------------------------------------------
def build_orchestration_graph_v2() -> StateGraph:
    """
    Updated workflow:
    Researcher -> Reasoner -> Validator -> [conditional] -> Writer
                                              |
                                              v (if low confidence)
                                    External Knowledge Agent -> Writer
    """
    # Import your existing agents from Module 4 (unchanged)
    from orchestration.multi_agent import researcher_agent, reasoner_agent, validator_agent
 
    workflow = StateGraph(AgentState)
 
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("reasoner", reasoner_agent)
    workflow.add_node("validator", validator_agent)
    workflow.add_node("external_knowledge", external_knowledge_agent)   # NEW
    workflow.add_node("writer", writer_agent_with_fallback)             # UPDATED
 
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "reasoner")
    workflow.add_edge("reasoner", "validator")
 
    # KEY FIX: conditional routing after validation
    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "external_knowledge": "external_knowledge",  # NEW path
            "write": "writer"
        }
    )
    workflow.add_edge("external_knowledge", "writer")  # after web search, always write
    workflow.add_edge("writer", END)
 
    return workflow.compile()
 
 
# ------------------------------------------------------------------
# Public interface — same signature as before, drop-in replacement
# ------------------------------------------------------------------
def orchestrated_ask_v2(question: str) -> dict:
    """
    Same as your original orchestrated_ask(), but now automatically
    falls back to web search when local graph confidence is low.
    """
    from langchain_core.messages import HumanMessage
 
    app = build_orchestration_graph_v2()
 
    initial_state = AgentState(
        question=question,
        research_result="",
        reasoning="",
        validation="",
        confidence=0.0,
        web_search_used=False,
        web_search_result="",
        final_answer="",
        cypher_queries=[],
        gnn_used=False,
        messages=[HumanMessage(content=question)]
    )
 
    result = app.invoke(initial_state)
 
    return {
        "question":          question,
        "final_answer":      result["final_answer"],
        "confidence":        result["confidence"],
        "web_search_used":   result["web_search_used"],   # NEW field
        "web_search_result": result.get("web_search_result", ""),
        "gnn_used":          result["gnn_used"],
        "cypher_queries":    result["cypher_queries"],
    }
 
 
# ------------------------------------------------------------------
# TEST — demonstrates the fix on an out-of-dataset question
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("TEST 1 — In-dataset question (should NOT trigger web search)")
    print("=" * 70)
    q1 = "Which entities are most connected in the graph?"
    r1 = orchestrated_ask_v2(q1)
    print(f"\nAnswer: {r1['final_answer']}")
    print(f"Web search used: {r1['web_search_used']}")
 
    print("\n" + "=" * 70)
    print("TEST 2 — Out-of-dataset question (SHOULD trigger web search)")
    print("=" * 70)
    q2 = "What is the current price of Bitcoin today?"
    r2 = orchestrated_ask_v2(q2)
    print(f"\nAnswer: {r2['final_answer']}")
    print(f"Web search used: {r2['web_search_used']}")