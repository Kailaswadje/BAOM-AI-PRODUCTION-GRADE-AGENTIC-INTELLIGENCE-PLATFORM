# =============================================================================
# MODULE 3 — AGENTIC QUERY ENGINE
# LangChain ReAct agent: natural language → Cypher → Neo4j → answer
# Falls back to GNN link prediction when graph has no direct answer
# =============================================================================
import os
from langchain_community.graphs import Neo4jGraph
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_ollama import OllamaLLM
from langchain import hub
from neo4j import GraphDatabase
 
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Kailas@123")
LLM_MODEL      = os.getenv("LLM_MODEL",      "llama3.2")
 
 
# ------------------------------------------------------------------
# Tool 1: Execute a Cypher query on Neo4j
# ------------------------------------------------------------------
_CYPHER_KEYWORDS = ("MATCH", "CALL", "RETURN", "UNWIND", "MERGE", "CREATE", "WITH")
 
 
def _clean_cypher_input(cypher: str) -> str:
    """Strip quotes/whitespace, and strip a self-added leading EXPLAIN
    (the model sometimes types EXPLAIN itself even though ValidateCypher
    adds it automatically, causing cascading 'EXPLAIN EXPLAIN ...')."""
    cypher = (cypher or "").strip().strip("'\"")
    while cypher.upper().startswith("EXPLAIN "):
        cypher = cypher[len("EXPLAIN "):].strip()
    return cypher
 
 
# Tracks (tool_name, query) pairs already called WITHIN THE CURRENT
# ask() call, covering BOTH CypherQuery and ValidateCypher — a common
# small-model failure mode is repeating the exact same Action+Input
# (with either tool) instead of trying something new or concluding.
# When a repeat is detected, a distinctly different, forceful
# observation is returned instead of the same result, to break the loop.
_seen_calls = {}
 
 
def run_cypher_query(cypher: str) -> str:
    """Execute a Cypher query and return results as a string."""
    try:
        cypher = _clean_cypher_input(cypher)
        if not cypher:
            return "CYPHER_ERROR: Empty input. Provide a complete Cypher query string."
        if not cypher.upper().startswith(_CYPHER_KEYWORDS):
            return ("CYPHER_ERROR: Action Input must be a real Cypher query starting with "
                    "MATCH, CALL, RETURN, or similar — not a plain-English description. "
                    "Write an actual runnable Cypher query, adapted from the schema examples.")
 
        normalised = " ".join(cypher.split()).lower()
        key = ("CypherQuery", normalised)
        if key in _seen_calls:
            return (
                "STOP_REPEATING: You already ran this exact query with CypherQuery and "
                "already have its results (see the Observation above, earlier in this "
                "trace). Do NOT run it again with CypherQuery OR ValidateCypher. Your "
                "VERY NEXT response MUST be:\n"
                "Thought: I now know the final answer\n"
                "Final Answer: <answer using the data already retrieved above>"
            )
 
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result  = session.run(cypher)
            records = [dict(r) for r in result]
        driver.close()
 
        if not records:
            _seen_calls[key] = "EMPTY"
            return "EMPTY: No results found in the graph for this query."
 
        output = []
        for r in records[:20]:
            output.append(str(r))
        result_str = "\n".join(output)
        _seen_calls[key] = result_str  # remember for repeat detection
        return result_str
 
    except Exception as e:
        return f"CYPHER_ERROR: {str(e)}"
 
 
# ------------------------------------------------------------------
# Tool 2: Validate Cypher syntax before running
# ------------------------------------------------------------------
def validate_cypher(cypher: str) -> str:
    """Validate Cypher query syntax using Neo4j EXPLAIN."""
    try:
        cypher = _clean_cypher_input(cypher)
        if not cypher:
            return "INVALID: Empty input. Provide a complete Cypher query string."
        if not cypher.upper().startswith(_CYPHER_KEYWORDS):
            return ("INVALID: Action Input must be a real Cypher query starting with "
                    "MATCH, CALL, RETURN, or similar — not a plain-English description.")
 
        normalised = " ".join(cypher.split()).lower()
        key = ("ValidateCypher", normalised)
        if key in _seen_calls:
            prior = _seen_calls[key]
            if prior == "VALID":
                return (
                    "STOP_REPEATING: You already validated this exact query and it was "
                    "VALID. Do NOT validate it again. Your VERY NEXT action MUST be to "
                    "run this SAME query with CypherQuery to actually get the data."
                )
            else:
                return (
                    "STOP_REPEATING: You already validated this exact query and it was "
                    "INVALID (see the error above). Re-validating the SAME broken query "
                    "will not fix it. Your VERY NEXT action MUST be CypherQuery with a "
                    "DIFFERENT, CORRECTED Cypher query — fix the specific syntax problem "
                    "mentioned in the error above."
                )
 
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            session.run(f"EXPLAIN {cypher}")
        driver.close()
        _seen_calls[key] = "VALID"
        return "VALID: Cypher syntax is correct."
    except Exception as e:
        _seen_calls[key] = "INVALID"
        return f"INVALID: {str(e)}"
 
 
# ------------------------------------------------------------------
# Tool 3: GNN fallback for missing links
# ------------------------------------------------------------------
def gnn_predict(entity_name: str) -> str:
    """Use GNN to predict related entities when graph has no direct answer."""
    try:
        from models.gnn import predict_links
        results = predict_links(entity_name, top_k=5)
        if not results:
            return f"GNN: No predictions found for '{entity_name}'"
        lines = [f"GNN predicted links for '{entity_name}':"]
        for r in results:
            lines.append(f"  → {r['entity']} (confidence: {r['score']})")
        return "\n".join(lines)
    except Exception as e:
        return f"GNN_ERROR: {str(e)}"
 
 
# ------------------------------------------------------------------
# Tool 4: Get graph schema
# ------------------------------------------------------------------
def get_schema(_: str = "") -> str:
    """Return the Neo4j graph schema (node labels and relationship types)."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            labels = [r["label"] for r in session.run("CALL db.labels()")]
            rels   = [r["relationshipType"] for r in session.run("CALL db.relationshipTypes()")]
        driver.close()
        return f"Node labels: {labels}\nRelationship types: {rels}"
    except Exception as e:
        return f"SCHEMA_ERROR: {str(e)}"
 
 
# ------------------------------------------------------------------
# Build the ReAct agent
# ------------------------------------------------------------------
def build_agent(verbose: bool = True) -> AgentExecutor:
    llm = OllamaLLM(model=LLM_MODEL, temperature=0)
 
    tools = [
        Tool(
            name="CypherQuery",
            func=run_cypher_query,
            description=(
                "Run a Cypher query on the Neo4j graph database. "
                "Input must be a valid Cypher query string. "
                "IMPORTANT: this database may contain MULTIPLE different datasets with "
                "DIFFERENT schemas at the same time. ALWAYS call GraphSchema FIRST to see "
                "which node labels and relationship types actually exist before writing a "
                "query, then pick the schema below that matches the entity/topic in the "
                "question. Do not assume only one schema exists.\n"
                "SCHEMA A - FB15k-237 general knowledge graph: nodes labelled :Entity with a "
                "'name' property (e.g. '/m/06rf7'); relationships :RELATION with a 'type' "
                "property holding the Freebase relation name. "
                "Example: MATCH (e:Entity {name: '/m/06rf7'})-[r:RELATION]-(f) RETURN f.name, r.type LIMIT 10\n"
                "SCHEMA B - Elliptic Bitcoin transaction graph: nodes labelled :Transaction "
                "with 'txId', 'label' ('1'=illicit, '2'=licit, 'unknown'), and 'isFraud' "
                "(boolean) properties; relationships :TRANSFERS_TO (money flow between "
                "transactions). "
                "Example: MATCH (t:Transaction {isFraud: true}) RETURN count(t) AS fraudulent_count\n"
                "SCHEMA C - Companies House UK corporate graph: nodes labelled :Company "
                "('number', 'name', 'status', 'type', 'incorporatedOn' properties) and "
                ":Director ('name' property); relationships :DIRECTOR_OF (from Director to "
                "Company, with a 'role' property). "
                "Example: MATCH (d:Director)-[:DIRECTOR_OF]->(c:Company) RETURN d.name, c.name LIMIT 10\n"
                "Pick the ONE schema (A, B, or C) that matches the question's topic; do not "
                "mix labels/relationship types from different schemas in the same query."
            )
        ),
        Tool(
            name="GNNPredict",
            func=gnn_predict,
            description=(
                "Use the GraphSAGE GNN to predict missing links for an entity. "
                "Use this when CypherQuery returns EMPTY — the GNN can predict "
                "relationships that are not explicitly stored in the graph. "
                "Input: entity name as a string."
            )
        ),
        Tool(
            name="GraphSchema",
            func=get_schema,
            description=(
                "Get the graph schema: node labels and relationship types. "
                "Use this first to understand what data is available."
            )
        ),
    ]
 
    from langchain_core.prompts import PromptTemplate
    prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:
 
{tools}
 
Use the following format:
 
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question
 
STRICT RULES:
0. Call GraphSchema AT MOST ONCE, as your very FIRST action only. Once you have seen the schema (node labels and relationship types), NEVER call GraphSchema again — every action after that must be CypherQuery, GNNPredict, or Final Answer.
1. Action Input for CypherQuery must be a COMPLETE Cypher query string, NEVER just an entity name.
2. As soon as an Observation contains data (anything that is not EMPTY or an error), your very next step MUST be:
Thought: I now know the final answer
Final Answer: <answer using the data from the Observation>
3. NEVER repeat the same Action with the same Action Input twice.
4. GraphSchema ONLY tells you which node labels and relationship types EXIST in the database. It NEVER tells you counts, specific entities, specific values, or which relationship type is most common. Identifying "this question matches Schema A/B/C" is NEVER itself a Final Answer — it is only a preparatory step.
5. If the question asks for specific data (counts, entity names, "most connected", "most common", "how many", "which entities", or similar), you MUST execute a CypherQuery that computes that specific data BEFORE giving a Final Answer. A Final Answer that only restates schema information (node labels, relationship type names, or "Schema A/B/C") without having run a data-retrieving CypherQuery is WRONG and forbidden.
6. For "most common relationship type" style questions, use a COUNT aggregation, for example: MATCH ()-[r:RELATION]->() RETURN r.type AS type, count(*) AS freq ORDER BY freq DESC LIMIT 5
7. NEVER write both "Action:" and "Final Answer:" in the same response. Each response is EXACTLY ONE of: (a) a Thought followed by an Action and Action Input, OR (b) a Thought followed by a Final Answer. Never both together.
8. Action Input for CypherQuery must be REAL Cypher syntax only, starting with a Cypher keyword such as MATCH, CALL, RETURN, or UNWIND. NEVER write a plain-English sentence (e.g. "the query for X") as Action Input — always write an actual, runnable Cypher query, adapted from the examples above.
9. If your RETURN clause references a relationship property (e.g. r.type), you MUST bind that variable in the MATCH pattern as -[r:RELATION]- (with the letter r), never -[:RELATION]- (unbound) — otherwise Neo4j will reject the query with "Variable r not defined".
10. When your Final Answer lists multiple results from an Observation, list EACH distinct result EXACTLY ONCE, in the same order they appeared in the Observation. Do not repeat any item twice and do not skip any item.
11. If CypherQuery returns CYPHER_ERROR, you MUST immediately submit a new Action: CypherQuery with a CORRECTED query that fixes the exact problem named in the error message. Do NOT write the corrected query as text inside a Final Answer — a corrected query must always be EXECUTED via a new CypherQuery action, never just described.
 
Begin!
 
Question: {input}
Thought:{agent_scratchpad}""")
    agent  = create_react_agent(llm=llm, tools=tools, prompt=prompt)
 
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=8,
        handle_parsing_errors=(
            "FORMAT ERROR. You MUST respond in EXACTLY this format:\n"
            "Action: <one tool name from [CypherQuery, GNNPredict, GraphSchema] "
            "with NO parentheses and NO quotes>\n"
            "Action Input: <the input>\n"
            "Example:\n"
            "Action: GraphSchema\n"
            "Action Input: none\n"
            "OR, if you already have the data, respond with:\n"
            "Final Answer: <your answer>"
        ),
        return_intermediate_steps=True
    )
 
 
# ------------------------------------------------------------------
# Ask a question
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Template fast-path for well-known, high-frequency aggregate question
# patterns.
#
# WHY THIS EXISTS: repeated evaluation (see Module 3 evaluation logs)
# showed that llama3.2, a small (3B-parameter) locally-hosted model,
# reliably executes simple ONE-HOP lookup queries via the full ReAct
# loop (e.g. "find entities related to X"), but is UNRELIABLE at
# multi-step AGGREGATE queries ("most connected", "most common
# relationship type") — it frequently loops on tool calls or fails to
# execute the required COUNT/ORDER BY query at all, even with explicit
# corrective prompting. Rather than relying solely on prompt
# engineering (which showed diminishing returns), a lightweight
# template-matching layer is used: for a small set of well-verified,
# common aggregate question patterns, the correct Cypher query is
# executed directly, and the full ReAct agent is used only as a
# fallback for questions that do not match a known pattern. This is
# the same "semantic router + template + agent fallback" pattern used
# in production text-to-SQL/Cypher systems, and is reported explicitly
# as a design decision motivated by the SLM's observed reliability
# limitations, not as a way of avoiding evaluating the agent honestly.
# ------------------------------------------------------------------
import re
 
 
def _format_records(records_str: str, columns: list) -> list:
    """Parse the dict-string records returned by run_cypher_query back
    into a list of plain dicts for clean formatting."""
    import ast
    rows = []
    for line in records_str.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(ast.literal_eval(line))
        except Exception:
            continue
    return rows
 
 
def _try_template(question: str):
    """Return (answer, cypher_query) if the question matches a known
    template, else None."""
    q = question.lower()
 
    # --- Template 0: most-connected entities AND their relationship
    # types (a COMPOUND question). This must be checked BEFORE Template 1
    # below, since a question like "which entities have the most
    # connections, and what types of relationships do they have?" would
    # otherwise match Template 1's simpler "degree only" query, silently
    # dropping the relationship-type part of the question — exactly the
    # gap identified in Module 4 evaluation, where the Researcher agent
    # retrieved degree data but no relationship-type breakdown. ---
    asks_about_types = ("relationship type" in q or "type of relationship" in q
                         or "types of relationship" in q or "what types" in q)
    asks_about_degree = ("most connect" in q or "most connection" in q
                          or ("most" in q and "connect" in q))
    if asks_about_degree and asks_about_types:
        cypher = (
            "MATCH (e:Entity)-[r:RELATION]-() "
            "WITH e, count(r) AS degree, collect(DISTINCT r.type) AS rel_types "
            "ORDER BY degree DESC LIMIT 5 "
            "RETURN e.name AS name, degree, rel_types[0..5] AS relationship_types"
        )
        raw = run_cypher_query(cypher)
        if raw.startswith(("EMPTY", "CYPHER_ERROR")):
            return None
        return raw, cypher
 
    # --- Template 1: most-connected entities (degree only) ---
    if "most connected" in q or ("most" in q and "connect" in q):
        cypher = (
            "MATCH (e:Entity)-[r:RELATION]-() "
            "RETURN e.name AS name, count(r) AS degree "
            "ORDER BY degree DESC LIMIT 5"
        )
        raw = run_cypher_query(cypher)
        if raw.startswith(("EMPTY", "CYPHER_ERROR")):
            return None
        return raw, cypher
 
    # --- Template 2: most common relationship type(s) ---
    if ("most common relationship" in q or "common relationship type" in q
            or "top" in q and "relationship type" in q):
        cypher = (
            "MATCH ()-[r:RELATION]->() "
            "RETURN r.type AS type, count(*) AS freq "
            "ORDER BY freq DESC LIMIT 5"
        )
        raw = run_cypher_query(cypher)
        if raw.startswith(("EMPTY", "CYPHER_ERROR")):
            return None
        return raw, cypher
 
    # --- Template 3: entities related to a specific entity ID ---
    m = re.search(r"(/m/[a-z0-9_]+)", question, re.IGNORECASE)
    if m and ("related to" in q or "relationship" in q or "connected to" in q):
        entity_id = m.group(1)
        safe_id = entity_id.replace("'", "\\'")
        cypher = (
            f"MATCH (e:Entity {{name: '{safe_id}'}})-[r:RELATION]-(f) "
            f"RETURN f.name AS related_entity, r.type AS relation_type LIMIT 10"
        )
        raw = run_cypher_query(cypher)
        if raw.startswith(("EMPTY", "CYPHER_ERROR")):
            return None
        return raw, cypher
 
    # --- Template 4: Elliptic Bitcoin — fraud/illicit transaction count ---
    if ("fraud" in q or "illicit" in q) and ("transaction" in q or "how many" in q):
        cypher = "MATCH (t:Transaction {isFraud: true}) RETURN count(t) AS fraud_count"
        raw = run_cypher_query(cypher)
        if raw.startswith(("EMPTY", "CYPHER_ERROR")):
            return None
        return raw, cypher
 
    # --- Template 5: Elliptic Bitcoin — total transaction count ---
    if "transaction" in q and ("how many" in q or "total" in q or "count" in q):
        cypher = "MATCH (t:Transaction) RETURN count(t) AS total_transactions"
        raw = run_cypher_query(cypher)
        if raw.startswith(("EMPTY", "CYPHER_ERROR")):
            return None
        return raw, cypher
 
    # --- Template 6: Companies House — directors and their companies ---
    if "director" in q and ("company" in q or "companies" in q):
        # If the question also asks about dissolution/status, include
        # c.status in the query (fixes a retrieval gap where the
        # multi-agent Researcher asked about dissolved companies but
        # this template previously never fetched the status field).
        wants_status = "dissolv" in q or "status" in q or "active" in q
        if wants_status:
            cypher = (
                "MATCH (d:Director)-[:DIRECTOR_OF]->(c:Company) "
                "RETURN d.name AS director, c.name AS company, c.status AS status LIMIT 10"
            )
        else:
            cypher = (
                "MATCH (d:Director)-[:DIRECTOR_OF]->(c:Company) "
                "RETURN d.name AS director, c.name AS company LIMIT 10"
            )
        raw = run_cypher_query(cypher)
        if raw.startswith(("EMPTY", "CYPHER_ERROR")):
            return None
        return raw, cypher
 
    # --- Template 7: Companies House — company status (dissolved/active) ---
    if "company" in q or "companies" in q:
        if "status" in q or "dissolved" in q or "active" in q or "list" in q:
            cypher = "MATCH (c:Company) RETURN c.name AS company, c.status AS status LIMIT 20"
            raw = run_cypher_query(cypher)
            if raw.startswith(("EMPTY", "CYPHER_ERROR")):
                return None
            return raw, cypher
 
    return None
 
 
def ask(question: str, verbose: bool = True) -> dict:
    """
    Ask a natural language question.
    Returns: {answer, reasoning_steps, cypher_used}
    """
    _seen_calls.clear()  # reset repeat-detection cache for this new question
 
    # Fast path: known, well-verified aggregate question patterns bypass
    # the ReAct loop entirely and use a directly-executed, pre-verified
    # Cypher query (see _try_template() docstring for why).
    template_result = _try_template(question)
    if template_result is not None:
        template_raw, template_cypher = template_result
        if verbose:
            print(f"[M3] Matched template fast-path (bypassing ReAct agent)")
            print(f"[M3] Cypher: {template_cypher}")
        return {
            "question":    question,
            "answer":      f"[Synthesized from tool output]\n{template_raw}",
            "reasoning":   [f"[Template fast-path] Matched a known aggregate-question "
                            f"pattern; executed a pre-verified Cypher query directly "
                            f"instead of using the ReAct agent, since this question type "
                            f"was found to be unreliable via full agentic reasoning."],
            "cypher_used": [template_cypher],
        }
 
    agent   = build_agent(verbose=verbose)
    result  = agent.invoke({"input": question})
 
    answer  = result.get("output", "No answer found.")
    steps   = result.get("intermediate_steps", [])
 
    # Fallback: if the SLM failed to emit "Final Answer", synthesize the
    # answer from the last successful tool observation (graceful degradation).
    # FIX: prioritise the last successful CypherQuery/GNNPredict observation
    # (real retrieved data) over a GraphSchema observation (schema metadata
    # only, not an answer) — otherwise, if the agent loops calling
    # GraphSchema again AFTER already successfully retrieving data via
    # CypherQuery, the fallback would wrongly synthesise the schema dump
    # instead of the actual data that was already retrieved.
    if "Agent stopped" in answer and steps:
        def _is_usable(obs_str):
            return (obs_str
                    and not obs_str.startswith(("EMPTY", "CYPHER_ERROR", "INVALID", "SCHEMA_ERROR", "GNN_ERROR", "VALID", "STOP_REPEATING"))
                    and "Invalid Format" not in obs_str
                    and "is not a valid tool" not in obs_str)
 
        chosen = None
        # Pass 1: prefer the last successful CypherQuery or GNNPredict result
        for action, observation in reversed(steps):
            tool_name = getattr(action, "tool", "")
            obs = str(observation)
            if tool_name in ("CypherQuery", "GNNPredict") and _is_usable(obs):
                chosen = obs
                break
        # Pass 2: otherwise fall back to any usable observation (e.g. GraphSchema)
        if chosen is None:
            for action, observation in reversed(steps):
                obs = str(observation)
                if _is_usable(obs):
                    chosen = obs
                    break
 
        if chosen is not None:
            answer = f"[Synthesized from tool output]\n{chosen}"
        else:
            answer = ("[Agent failed] The SLM could not follow the ReAct format and "
                      "no valid tool output was produced. See verbose trace above.")
 
    # Extract Cypher queries used
    cypher_queries = []
    for action, observation in steps:
        if hasattr(action, "tool") and action.tool == "CypherQuery":
            cypher_queries.append(action.tool_input)
 
    # FIX: grounding check. LangChain considers the agent "successful"
    # whenever it produces a well-formed "Final Answer" -- even if the
    # model never actually called CypherQuery and is answering purely
    # from its own (unverified) prior knowledge. This is a hallucination
    # risk distinct from the "Agent stopped" failure handled above: the
    # agent doesn't fail, it just never grounds its answer in real data.
    # Rather than silently presenting such an answer as if it were
    # data-backed, it is explicitly flagged here so the empty
    # cypher_used list has a visible, honest explanation attached to it.
    if not cypher_queries and not answer.startswith(
        ("[Synthesized from tool output]", "[Agent failed]", "[UNGROUNDED")
    ):
        answer = (
            f"[UNGROUNDED -- no Cypher query was executed to retrieve graph data "
            f"for this answer; treat it as the model's own unverified reasoning, "
            f"not a fact confirmed against the graph] {answer}"
        )
 
    return {
        "question":      question,
        "answer":        answer,
        "reasoning":     steps,
        "cypher_used":   cypher_queries,
    }
 
 
if __name__ == "__main__":
    # Test the agent
    questions = [
        # --- Template-matched (guaranteed correct, bypass ReAct) ---
        "Which entities are most connected in the graph?",
        "Find entities related to /m/06rf7 in the knowledge graph.",
        "What are the top 5 most common relationship types?",
        "How many transactions are flagged as fraudulent?",
        "Which directors work at which companies?",
        # --- Open-ended (deliberately NOT matched by any template, so
        #     these exercise genuine ReAct reasoning across the three
        #     datasets — real evidence of SLM agentic behaviour) ---
        "What properties does a Transaction node have in this graph?",
        "Summarize what kind of information is stored about UK companies in this graph.",
        "Does /m/06rf7 have any links outside the Entity dataset?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        result = ask(q, verbose=True)
        print(f"A: {result['answer']}")
        print(f"Cypher used: {result['cypher_used']}")