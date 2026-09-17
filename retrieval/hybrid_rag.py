# =============================================================================
# MODULE 5 — HYBRID RAG PIPELINE
# Microsoft GraphRAG: text → knowledge graph extraction
# HippoRAG: cheap multi-hop retrieval via Personalised PageRank
# Arize Phoenix: full observability of every LLM/agent call
# =============================================================================
 
import os
import json
from dotenv import load_dotenv
 
load_dotenv()  # FIX: was missing — .env variables (including
                # COMPANIES_HOUSE_API_KEY) were never being loaded,
                # so the live API fetch always skipped itself even
                # when the key was correctly set in .env.
import networkx as nx
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
import phoenix as px
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from openinference.instrumentation.langchain import LangChainInstrumentor
 
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
 
 
# ------------------------------------------------------------------
# 5a. Arize Phoenix observability setup
# Must be called once at startup
# ------------------------------------------------------------------
def setup_observability():
    """Initialise Arize Phoenix tracing for all LLM and agent calls."""
    session = px.launch_app()
    print(f"[M5] Arize Phoenix dashboard: {session.url}")
 
    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(px.tracer_provider().get_tracer("agentic_platform"))
    )
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument()
    print("[M5] Observability active — all LLM calls are now traced")
    return session
 
 
# ------------------------------------------------------------------
# 5b. GraphRAG: extract knowledge graph from text
# Simplified implementation of Microsoft GraphRAG pipeline
# ------------------------------------------------------------------
class GraphRAGExtractor:
    """
    Extract entities and relationships from raw text.
    Simplified version of Microsoft GraphRAG (Edge et al., 2024).
    """
 
    def __init__(self):
        self.llm      = OllamaLLM(model=LLM_MODEL, temperature=0)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        self.graph    = nx.DiGraph()
 
    def extract_from_text(self, text: str) -> nx.DiGraph:
        """Extract entity-relationship triples from text into a NetworkX graph."""
        chunks  = self.splitter.split_text(text)
        print(f"[M5] GraphRAG: processing {len(chunks)} text chunks...")
 
        for i, chunk in enumerate(chunks):
            prompt = f"""Extract entities and relationships from this text.
Return ONLY JSON in this exact format:
{{"triples": [{{"head": "entity1", "relation": "RELATION", "tail": "entity2"}}]}}
 
Text: {chunk}
 
JSON output:"""
            try:
                response = self.llm.invoke(prompt)
                # Find JSON block in response
                start = response.find("{")
                end   = response.rfind("}") + 1
                if start == -1 or end == 0:
                    continue
                data = json.loads(response[start:end])
                for triple in data.get("triples", []):
                    h = triple.get("head", "").strip()
                    r = triple.get("relation", "RELATED_TO").strip()
                    t = triple.get("tail", "").strip()
                    if h and t:
                        self.graph.add_edge(h, t, relation=r)
            except (json.JSONDecodeError, Exception):
                continue  # Skip malformed chunks
 
            if (i + 1) % 10 == 0:
                print(f"[M5] GraphRAG: {i+1}/{len(chunks)} chunks done, "
                      f"{self.graph.number_of_nodes()} nodes extracted")
 
        print(f"[M5] GraphRAG complete: {self.graph.number_of_nodes()} entities, "
              f"{self.graph.number_of_edges()} relations")
        return self.graph
 
    def extract_from_file(self, path: str) -> nx.DiGraph:
        with open(path, "r", encoding="utf-8") as f:
            return self.extract_from_text(f.read())
 
    def get_community_summaries(self) -> List[Dict]:
        """Generate community-level summaries (GraphRAG global context)."""
        communities = list(nx.weakly_connected_components(self.graph))
        summaries   = []
        for i, community in enumerate(communities[:10]):  # Top 10 communities
            nodes   = list(community)[:20]
            subgraph = self.graph.subgraph(nodes)
            edges   = [(u, v, d.get("relation", "?"))
                       for u, v, d in subgraph.edges(data=True)]
            summary = {
                "community_id": i,
                "entities":     nodes,
                "relations":    edges,
                "size":         len(community)
            }
            summaries.append(summary)
        return summaries
 
 
# ------------------------------------------------------------------
# 5c. HippoRAG: Personalised PageRank retrieval
# Inspired by Gutierrez et al. (NeurIPS 2024)
# ------------------------------------------------------------------
class HippoRAG:
    """
    Brain-inspired multi-hop retrieval using Personalised PageRank.
    10-30x cheaper than GraphRAG for retrieval.
    """
 
    def __init__(self, graph: nx.DiGraph = None):
        self.graph  = graph or nx.DiGraph()
        self.llm    = OllamaLLM(model=LLM_MODEL, temperature=0)
 
    def add_documents(self, documents: List[str]):
        """Build concept graph from documents (parahippocampal encoding)."""
        for doc in documents:
            # Extract key concepts as nodes
            prompt = f"""List the 5 most important concepts or entities in this text.
Return ONLY a JSON array of strings: ["concept1", "concept2", ...]
 
Text: {doc[:500]}
 
JSON:"""
            try:
                response  = self.llm.invoke(prompt)
                start     = response.find("[")
                end       = response.rfind("]") + 1
                concepts  = json.loads(response[start:end])
                # Connect concepts within same document (co-occurrence graph)
                for i in range(len(concepts)):
                    for j in range(i + 1, len(concepts)):
                        self.graph.add_edge(concepts[i], concepts[j], weight=1.0)
            except Exception:
                continue
 
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve related concepts using Personalised PageRank.
        The query seeds the PageRank — connected nodes are ranked by relevance.
        """
        if self.graph.number_of_nodes() == 0:
            return []
 
        # Find seed nodes most related to query
        prompt = f"""List the 3 most relevant entities from this query for a knowledge graph search.
Return ONLY a JSON array: ["entity1", "entity2", "entity3"]
 
Query: {query}
JSON:"""
        try:
            response     = self.llm.invoke(prompt)
            start        = response.find("[")
            end          = response.rfind("]") + 1
            seed_entities = json.loads(response[start:end])
        except Exception:
            seed_entities = []
 
        # Build personalisation dict for PPR
        personalisation = {}
        graph_nodes     = list(self.graph.nodes())
        for seed in seed_entities:
            # Fuzzy match seed to graph nodes
            for node in graph_nodes:
                if seed.lower() in node.lower() or node.lower() in seed.lower():
                    personalisation[node] = 1.0
 
        if not personalisation:
            # Fall back to uniform PageRank
            personalisation = None
 
        try:
            pr_scores = nx.pagerank(
                self.graph,
                alpha=0.85,
                personalization=personalisation,
                max_iter=100
            )
        except nx.NetworkXError:
            return []
 
        # Return top-k nodes by PPR score
        sorted_nodes = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for node, score in sorted_nodes[:top_k]:
            neighbours = list(self.graph.successors(node))[:5]
            results.append({
                "entity":      node,
                "ppr_score":   round(score, 6),
                "related_to":  neighbours
            })
 
        return results
 
    def multi_hop_retrieve(self, query: str, hops: int = 2, top_k: int = 5) -> List[Dict]:
        """Extend retrieval across multiple hops."""
        initial    = self.retrieve(query, top_k=top_k)
        all_results = list(initial)
        seen       = {r["entity"] for r in initial}
 
        for _ in range(hops - 1):
            new_queries = [r["entity"] for r in all_results[-top_k:]]
            for nq in new_queries:
                hop_results = self.retrieve(nq, top_k=3)
                for r in hop_results:
                    if r["entity"] not in seen:
                        all_results.append(r)
                        seen.add(r["entity"])
 
        return all_results[:top_k * hops]
 
 
# ------------------------------------------------------------------
# 5d. RAGAS evaluation
# ------------------------------------------------------------------
def evaluate_ragas(questions: List[str], answers: List[str],
                   contexts: List[List[str]], ground_truths: List[str]) -> Dict:
    """Evaluate RAG pipeline using RAGAS metrics."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
 
        data = Dataset.from_dict({
            "question":      questions,
            "answer":        answers,
            "contexts":      contexts,
            "ground_truth":  ground_truths,
        })
        result = evaluate(data, metrics=[faithfulness, answer_relevancy])
        print(f"[M5] RAGAS faithfulness: {result['faithfulness']:.4f}")
        print(f"[M5] RAGAS answer relevancy: {result['answer_relevancy']:.4f}")
        return dict(result)
    except ImportError:
        print("[M5] RAGAS not installed: pip install ragas")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}
 
 
# ------------------------------------------------------------------
# 5e. Combined hybrid RAG pipeline
# ------------------------------------------------------------------
class HybridRAGPipeline:
    """GraphRAG + HippoRAG working together."""
 
    def __init__(self):
        self.extractor = GraphRAGExtractor()
        self.hipporag  = HippoRAG()
 
    def ingest(self, text_path: str):
        """Ingest a text document through both pipelines."""
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
 
        print("[M5] Running GraphRAG extraction...")
        self.extractor.extract_from_text(text)
 
        print("[M5] Building HippoRAG concept graph...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks   = splitter.split_text(text)
        self.hipporag.add_documents(chunks)
 
        # Copy GraphRAG graph into HippoRAG for richer retrieval
        for u, v, d in self.extractor.graph.edges(data=True):
            self.hipporag.graph.add_edge(u, v, **d)
 
        print("[M5] Hybrid RAG pipeline ready")
 
    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """Retrieve using HippoRAG PPR (cheaper) then enrich with GraphRAG."""
        hippo_results = self.hipporag.multi_hop_retrieve(query, hops=2, top_k=top_k)
        communities   = self.extractor.get_community_summaries()
 
        return {
            "query":        query,
            "local_results":  hippo_results,
            "global_context": communities[:3],
        }
 
 
# Company numbers used for the LIVE Companies House REST API demonstration
# below. Replace with the same numbers used in loader.py's
# load_companies_house() call for full consistency with the loaded graph.
_COMPANIES_HOUSE_SAMPLE_NUMBERS = ["00445790", "00102498", "00214436"]
 
 
def _fetch_companies_house_live() -> list:
    """
    Fetch company + officer (director) data DIRECTLY from the live
    Companies House REST API (not from Neo4j), using HTTP Basic Auth
    with the API key as username and a blank password -- the same
    authentication pattern used in loader.py's load_companies_house().
    Returns a list of natural-language sentences built from the live
    API response.
    """
    import requests
 
    api_key = os.getenv("COMPANIES_HOUSE_API_KEY", "")
    if not api_key:
        print("[M5] COMPANIES_HOUSE_API_KEY not set; skipping live API fetch.")
        return []
 
    base_url = "https://api.company-information.service.gov.uk"
    auth     = (api_key, "")
    sentences = []
 
    for number in _COMPANIES_HOUSE_SAMPLE_NUMBERS:
        try:
            r = requests.get(f"{base_url}/company/{number}", auth=auth, timeout=10)
            if r.status_code != 200:
                print(f"[M5] Companies House API: {number} returned {r.status_code}, skipping")
                continue
            data = r.json()
            name   = data.get("company_name", number)
            status = data.get("company_status", "unknown")
            inc    = data.get("date_of_creation", "an unknown date")
            sentences.append(
                f"{name} (company number {number}) has status {status} and "
                f"was incorporated on {inc}."
            )
 
            r2 = requests.get(f"{base_url}/company/{number}/officers", auth=auth, timeout=10)
            if r2.status_code == 200:
                officers = r2.json().get("items", [])
                for officer in officers[:3]:
                    officer_name = officer.get("name", "an officer")
                    role         = officer.get("officer_role", "director")
                    sentences.append(f"{officer_name} is a {role} of {name}.")
        except Exception as e:
            print(f"[M5] Companies House API error for {number}: {e}")
            continue
 
    if sentences:
        print(f"[M5] Fetched {len(sentences)} sentences live from the Companies House REST API.")
    return sentences
 
 
def generate_real_text_from_graph() -> str:
    """
    Build ingestion text from all THREE loaded datasets, using the
    appropriate source for each:
      - FB15k-237       : read from Neo4j (already loaded there)
      - Elliptic Bitcoin: read from Neo4j (already loaded there)
      - Companies House : fetched LIVE from the Companies House REST API
                           (not from Neo4j), demonstrating the platform
                           can ingest directly from an external API as
                           well as from the graph database.
    Falls back to a fictional example only if every source is unavailable.
    """
    NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Kailas@123")
 
    sentences = []
 
    # --- Source 1: FB15k-237, from Neo4j ---
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            top_entities = list(session.run(
                "MATCH (e:Entity)-[r:RELATION]-() "
                "RETURN e.name AS name, count(r) AS degree "
                "ORDER BY degree DESC LIMIT 3"
            ))
            for row in top_entities:
                sentences.append(
                    f"Entity {row['name']} has {row['degree']} connections "
                    f"in the general knowledge graph."
                )
 
            top_types = list(session.run(
                "MATCH ()-[r:RELATION]->() "
                "RETURN r.type AS type, count(*) AS freq "
                "ORDER BY freq DESC LIMIT 3"
            ))
            for row in top_types:
                sentences.append(
                    f"The relationship type '{row['type']}' occurs {row['freq']} "
                    f"times in the knowledge graph."
                )
 
            # --- Source 2: Elliptic Bitcoin, from Neo4j ---
            stats = session.run(
                "MATCH (t:Transaction) "
                "RETURN count(t) AS total, "
                "sum(CASE WHEN t.isFraud THEN 1 ELSE 0 END) AS fraud"
            ).single()
            if stats and stats["total"]:
                sentences.append(
                    f"The transaction graph contains {stats['total']} transactions, "
                    f"of which {stats['fraud']} are flagged as fraudulent."
                )
 
            director_rows = list(session.run(
                "MATCH (d:Director)-[:DIRECTOR_OF]->(c1:Company), "
                "      (d)-[:DIRECTOR_OF]->(c2:Company) "
                "WHERE c1.number < c2.number "
                "RETURN d.name AS director, c1.name AS company1, c2.name AS company2 "
                "LIMIT 5"
            ))
            for row in director_rows:
                sentences.append(
                    f"{row['company1']} and {row['company2']} share a director, "
                    f"{row['director']}, according to the graph database."
                )
        driver.close()
    except Exception as e:
        print(f"[M5] Could not read FB15k-237/Elliptic data from Neo4j ({e}).")
 
    # --- Source 3: Companies House, LIVE from the REST API ---
    sentences.extend(_fetch_companies_house_live())
 
    if sentences:
        text = " ".join(sentences)
        print(f"[M5] Generated {len(sentences)} real sentences from all 3 sources "
              f"(FB15k-237 + Elliptic via Neo4j, Companies House via live API).")
        return text
 
    # Fallback: original fictional example (used only if every source fails)
    print("[M5] All real data sources unavailable; using fictional fallback text.")
    return """
    The company XYZ Corp reported quarterly losses of £2.3M.
    Director John Smith also serves on the board of ABC Ltd.
    ABC Ltd was flagged for late filing in 2024.
    XYZ Corp and ABC Ltd share the same registered address.
    """
 
 
if __name__ == "__main__":
    # setup_observability()
    pipeline = HybridRAGPipeline()
 
    # Use REAL data from Companies House UK + Elliptic Bitcoin (Neo4j),
    # falling back to a fictional example only if unavailable.
    real_text = generate_real_text_from_graph()
 
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(real_text)
        tmp = f.name
 
    pipeline.ingest(tmp)
    result = pipeline.retrieve("Which companies share directors?")
    print(json.dumps(result, indent=2, default=str))
    os.unlink(tmp)