# =============================================================================
# MODULE 8 — DEMO & DEPLOYMENT
# Streamlit web app: upload data → ask questions → see graph + reasoning
# Deploy free on HuggingFace Spaces
# =============================================================================
 
import streamlit as st
import json
import pandas as pd
import tempfile
import os
import sys
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
st.set_page_config(
    page_title="Agentic Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)
 
# ------------------------------------------------------------------
# Sidebar: model selector + settings
# ------------------------------------------------------------------
st.sidebar.title("⚙ Settings")
model_choice = st.sidebar.selectbox(
    "Language Model",
    ["LLaMA 3.2 (7B)", "Phi-3 Mini (3.8B)", "FinBERT (110M)"],
    index=0
)
use_orchestration = st.sidebar.checkbox("Multi-agent orchestration (M4)", value=True)
use_web_fallback  = st.sidebar.checkbox(
    "Web search fallback for out-of-dataset questions",
    value=True,
    help="When the graph has low confidence (data not found), automatically "
         "search the web for an answer instead of just saying 'I don't know' "
         "-- see orchestration/external_knowledge_agent.py"
)
show_reasoning    = st.sidebar.checkbox("Show agent reasoning chain", value=True)
top_k_gnn         = st.sidebar.slider("GNN top-K predictions", 1, 10, 5)
 
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset loaded:**")
if "dataset_loaded" in st.session_state and st.session_state.dataset_loaded:
    st.sidebar.success(f"✓ {st.session_state.get('dataset_name', 'Dataset')}")
else:
    st.sidebar.warning("No dataset loaded")
 
# ------------------------------------------------------------------
# Main layout: 3 columns
# ------------------------------------------------------------------
col_left, col_centre, col_right = st.columns([1, 2, 1.5])
 
# ── LEFT: Dataset upload ──────────────────────────────────────────
with col_left:
    st.subheader("📂 Data upload")
 
    upload_type = st.radio(
        "Data source",
        ["CSV file", "JSON file", "FB15k-237", "Elliptic Bitcoin", "Companies House API"]
    )
 
    if upload_type in ("CSV file", "JSON file"):
        uploaded = st.file_uploader(
            f"Upload {upload_type}",
            type=["csv"] if upload_type == "CSV file" else ["json"]
        )
        if uploaded and st.button("Load into graph"):
            with st.spinner("Loading data into Neo4j..."):
                try:
                    suffix = ".csv" if upload_type == "CSV file" else ".json"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                        f.write(uploaded.read())
                        tmp_path = f.name
                    from data.loader import Neo4jLoader
                    loader = Neo4jLoader()
                    if upload_type == "CSV file":
                        loader.load_csv(tmp_path)
                    os.unlink(tmp_path)
                    nodes, rels = loader.get_stats()
                    loader.close()
                    st.session_state.dataset_loaded = True
                    st.session_state.dataset_name = uploaded.name
                    st.success(f"Loaded! {nodes:,} nodes, {rels:,} relationships")
                except Exception as e:
                    st.error(f"Load error: {e}")
 
    elif upload_type == "FB15k-237":
        if st.button("Load FB15k-237 (benchmark)"):
            with st.spinner("Downloading and loading FB15k-237..."):
                try:
                    import subprocess
                    os.makedirs("data/fb15k237", exist_ok=True)
                    url = ("https://raw.githubusercontent.com/DeepGraphLearning/"
                           "KnowledgeGraphEmbedding/master/data/FB15k-237/train.txt")
                    subprocess.run(["wget", "-q", "-O", "data/fb15k237/train.txt", url], check=True)
                    from data.loader import Neo4jLoader
                    loader = Neo4jLoader()
                    count  = loader.load_fb15k237("data/fb15k237/train.txt")
                    loader.close()
                    st.session_state.dataset_loaded = True
                    st.session_state.dataset_name   = "FB15k-237"
                    st.success(f"Loaded {count:,} triples!")
                except Exception as e:
                    st.error(f"Load error: {e}")
 
    elif upload_type == "Elliptic Bitcoin":
        st.info("Elliptic Bitcoin is loaded via CSV files (features, classes, edgelist) "
                "using data/loader.py's load_elliptic() method, typically as part of the "
                "project's initial data-loading step rather than via this UI.")
        if st.button("Check Elliptic Bitcoin status"):
            with st.spinner("Checking graph for Elliptic Bitcoin data..."):
                try:
                    from data.loader import Neo4jLoader
                    loader = Neo4jLoader()
                    with loader.driver.session() as session:
                        count = session.run("MATCH (t:Transaction) RETURN count(t) AS c").single()["c"]
                    loader.close()
                    if count > 0:
                        st.session_state.dataset_loaded = True
                        st.session_state.dataset_name   = "Elliptic Bitcoin"
                        st.success(f"Elliptic Bitcoin already loaded: {count:,} transactions found.")
                    else:
                        st.warning("No Elliptic Bitcoin data found in the graph yet. "
                                   "Load it first via data/loader.py's load_elliptic() method.")
                except Exception as e:
                    st.error(f"Check error: {e}")
 
    elif upload_type == "Companies House API":
        api_key = st.text_input("Companies House API key", type="password")
        numbers = st.text_area("Company numbers (one per line)")
        if st.button("Fetch from API") and api_key and numbers:
            with st.spinner("Fetching from Companies House..."):
                try:
                    from data.loader import Neo4jLoader
                    loader     = Neo4jLoader()
                    num_list   = [n.strip() for n in numbers.strip().split("\n") if n.strip()]
                    loaded     = loader.load_companies_house(api_key, num_list)
                    loader.close()
                    st.session_state.dataset_loaded = True
                    st.session_state.dataset_name   = "Companies House UK"
                    st.success(f"Loaded {loaded} companies!")
                except Exception as e:
                    st.error(f"API error: {e}")
 
    # Graph stats
    st.markdown("---")
    st.subheader("📊 Graph stats")
    if st.button("Refresh stats"):
        try:
            from data.loader import Neo4jLoader
            loader = Neo4jLoader()
            nodes, rels = loader.get_stats()
            loader.close()
            col1, col2 = st.columns(2)
            col1.metric("Nodes", f"{nodes:,}")
            col2.metric("Relationships", f"{rels:,}")
        except Exception as e:
            st.error(f"Stats error: {e}")
 
# ── CENTRE: Question + Answer ─────────────────────────────────────
with col_centre:
    st.subheader("💬 Ask a question")
 
    question = st.text_area(
        "Enter your question in plain English",
        placeholder="Which companies share directors and also have late filing history?",
        height=100
    )
 
    col_ask, col_clear = st.columns([3, 1])
    ask_clicked  = col_ask.button("Ask", type="primary", use_container_width=True)
    clear_clicked = col_clear.button("Clear", use_container_width=True)
 
    if clear_clicked:
        if "history" in st.session_state:
            st.session_state.history = []
 
    if ask_clicked and question.strip():
        # FIX: the model selector dropdown previously had no effect --
        # LLM_MODEL was never set based on the user's choice, so the
        # agent always used whatever the environment default was,
        # regardless of what the user selected here.
        if model_choice.startswith("LLaMA"):
            os.environ["LLM_MODEL"] = "llama3.2"
        elif model_choice.startswith("Phi-3"):
            os.environ["LLM_MODEL"] = "phi3:mini"
        elif model_choice.startswith("FinBERT"):
            st.warning("FinBERT is an encoder-only classification model and cannot "
                       "answer open-ended questions directly (see Module 6 comparison "
                       "for why) -- using LLaMA 3.2 for this query instead.")
            os.environ["LLM_MODEL"] = "llama3.2"
 
        with st.spinner(f"Agents working ({model_choice})..."):
            try:
                # FIX: agent.react_agent and orchestration.multi_agent both
                # read LLM_MODEL into a module-level constant AT IMPORT
                # TIME. Since Streamlit keeps modules cached across
                # reruns within a session, simply setting os.environ
                # above has no effect on an already-imported module --
                # the model dropdown would silently keep using whichever
                # model was selected FIRST. Reloading forces the module
                # to re-read the current LLM_MODEL value.
                import importlib
                import agent.react_agent as _react_agent_module
                importlib.reload(_react_agent_module)
 
                if use_orchestration:
                    import orchestration.multi_agent as _multi_agent_module
                    importlib.reload(_multi_agent_module)
 
                    if use_web_fallback:
                        # Use the web-search-fallback-enabled pipeline
                        # (orchestration/external_knowledge_agent.py)
                        # instead of the base multi-agent pipeline, so
                        # out-of-dataset questions (low graph confidence)
                        # automatically get an answer from a live web
                        # search rather than just "I don't know".
                        import orchestration.external_knowledge_agent as _ewk_module
                        importlib.reload(_ewk_module)
                        from orchestration.external_knowledge_agent import orchestrated_ask_v2
                        result = orchestrated_ask_v2(question)
                    else:
                        from orchestration.multi_agent import orchestrated_ask
                        result = orchestrated_ask(question)
                else:
                    from agent.react_agent import ask
                    raw    = ask(question, verbose=False)
                    result = {
                        "final_answer":   raw["answer"],
                        "confidence":     0.7,
                        "cypher_queries": raw["cypher_used"],
                        "gnn_used":       False,
                        "reasoning":      str(raw["reasoning"]),
                    }
 
                # Show answer
                st.success("Answer")
                st.write(result["final_answer"])
 
                # Metrics
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Confidence", f"{result.get('confidence', 0):.0%}")
                mc2.metric("GNN used", "Yes" if result.get("gnn_used") else "No")
                mc3.metric("Cypher queries", len(result.get("cypher_queries", [])))
                mc4.metric("Web search used", "Yes" if result.get("web_search_used") else "No")
 
                # Reasoning chain
                if show_reasoning and result.get("reasoning"):
                    with st.expander("Agent reasoning chain"):
                        st.text(result["reasoning"][:2000])
 
                # Cypher queries used
                if result.get("cypher_queries"):
                    with st.expander("Cypher queries executed"):
                        for cq in result["cypher_queries"]:
                            st.code(cq, language="cypher")
 
                # Store in history
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({
                    "question": question,
                    "answer":   result["final_answer"],
                    "model":    model_choice,
                })
 
            except Exception as e:
                st.error(f"Agent error: {e}")
                st.info("Make sure Neo4j is running and Ollama has the selected model.")
 
    # History
    if "history" in st.session_state and st.session_state.history:
        st.markdown("---")
        st.subheader("History")
        for item in reversed(st.session_state.history[-5:]):
            with st.expander(f"Q: {item['question'][:60]}..."):
                st.write(f"**Model:** {item['model']}")
                st.write(f"**Answer:** {item['answer']}")
 
# ── RIGHT: Graph visualisation ────────────────────────────────────
with col_right:
    st.subheader("🕸 Graph subgraph")
 
    if st.button("Visualise graph sample"):
        try:
            from neo4j import GraphDatabase
            uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER",      "neo4j")
            pwd  = os.getenv("NEO4J_PASSWORD",  "Kailas@123")
            driver = GraphDatabase.driver(uri, auth=(user, pwd))
            with driver.session() as session:
                records = list(session.run(
                    "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50"
                ))
            driver.close()
 
            if records:
                # Build simple adjacency for display
                nodes_set, edges_list = set(), []
                for rec in records:
                    n_name = rec["n"].get("name") or rec["n"].get("txId") or str(rec["n"].id)
                    m_name = rec["m"].get("name") or rec["m"].get("txId") or str(rec["m"].id)
                    nodes_set.add(n_name)
                    nodes_set.add(m_name)
                    edges_list.append((n_name, m_name))
 
                import networkx as nx
                G  = nx.DiGraph()
                G.add_nodes_from(nodes_set)
                G.add_edges_from(edges_list)
 
                # Display stats instead of full graph (pyvis optional)
                st.metric("Nodes in sample", G.number_of_nodes())
                st.metric("Edges in sample", G.number_of_edges())
 
                # Edge table
                df = pd.DataFrame(edges_list[:20], columns=["From", "To"])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No graph data found. Load a dataset first.")
        except Exception as e:
            st.error(f"Visualisation error: {e}")
 
    st.markdown("---")
    st.subheader("📈 Metrics")
 
    # SLM comparison button
    if st.button("Run SLM comparison (Module 6)"):
        with st.spinner("Running SLM comparison — this takes several minutes..."):
            try:
                from comparison import run_comparison
                results = run_comparison()
                model_results = results.get("model_results", [])
                if model_results:
                    df = pd.DataFrame([{
                        "Model":    r["model"],
                        "Accuracy": f"{r['mean_accuracy']:.3f}",
                        "±Std":     f"{r['std_accuracy']:.3f}",
                        "Latency":  f"{r['mean_latency_s']:.1f}s",
                    } for r in model_results])
                    st.dataframe(df, use_container_width=True)
                    pval = results["statistical_test"]["p_value"]
                    if pval < 0.05:
                        st.success(f"Statistically significant! p={pval:.4f}")
                    else:
                        st.warning(f"Not significant. p={pval:.4f}")
            except Exception as e:
                st.error(f"Comparison error: {e}")
 
# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center>A Production-Grade Agentic Intelligence Platform · "
    "MSc Data Science & AI · University of Liverpool · "
    "[GitHub](https://github.com/Kailaswadje) · [HuggingFace](https://huggingface.co/KailasWadje01)</center>",
    unsafe_allow_html=True
)
