# BAOM AI — A Production-Grade Agentic Intelligence Platform

**Knowledge Graph Reasoning with Multi-Agent Orchestration and Small Language Models**

An MSc dissertation project (COMP702, University of Liverpool) that integrates three structurally distinct real-world datasets into a single Neo4j knowledge graph, and answers natural-language questions over that graph using a locally-hosted, agentic AI pipeline — without a paid cloud LLM API and without the user writing a single manual query.

- **Author:** Kailas Balaji Wadaje (Student ID: 201960479)
- **Programme:** MSc Data Science & Artificial Intelligence, University of Liverpool
- **Supervisor:** Dr Konstantinos Tsakalidis
- **GitHub:** [github.com/Kailaswadje](https://github.com/Kailaswadje)
- **Hugging Face:** 
---
![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20Database-008CC1?logo=neo4j&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Agent%20Framework-1C3C3C?logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM%20Inference-000000?logo=ollama&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-GNN-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Datasets](#datasets)
5. [Repository Structure](#repository-structure)
6. [Getting Started](#getting-started)
7. [Configuration](#configuration)
8. [Usage](#usage)
9. [Module-by-Module Overview](#module-by-module-overview)
10. [Results Summary](#results-summary)
11. [Testing](#testing)
12. [Tech Stack](#tech-stack)
13. [Known Limitations](#known-limitations)
14. [Future Work](#future-work)
15. [Ethical Compliance](#ethical-compliance)
16. [Use of Generative AI](#use-of-generative-ai)
17. [References](#references)
18. [Acknowledgements](#acknowledgements)
19. [Author](#author)

---

## Overview

Knowledge graphs are increasingly used to store interconnected organisational data, but extracting a natural-language answer from one conventionally requires a person who knows a graph query language (Cypher) and the specific schema of the graph being queried. Commercial cloud LLM APIs offer one way around this barrier, but their recurring cost, dependence on constant connectivity, and requirement to transmit potentially sensitive data to a third party make them unsuitable for many small organisations.

**BAOM AI** is a locally-deployable, eight-module platform that addresses this problem using small, open-weight language models running entirely on consumer-grade hardware via [Ollama](https://ollama.com/), rather than any paid cloud API. It integrates three structurally distinct datasets — a general-knowledge graph, a financial transaction graph, and a UK corporate registry — into a single Neo4j graph database, and exposes them through a Graph Neural Network, a tool-using ReAct reasoning agent, a multi-agent orchestration pipeline with a deterministic confidence mechanism, a hybrid retrieval layer, and a deployed web application.

The project was developed and evaluated on an 8GB-RAM laptop with no dedicated GPU, reflecting the real deployment envelope of the small organisations this platform is intended for. Every module was developed using a **diagnose → fix → verify** methodology: each module was evaluated honestly, its specific failure modes were identified and evidenced, a targeted fix was implemented, and the fix's effect was independently re-verified before moving on. This README documents the platform's real, evidenced capabilities and limitations — not an idealised description of what it was intended to do.

---

## Key Features

- 🔗 **Multi-dataset knowledge graph** — 218,778 nodes and 506,964 relationships across three datasets in one Neo4j instance
- 🧠 **Relation-aware Graph Neural Network** — GraphSAGE encoder + DistMult decoder for link prediction
- 🤖 **ReAct reasoning agent** — schema-grounded, tool-using single-agent query answering, with a verified-template fast-path for high-frequency questions
- 👥 **Multi-agent orchestration** — Researcher → Reasoner → Validator → Writer pipeline with a **fully deterministic** confidence score (not self-reported by the LLM)
- 🌐 **External Knowledge Agent** — live web-search fallback for genuinely out-of-dataset questions, avoiding hallucinated answers
- 🔍 **Hybrid retrieval** — combines GraphRAG (global/thematic) and HippoRAG (local/associative) retrieval over the same graph
- 📊 **Statistically rigorous SLM comparison** — three architecturally distinct small language models benchmarked with formal ANOVA + Tukey HSD testing
- 🛠️ **QLoRA fine-tuning** — parameter-efficient fine-tuning of a small model on graph-derived question–answer pairs
- ✅ **25/25 automated tests passing** across all eight modules
- 💻 **Fully local deployment** — Ollama + Neo4j Desktop, no dedicated GPU required (except for the fine-tuning step, run on a free Colab T4)

---

## System Architecture

```
 FB15k-237 (file)     Elliptic Bitcoin (file)     Companies House (live API)
        │                      │                          │
        └──────────────────────┼──────────────────────────┘
                                ▼
                     Module 1: Data Loader
                                │
                                ▼
              Neo4j Graph Database (218,778 nodes · 506,964 edges)
                    │           │            │            │
                    ▼           ▼            ▼            ▼
              Module 2:    Module 3:     Module 5:    Module 6:
                GNN       ReAct Agent   Hybrid RAG   SLM Comparison
                    │           │            │
                    └───────────┼────────────┘
                                ▼
                Module 4: Multi-Agent Orchestration
                  (+ External Knowledge Agent)
                                │
                                ▼
                  Module 8: Streamlit Application
                                │
                                ▼
        End user — natural-language question in, grounded answer out
```

Module 7 (QLoRA fine-tuning) reads from the same Neo4j instance to generate training data, but the GPU-bound fine-tuning step itself runs on a remote Google Colab T4 GPU rather than locally.

---

## Datasets

| Dataset | Type | Nodes | Edges | Source |
|---|---|---|---|---|
| **FB15k-237** | General knowledge graph | 14,505 (`:Entity`) | 272,115 (`:RELATION`) | Freebase subset (Toutanova & Chen, 2015) |
| **Elliptic Bitcoin** | Financial transaction graph | 203,769 (`:Transaction`) | 234,355 (`:TRANSFERS_TO`) | Published Kaggle dataset (Weber et al., 2019) |
| **Companies House UK** | Corporate registry | 20 (`:Company`) + 482 (`:Director`) | 494 (`:DIRECTOR_OF`) | Live UK government REST API |

All three datasets were chosen specifically because they are **structurally distinct**, so that every module could be tested for genuine cross-schema generalisation rather than being implicitly tuned to a single dataset's structure. All three are publicly available, open-licence, non-personal, structured data — see [Ethical Compliance](#ethical-compliance).
---
Elliptic Data: https://www.kaggle.com/datasets/kailaswadje/elliptic
FB15K-237 Data: https://www.kaggle.com/datasets/kailaswadje/fb15k-237-dataset

## Repository Structure

```
.
├── data/
│   └── loader.py                          # Module 1 — loads all 3 datasets into Neo4j
├── models/
│   └── gnn.py                             # Module 2 — GraphSAGE + DistMult GNN (combined heterogeneous model)
├── agent/
│   └── react_agent.py                     # Module 3 — ReAct single-agent query answering
├── orchestration/
│   ├── multi_agent.py                     # Module 4 — multi-agent orchestration + deterministic confidence
│   ├── external_knowledge_agent.py        # Module 4 extension — web-search fallback agent
│   └── evaluate_fix.py                    # Before/after evaluation harness for the External Knowledge Agent
├── retrieval/
│   └── hybrid_rag.py                      # Module 5 — GraphRAG + HippoRAG hybrid retrieval
├── slm/
│   └── comparison.py                      # Module 6 — statistical SLM comparison (ANOVA/Tukey HSD)
├── mlops/
│   └── pipeline.py                        # Module 7 — QLoRA training-data generation + fine-tuning pipeline
├── demo/
│   └── app.py                             # Module 8 — Streamlit web application
├── tests/
│   └── test_all.py                        # Automated test suite (25 tests, pytest)
├── data/financial_qa.json                 # Generated QLoRA training data (Q&A pairs from the live graph)
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.13 (the automated test suite was run and confirmed passing under Python 3.13.0)
- [Neo4j Desktop](https://neo4j.com/download/) (developed and tested against Neo4j Desktop 2.2.1)
- [Ollama](https://ollama.com/) installed locally, with the following models pulled:
  - `llama3.2` (7B)
  - `phi3` (Phi-3-Mini, 3.8B)
- A free [Companies House API](https://developer.company-information.service.gov.uk/) key, for live corporate-registry access

### Installation

```bash
git clone https://github.com/Kailaswadje/<repository-name>.git
cd <repository-name>
pip install -r requirements.txt --break-system-packages
```

Principal Python dependencies (see [Tech Stack](#tech-stack) for the full list): `neo4j`, `torch`, `torch_geometric`, `langchain`, `langgraph`, `transformers`, `trl`, `peft`, `bitsandbytes`, `streamlit`, `pytest`, `requests`, `python-dotenv`.

---

## Configuration

Create a `.env` file in the project root with your own credentials (never commit this file):

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your-neo4j-password>
COMPANIES_HOUSE_API_KEY=<your-companies-house-api-key>
```

The Companies House API specifically requires **HTTP Basic Authentication** (the API key as the username, with an empty password) — a plain bearer-token header will be silently rejected with a 401 error.

---

## Usage

**Load all three datasets into Neo4j:**
```bash
python -m data.loader
```

**Train the combined heterogeneous GNN:**
```bash
python -m models.gnn
```

**Run the automated test suite:**
```bash
pytest tests/test_all.py -v
```

**Launch the deployed web application:**
```bash
streamlit run demo/app.py
```
Then open `localhost:8501` in your browser. From the Settings panel you can select the language model (LLaMA-3.2 or Phi-3-Mini), toggle multi-agent orchestration, toggle the web-search fallback for out-of-dataset questions, and ask a question in plain English.

**Run the SLM statistical comparison:**
```bash
python -m slm.comparison
```

**Run the External Knowledge Agent before/after evaluation:**
```bash
python -m orchestration.evaluate_fix
```

---

## Module-by-Module Overview

### Module 1 — Data Loader
Ingests FB15k-237 and Elliptic Bitcoin from local files, and Companies House UK from the live REST API, into one shared Neo4j instance using distinct node labels and relationship types per dataset. Elliptic Bitcoin's ~200,000 rows are written in UNWIND-based batches of 2,000, reducing load time from over an hour to under fifteen minutes compared to an unbatched, one-transaction-per-row approach.

### Module 2 — Graph Neural Network
A two-layer GraphSAGE encoder with a relation-aware DistMult decoder. The standalone FB15k-237 model improved from an initial MRR of 0.0031 to 0.1516 (a 48× improvement) across four diagnosed iterations (data-integrity fix → deterministic features → learnable embeddings → relation-awareness). A combined heterogeneous variant, trained jointly across all three datasets with a per-node-type fallback projection, achieves MRR 0.1904 (RELATION), 0.2363 (TRANSFERS_TO), and 0.9929 (DIRECTOR_OF, on a small 494-edge sample — treated as a caveat, not a robust generalisation claim; see [Known Limitations](#known-limitations)).

### Module 3 — ReAct Single Agent
A LangChain ReAct agent with three tools (`GraphSchema`, `CypherQuery`, `GNNPredict`). Testing surfaced two failure modes — invented relationship types and non-termination without a final answer — addressed with explicit schema grounding, a repeat-call guard, and a graceful-degradation fallback. A code-level grounding check flags any answer not actually supported by an executed Cypher query with an `[UNGROUNDED]` marker. A template fast-path bypasses the agent entirely for well-verified, high-frequency question patterns.

### Module 4 — Multi-Agent Orchestration
Four agents (Researcher → Reasoner → Validator → Writer) coordinated with LangGraph. An initial design asked the LLM to self-report its own confidence, which was found to be non-reproducible (the same question scored 0.20 in one run and 0.80 in another). This was replaced with a fully deterministic calculation based only on whether real Cypher evidence was found and the Validator's independently-parsed status (VERIFIED / PARTIAL / UNVERIFIED), yielding one of four fixed confidence values: 0.20, 0.35, 0.55, or 0.90.

### Module 4 Extension — External Knowledge Agent
A fifth agent, triggered when confidence falls below 0.6, performs a live DuckDuckGo web search rather than relying on the LLM's own (potentially outdated) training knowledge. Fixing a vague, temporally-ambiguous search query (by explicitly injecting the current year) turned a hallucinated "Argentina" answer to "who won the latest World Cup" into a correctly web-grounded "Spain" answer, at confidence 0.85.

### Module 5 — Hybrid Retrieval
Combines GraphRAG's community-detection-based global view with HippoRAG's Personalised-PageRank-based local view, run over real text generated live from the Neo4j graph and the Companies House API — extracting 31 entities and 25 relations, with "Company" ranked as the most structurally central concept (PPR = 0.218).

### Module 6 — SLM Comparison
Benchmarks LLaMA-3.2 (7B), Phi-3-Mini (3.8B), and FinBERT (110M, an encoder-only negative control) on 24 grounded questions, three runs each. Results: **LLaMA-3.2 83.3%**, **Phi-3-Mini 95.8%**, **FinBERT 0%** (by architectural design, since it cannot generate open-ended text). One-way ANOVA: F = 88.50, p < 0.0001; all pairwise Tukey HSD comparisons significant (p < 0.001).

### Module 7 — QLoRA Fine-tuning
Generates question–answer pairs directly from live Neo4j queries across all three datasets, then fine-tunes `microsoft/Phi-3-mini-4k-instruct` using 4-bit quantisation and LoRA adapters on a Google Colab T4 GPU. Training loss fell from 1.02 to 0.035 over 500 steps.

### Module 8 — Streamlit Application
The deployed, end-user-facing web interface: model selection, multi-agent orchestration toggle, web-search fallback toggle, GNN top-K slider, an "Ask a question" panel showing confidence / GNN-used / Cypher-queries-executed / web-search-used metrics, and expandable agent-reasoning-chain and executed-Cypher-query panels for full answer provenance.

---

## Results Summary

| Metric | Result |
|---|---|
| GNN MRR improvement (standalone, FB15k-237) | 0.0031 → 0.1516 (**48×**) |
| Combined GNN Hits@10 (Companies House, DIRECTOR_OF) | **1.00** |
| ReAct questions answered via verified template | **5/8** |
| Out-of-dataset questions correctly resolved via web search | **3/3** |
| Deterministic confidence on fully graph-verified answers | **0.90** |
| Best SLM accuracy (Phi-3-Mini) | **95.8%** |
| Statistical significance of model comparison | **p < 0.0001** |
| Automated tests passing | **25/25** |
| Typical end-to-end multi-agent response time | **≈ 0.5–1 minute** |

---

## Testing

The full pytest suite (25 tests across 8 test classes) covers: Neo4j connectivity and CSV ingestion; the combined GNN's model construction, negative sampling, and decode function; all four ReAct agent tools; the multi-agent state graph's construction; both hybrid-RAG components; the SLM comparison's scoring function and statistical-test wrapper; multi-dataset coverage; and MLOps packaging.

```
============================= test session starts =============================
platform win32 -- Python 3.13.0, pytest-9.1.1, pluggy-1.6.0
collected 25 items
...
============================= 25 passed in 25.56s =============================
```

---

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Graph database | [Neo4j](https://neo4j.com/) (Desktop 2.2.1) |
| LLM inference (local) | [Ollama](https://ollama.com/) — LLaMA-3.2 (7B), Phi-3-Mini (3.8B) |
| Agent framework | [LangChain](https://docs.langchain.com/) (ReAct agent), [LangGraph](https://docs.langchain.com/) (multi-agent orchestration) |
| GNN | PyTorch, PyTorch Geometric |
| Fine-tuning | Hugging Face `transformers`, `trl`, `peft`, `bitsandbytes` (QLoRA) |
| Web application | [Streamlit](https://docs.streamlit.io/) |
| Testing | pytest |
| External data access | `requests` (Companies House REST API), DuckDuckGo web search |
| GPU (fine-tuning only) | Google Colab, T4 GPU |

---

## Known Limitations

Documented honestly, in the interest of exposing rather than only asserting the solution of every limitation:

- The GNN evaluation sample (500 edges) is drawn from the training graph, not a held-out train/validation/test split — reported metrics should be treated as an optimistic upper bound.
- The near-perfect DIRECTOR_OF result (MRR 0.9929) is measured on only 494 edges, and is more consistent with memorisation than with demonstrated generalisation on a dataset this small.
- The deterministic confidence mechanism only credits `CypherQuery` evidence, not `GraphSchema` evidence — a correct schema-only answer can be under-credited with low confidence.
- The SLM comparison's scoring function uses stop-word-filtered keyword overlap, a heuristic rather than a semantic judgement of correctness.
- No human-baseline or inter-rater check exists for any automated evaluation in this project, since the project's A0 ethical-approval category precludes human-participant involvement.

---

## Future Work

- A proper held-out train/validation/test edge split for the GNN, particularly for the DIRECTOR_OF edge type.
- A semantic, rather than keyword-overlap, scoring mechanism for the SLM comparison.
- Extending the deterministic confidence mechanism to credit `GraphSchema`-derived evidence.
- A systematic, repeated-run audit of the confidence mechanism's determinism in practice.
- Testing on a fourth, larger corporate dataset to properly evaluate DIRECTOR_OF link-prediction generalisation.
- Evaluating additional small-model families beyond LLaMA-3.2 and Phi-3-Mini.

---

## Ethical Compliance

**Ethical Approval Category: A0** (Data Category A, Participant Category 0).

This project does not use any data derived from humans or animals, and does not involve human participants in any activity, including requirements analysis or software evaluation. All three datasets are publicly available, open-licence, non-personal, structured data. All testing and evaluation is automated against pre-verified gold answers computed from the live graph, or against formal statistical tests — no human evaluators, interviewees, or survey respondents were involved at any stage.

---

## Use of Generative AI

In accordance with the University's policy on generative AI use, this project discloses that generative AI was used only in two bounded capacities: as a **debugging aid**, to help diagnose likely causes of runtime errors or test failures (with every suggested fix independently implemented and verified before acceptance); and as a **grammatical and language-proficiency aid**, to review and correct the clarity and grammar of the author's own draft text while writing the accompanying dissertation. Generative AI was not used to generate this project's technical content, findings, or conclusions.

---

## References

- Bai, X. et al. (2025) 'Top Ten Challenges Towards Agentic Neural Graph Databases', arXiv:2501.14224.
- Dettmers, T. et al. (2023) 'QLoRA: Efficient Finetuning of Quantized LLMs', arXiv:2305.14314.
- Edge, D. et al. (2024) 'From Local to Global: A Graph RAG Approach to Query-Focused Summarization', arXiv:2404.16130.
- Gutierrez, B. J. et al. (2024) 'HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models', arXiv:2405.14831.
- Hamilton, W., Ying, R. and Leskovec, J. (2017) 'Inductive Representation Learning on Large Graphs', arXiv:1706.02216.
- Toutanova, K. and Chen, D. (2015) 'Observed versus Latent Features for Knowledge Base and Text Inference', Proceedings of the 3rd Workshop on Continuous Vector Space Models and their Compositionality.
- Weber, M. et al. (2019) 'Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics', arXiv:1908.02591.
- Yang, B. et al. (2015) 'Embedding Entities and Relations for Learning and Inference in Knowledge Bases', ICLR.
- Yao, S. et al. (2023) 'ReAct: Synergizing Reasoning and Acting in Language Models', ICLR.

---

## Acknowledgements

Thanks to Dr Konstantinos Tsakalidis, my dissertation supervisor, for his guidance throughout this project, and to the University of Liverpool for its support.

---

## Author

**Kailas Balaji Wadaje**
MSc Data Science & Artificial Intelligence, University of Liverpool
[GitHub](https://github.com/Kailaswadje) · [Hugging Face](https://huggingface.co/KailasWadje01)
