# =============================================================================
# MODULE 7 — MLOPS PIPELINE
# QLoRA fine-tuning (run on Google Colab T4 GPU)
# Docker packaging, PyPI distribution, GitHub Actions CI/CD
# =============================================================================
 
import os
import json
import subprocess
from typing import Optional
 
 
# ------------------------------------------------------------------
# 7a. QLoRA fine-tuning (designed for Google Colab — free T4 GPU)
# ------------------------------------------------------------------
def finetune_with_qlora(
    base_model: str = "microsoft/phi-3-mini-4k-instruct",
    dataset_path: str = "data/financial_qa.json",
    output_dir: str = "models/finetuned",
    epochs: int = 3,
    batch_size: int = 4,
    max_steps: int = 500,
    save_steps: int = 100,
):
    """
    QLoRA fine-tuning of the best model from Module 6.
    Run this on Google Colab (T4 GPU) — not on local 8GB RAM laptop.
    
    Install requirements on Colab:
        pip install transformers peft bitsandbytes accelerate datasets trl
    """
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer
        from datasets import load_dataset
 
        print(f"[M7] QLoRA fine-tuning: {base_model}")
        print(f"[M7] Dataset: {dataset_path}")
        print(f"[M7] Output: {output_dir}")
 
        # 4-bit quantisation config (QLoRA)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
 
        # Load model in 4-bit
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
 
        # LoRA config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
 
        # Load financial QA dataset
        dataset = load_dataset("json", data_files=dataset_path, split="train")
 
        def format_prompt(example):
            return {
                "text": f"### Question:\n{example['question']}\n\n### Answer:\n{example['answer']}"
            }
        dataset = dataset.map(format_prompt)
 
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            max_steps=max_steps,
            save_steps=save_steps,
            logging_steps=50,
            learning_rate=2e-4,
            fp16=True,
            report_to="none",
            save_total_limit=2,
        )
 
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=512,
        )
 
        print("[M7] Starting QLoRA training...")
        trainer.train()
        trainer.save_model(output_dir)
        print(f"[M7] Fine-tuned model saved → {output_dir}")
        return output_dir
 
    except ImportError as e:
        print(f"[M7] QLoRA dependencies not installed: {e}")
        print("[M7] Run on Google Colab: pip install transformers peft bitsandbytes accelerate trl")
        return None
 
 
# ------------------------------------------------------------------
# 7b. Generate financial QA training data from Neo4j
# ------------------------------------------------------------------
def generate_training_data(output_path: str = "data/financial_qa.json"):
    """Generate QA pairs from the Neo4j knowledge graph for fine-tuning.
 
    NOTE: rewritten for the FB15k-237 (Entity/RELATION) schema actually
    loaded in this project's Neo4j instance. The original templates
    assumed a Companies House schema (Transaction/Company/Director)
    which does not exist here and produced 0 QA pairs.
    """
    from neo4j import GraphDatabase
 
    uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USER",     "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "Kailas@123")
 
    driver = GraphDatabase.driver(uri, auth=(user, password))
    qa_pairs = []
 
    templates = [
        ("MATCH (e:Entity)-[r:RELATION]-() RETURN e.name AS name, count(r) AS degree "
         "ORDER BY degree DESC LIMIT 5",
         "Which entities are most connected in the graph?",
         lambda r: "The most connected entities are: " + ", ".join(
             f"{row['name']} ({row['degree']} connections)" for row in r
         )),
 
        ("MATCH (n:Entity) RETURN count(n) AS c",
         "How many entities are in the knowledge graph?",
         lambda r: f"There are {r[0]['c']} entities in the knowledge graph."),
 
        ("MATCH ()-[r:RELATION]->() RETURN count(r) AS c",
         "How many relationships are in the knowledge graph?",
         lambda r: f"There are {r[0]['c']} relationships in the knowledge graph."),
 
        ("MATCH ()-[r:RELATION]->() RETURN r.type AS rtype, count(*) AS freq "
         "ORDER BY freq DESC LIMIT 5",
         "What are the top 5 most common relationship types?",
         lambda r: "The most common relationship types are: " + ", ".join(
             f"{row['rtype']} ({row['freq']} occurrences)" for row in r
         )),
 
        ("MATCH (e:Entity {name: '/m/06rf7'})-[r:RELATION]-(f:Entity) "
         "RETURN f.name AS name, r.type AS rtype LIMIT 5",
         "Find entities related to /m/06rf7 in the knowledge graph.",
         lambda r: "Entities related to /m/06rf7: " + ", ".join(
             f"{row['name']} (via {row['rtype']})" for row in r
         )),
 
        # --- Elliptic Bitcoin (Transaction/TRANSFERS_TO) ---
        ("MATCH (t:Transaction) RETURN count(t) AS c",
         "How many transactions are in the Bitcoin transaction graph?",
         lambda r: f"There are {r[0]['c']} transactions in the Bitcoin transaction graph."),
 
        ("MATCH (t:Transaction {isFraud: true}) RETURN count(t) AS c",
         "How many transactions are flagged as fraudulent?",
         lambda r: f"There are {r[0]['c']} transactions flagged as fraudulent."),
 
        ("MATCH ()-[r:TRANSFERS_TO]->() RETURN count(r) AS c",
         "How many transfer relationships exist in the transaction graph?",
         lambda r: f"There are {r[0]['c']} TRANSFERS_TO relationships in the transaction graph."),
 
        # --- Companies House UK (Company/Director/DIRECTOR_OF) ---
        ("MATCH (c:Company) RETURN count(c) AS c",
         "How many companies are in the corporate graph?",
         lambda r: f"There are {r[0]['c']} companies in the corporate graph."),
 
        ("MATCH (d:Director) RETURN count(d) AS c",
         "How many directors are in the corporate graph?",
         lambda r: f"There are {r[0]['c']} directors in the corporate graph."),
 
        ("MATCH (d:Director)-[:DIRECTOR_OF]->(c1:Company), (d)-[:DIRECTOR_OF]->(c2:Company) "
         "WHERE c1.number < c2.number "
         "RETURN d.name AS director, c1.name AS company1, c2.name AS company2 LIMIT 5",
         "Which companies share a director?",
         lambda r: ("Companies sharing a director: " + ", ".join(
             f"{row['company1']} and {row['company2']} (via {row['director']})" for row in r
         )) if r else "No companies share a director."),
 
        ("MATCH (c:Company) RETURN c.name AS name, c.status AS status LIMIT 5",
         "List some companies and their status.",
         lambda r: "Companies: " + ", ".join(
             f"{row['name']} ({row['status']})" for row in r
         )),
    ]
 
    with driver.session() as session:
        for cypher, question, fmt in templates:
            try:
                result = list(session.run(cypher))
                if result:
                    qa_pairs.append({
                        "question": question,
                        "answer":   fmt(result),
                        "cypher":   cypher
                    })
            except Exception as e:
                print(f"[M7] Skipped template: {e}")
 
    driver.close()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)
    print(f"[M7] Generated {len(qa_pairs)} QA pairs → {output_path}")
    return qa_pairs
 
 
# ------------------------------------------------------------------
# 7c. Docker management
# ------------------------------------------------------------------
def build_docker_image(tag: str = "agentic-platform:latest"):
    """Build Docker image for the platform."""
    dockerfile = """FROM python:3.11-slim
 
WORKDIR /app
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY . .
 
ENV NEO4J_URI=bolt://neo4j:7687
ENV NEO4J_USER=neo4j
ENV NEO4J_PASSWORD=password123
ENV LLM_MODEL=llama3.2
 
EXPOSE 8501
 
CMD ["streamlit", "run", "demo/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    compose = """version: '3.8'
services:
  neo4j:
    image: neo4j:5
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password123
    volumes:
      - neo4j_data:/data
 
  platform:
    build: .
    ports:
      - "8501:8501"
    depends_on:
      - neo4j
    environment:
      - NEO4J_URI=bolt://neo4j:7687
 
volumes:
  neo4j_data:
"""
    with open("Dockerfile", "w") as f:
        f.write(dockerfile)
    with open("docker-compose.yml", "w") as f:
        f.write(compose)
    print("[M7] Dockerfile and docker-compose.yml created")
 
    result = subprocess.run(
        ["docker", "build", "-t", tag, "."],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"[M7] Docker image built: {tag}")
    else:
        print(f"[M7] Docker build error: {result.stderr}")
    return result.returncode == 0
 
 
# ------------------------------------------------------------------
# 7d. Generate setup.py for PyPI
# ------------------------------------------------------------------
def generate_package_files(package_name: str = "agentic-intelligence-platform"):
    """Generate setup.py and pyproject.toml for PyPI distribution."""
    setup_py = f'''from setuptools import setup, find_packages
 
setup(
    name="{package_name}",
    version="0.1.0",
    description="Production-grade agentic intelligence platform: "
                "GNN + Multi-Agent + SLM for domain-agnostic graph reasoning",
    author="[Your Name]",
    packages=find_packages(),
    install_requires=[
        "neo4j>=5.0",
        "torch>=2.0",
        "torch-geometric>=2.3",
        "langchain>=0.2",
        "langchain-ollama>=0.1",
        "langchain-community>=0.2",
        "langgraph>=0.1",
        "crewai>=0.1",
        "transformers>=4.40",
        "streamlit>=1.30",
        "networkx>=3.0",
        "scipy>=1.10",
        "numpy>=1.24",
        "requests>=2.28",
    ],
    python_requires=">=3.11",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
)
'''
    pyproject = f'''[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.backends.legacy:build"
 
[project]
name = "{package_name}"
version = "0.1.0"
requires-python = ">=3.11"
'''
    with open("setup.py", "w") as f:
        f.write(setup_py)
    with open("pyproject.toml", "w") as f:
        f.write(pyproject)
    print(f"[M7] Package files created for '{package_name}'")
    print("[M7] To publish: python -m build && twine upload dist/*")
 
 
# ------------------------------------------------------------------
# 7e. GitHub Actions CI/CD workflow
# ------------------------------------------------------------------
def generate_github_actions():
    """Generate .github/workflows/ci.yml for automated testing."""
    os.makedirs(".github/workflows", exist_ok=True)
 
    ci_yaml = """name: CI
 
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
 
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      neo4j:
        image: neo4j:5
        ports:
          - 7687:7687
        env:
          NEO4J_AUTH: neo4j/password123
 
    steps:
      - uses: actions/checkout@v4
 
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
 
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
 
      - name: Run unit tests
        run: pytest tests/ -v --cov=. --cov-report=xml
        env:
          NEO4J_URI: bolt://localhost:7687
          NEO4J_USER: neo4j
          NEO4J_PASSWORD: password123
 
      - name: Upload coverage
        uses: codecov/codecov-action@v4
"""
    with open(".github/workflows/ci.yml", "w") as f:
        f.write(ci_yaml)
    print("[M7] GitHub Actions CI/CD workflow created")
 
 
if __name__ == "__main__":
    generate_training_data()
    generate_package_files()
    generate_github_actions()
    build_docker_image()