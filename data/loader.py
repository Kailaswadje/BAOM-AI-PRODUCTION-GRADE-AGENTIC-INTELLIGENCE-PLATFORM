# =============================================================================
# MODULE 1 — DATA FOUNDATION
# Neo4j graph database loader — domain-agnostic
# Supports: CSV, JSON, Companies House UK API, Elliptic Bitcoin, FB15k-237
# =============================================================================
 
import os
import csv
import json
import requests
from neo4j import GraphDatabase
from dotenv import load_dotenv
 
# Load variables from .env file (NEO4J_URI, NEO4J_PASSWORD, COMPANIES_HOUSE_API_KEY, etc.)
load_dotenv()
 
NEO4J_URI                = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER               = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD           = os.getenv("NEO4J_PASSWORD", "Kailas@123")
COMPANIES_HOUSE_API_KEY  = os.getenv("COMPANIES_HOUSE_API_KEY", "")
 
 
class Neo4jLoader:
    """Domain-agnostic loader: CSV / JSON / API → Neo4j property graph."""
 
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        print("[M1] Connected to Neo4j")
 
    def close(self):
        self.driver.close()
 
    # ------------------------------------------------------------------
    # 1a. FB15k-237 benchmark knowledge graph
    # ------------------------------------------------------------------
    def load_fb15k237(self, train_path: str):
        """Load FB15k-237 triples (head, relation, tail) into Neo4j."""
        count = 0
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            with open(train_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 3:
                        continue
                    head, relation, tail = parts
                    rel_label = relation.replace("/", "_").replace(".", "_").strip("_")
                    session.run(
                        """
                        MERGE (h:Entity {name: $head})
                        MERGE (t:Entity {name: $tail})
                        MERGE (h)-[:RELATION {type: $relation, label: $rel_label}]->(t)
                        """,
                        head=head, tail=tail,
                        relation=relation, rel_label=rel_label
                    )
                    count += 1
                    if count % 10000 == 0:
                        print(f"[M1] FB15k-237: {count} triples loaded...")
        print(f"[M1] FB15k-237 complete: {count} triples → Neo4j")
        return count
 
    # ------------------------------------------------------------------
    # 1b. Elliptic Bitcoin financial fraud dataset
    # ------------------------------------------------------------------
    def load_elliptic(self, features_path: str, edges_path: str, classes_path: str,
                       batch_size: int = 2000):
        """
        Load Elliptic Bitcoin transaction graph.
 
        FIXED (vs original): the original version ran one individual Neo4j
        write transaction PER ROW with no progress reporting, which for
        ~200k nodes and ~230k edges meant nothing printed until the very
        end and the whole load could take 1-2+ hours. This version uses
        UNWIND-based batched writes (batch_size rows per transaction) and
        prints progress every batch, which is both much faster and
        immediately visible in the terminal.
        """
        # Load class labels
        labels = {}
        with open(classes_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["txId"]] = row["class"]
 
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Transaction) REQUIRE t.txId IS UNIQUE")
 
            # ---- Load nodes with features, batched ----
            count_nodes = 0
            batch = []
            with open(features_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    tx_id    = row[0]
                    label    = labels.get(tx_id, "unknown")
                    is_fraud = (label == "1")
                    batch.append({"txId": tx_id, "label": label, "isFraud": is_fraud})
                    count_nodes += 1
 
                    if len(batch) >= batch_size:
                        session.run(
                            """
                            UNWIND $rows AS row
                            MERGE (t:Transaction {txId: row.txId})
                            SET t.label = row.label, t.isFraud = row.isFraud
                            """,
                            rows=batch
                        )
                        print(f"[M1] Elliptic: {count_nodes} transactions loaded...")
                        batch = []
 
                if batch:
                    session.run(
                        """
                        UNWIND $rows AS row
                        MERGE (t:Transaction {txId: row.txId})
                        SET t.label = row.label, t.isFraud = row.isFraud
                        """,
                        rows=batch
                    )
                    print(f"[M1] Elliptic: {count_nodes} transactions loaded (final batch)...")
 
            # ---- Load edges, batched ----
            count_edges = 0
            batch = []
            with open(edges_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    batch.append({"src": row["txId1"], "dst": row["txId2"]})
                    count_edges += 1
 
                    if len(batch) >= batch_size:
                        session.run(
                            """
                            UNWIND $rows AS row
                            MATCH (a:Transaction {txId: row.src})
                            MATCH (b:Transaction {txId: row.dst})
                            MERGE (a)-[:TRANSFERS_TO]->(b)
                            """,
                            rows=batch
                        )
                        print(f"[M1] Elliptic: {count_edges} edges loaded...")
                        batch = []
 
                if batch:
                    session.run(
                        """
                        UNWIND $rows AS row
                        MATCH (a:Transaction {txId: row.src})
                        MATCH (b:Transaction {txId: row.dst})
                        MERGE (a)-[:TRANSFERS_TO]->(b)
                        """,
                        rows=batch
                    )
                    print(f"[M1] Elliptic: {count_edges} edges loaded (final batch)...")
 
        print(f"[M1] Elliptic complete: {count_nodes} transactions, {count_edges} edges → Neo4j")
        return count_nodes, count_edges
 
    # ------------------------------------------------------------------
    # 1c. Companies House UK (live government API)
    # ------------------------------------------------------------------
    def load_companies_house(self, api_key: str, company_numbers: list):
        """Fetch company data from Companies House UK API and load into Neo4j."""
        base_url = "https://api.company-information.service.gov.uk"
        auth = (api_key, "")  # FIXED: Companies House needs HTTP Basic Auth (api_key, blank password), not a raw header
        loaded   = 0
 
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.number IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Director) REQUIRE d.name IS UNIQUE")
 
            for number in company_numbers:
                try:
                    r = requests.get(f"{base_url}/company/{number}", auth=auth, timeout=10)
                    if r.status_code != 200:
                        print(f"[M1] Companies House: {number} returned status {r.status_code}, skipping")
                        continue
                    data = r.json()
                    session.run(
                        """
                        MERGE (c:Company {number: $number})
                        SET c.name = $name,
                            c.status = $status,
                            c.type = $type,
                            c.incorporatedOn = $inc
                        """,
                        number=number,
                        name=data.get("company_name", ""),
                        status=data.get("company_status", ""),
                        type=data.get("type", ""),
                        inc=data.get("date_of_creation", "")
                    )
 
                    # Fetch officers (directors)
                    r2 = requests.get(f"{base_url}/company/{number}/officers", auth=auth, timeout=10)
                    if r2.status_code == 200:
                        for officer in r2.json().get("items", []):
                            if officer.get("officer_role") in ("director", "secretary"):
                                name = officer.get("name", "")
                                session.run(
                                    """
                                    MERGE (d:Director {name: $name})
                                    MERGE (c:Company {number: $number})
                                    MERGE (d)-[:DIRECTOR_OF {role: $role}]->(c)
                                    """,
                                    name=name,
                                    number=number,
                                    role=officer.get("officer_role", "director")
                                )
                    loaded += 1
                    print(f"[M1] Companies House: loaded {number}")
                except Exception as e:
                    print(f"[M1] Error loading {number}: {e}")
 
        print(f"[M1] Companies House: {loaded} companies → Neo4j")
        return loaded
 
    # ------------------------------------------------------------------
    # 1d. Generic CSV loader (any domain)
    # ------------------------------------------------------------------
    def load_csv(self, path: str, node_label: str = "Entity",
                 id_col: str = "id", rel_col: str = None, target_col: str = None):
        """Load any CSV file into Neo4j as nodes (and optional relationships)."""
        count = 0
        with self.driver.session() as session:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    props = {k: v for k, v in row.items()}
                    session.run(
                        f"MERGE (n:{node_label} {{id: $id}}) SET n += $props",
                        id=row.get(id_col, str(count)), props=props
                    )
                    if rel_col and target_col and row.get(rel_col) and row.get(target_col):
                        session.run(
                            f"""
                            MATCH (a:{node_label} {{id: $src}})
                            MATCH (b:{node_label} {{id: $dst}})
                            MERGE (a)-[:RELATED_TO]->(b)
                            """,
                            src=row[id_col], dst=row[target_col]
                        )
                    count += 1
        print(f"[M1] CSV loaded: {count} rows → Neo4j as {node_label}")
        return count
 
    # ------------------------------------------------------------------
    # 1e. Stats check
    # ------------------------------------------------------------------
    def get_stats(self):
        with self.driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rels  = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"[M1] Graph stats — Nodes: {nodes}, Relationships: {rels}")
        return nodes, rels
 
 
if __name__ == "__main__":
    loader = Neo4jLoader()
 
    # ------------------------------------------------------------------
    # 1. Load FB15k-237 — knowledge graph benchmark (for GNN training in Module 2)
    # ------------------------------------------------------------------
    print("\n[M1] === Loading FB15k-237 ===")
    loader.load_fb15k237("data/fb15k237/train.txt")
 
    # ------------------------------------------------------------------
    # 2. Load Elliptic Bitcoin — financial fraud dataset
    #    Downloaded from Kaggle, placed in data/elliptic/
    # ------------------------------------------------------------------
    print("\n[M1] === Loading Elliptic Bitcoin ===")
    loader.load_elliptic(
        features_path="data/elliptic/elliptic_txs_features.csv",
        edges_path="data/elliptic/elliptic_txs_edgelist.csv",
        classes_path="data/elliptic/elliptic_txs_classes.csv"
    )
 
    # ------------------------------------------------------------------
    # 3. Load Companies House UK — live government API
    #    20 real UK companies (FIX: missing comma after "01003142" corrected —
    #    that bug was silently merging two entries into one bad company number)
    # ------------------------------------------------------------------
    print("\n[M1] === Loading Companies House UK ===")
    if COMPANIES_HOUSE_API_KEY:
        loader.load_companies_house(
            api_key=COMPANIES_HOUSE_API_KEY,
            company_numbers=[
                "00000006",   # Marine and General Mutual Life Assurance Society
                "00445790",   # Tesco PLC
                "00102498",   # BP p.l.c.
                "01026167",   # Barclays Bank UK PLC
                "00014259",   # HSBC Holdings
                "SC171417",   # Sainsbury's
                "NF001553",   # Marks and Spencer P.L.C.
                "01471587",   # Vodafone
                "00041424",   # Unilever
                "01003142",   # Rolls-Royce                <- FIXED: comma added here
                "NF002699",   # British Airways
                "16463192",   # BT Group Properties Ltd
                "10991462",   # National Grid Electricity Distribution Holdings Ltd
                "17245532",   # Rio Tinto Bauxite Atlantic Ltd
                "12215835",   # GlaxoSmithKline Ltd
                "02723534",   # AstraZeneca PLC
                "00023307",   # Diageo
                "01470151",   # BAE Systems
                "00527217",   # Reckitt Benckiser
                "04083914",   # Compass Group
            ]
        )
    else:
        print("[M1] Skipped — COMPANIES_HOUSE_API_KEY not set in .env")
 
    # ------------------------------------------------------------------
    # Final stats
    # ------------------------------------------------------------------
    nodes, rels = loader.get_stats()
    print(f"\n[M1] Graph ready: {nodes} nodes, {rels} relationships")
    loader.close()