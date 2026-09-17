# =============================================================================
# MODULE 2b — COMBINED HETEROGENEOUS GRAPH NEURAL NETWORK
# Trains a SINGLE GraphSAGE-style model jointly across all three loaded
# datasets, using PyTorch Geometric's heterogeneous graph support:
#   - FB15k-237      : (Entity)-[RELATION]->(Entity)
#   - Elliptic Bitcoin: (Transaction)-[TRANSFERS_TO]->(Transaction)
#   - Companies House : (Director)-[DIRECTOR_OF]->(Company)
#
# This is a SEPARATE experiment from models/gnn.py (which remains the
# primary, FB15k-237-only relation-aware model reported as the main
# Module 2 result). This combined model demonstrates that the same
# neural architecture generalises across heterogeneous schemas, at the
# cost of the relation-aware (DistMult) scoring used for FB15k-237
# specifically — this version uses a simpler shared dot-product decoder
# across all edge types, since the three datasets do not share a
# meaningful joint relation vocabulary the way FB15k-237's 237 relation
# types do internally.
#
# HONEST NOTE ON DATASET SIZE IMBALANCE: FB15k-237 (~14.5k nodes,
# ~272k edges) and Elliptic Bitcoin (~204k nodes, ~234k edges) are
# large; Companies House UK (~20 companies + directors, ~20-40 edges)
# is tiny by comparison. The DIRECTOR_OF metrics below should be read
# as illustrative rather than statistically robust, given the very
# small sample size for that edge type.
# =============================================================================
 
import os
import warnings
 
# Suppress the PyTorch Geometric UserWarning about node types (e.g.
# 'Director') that never appear as an edge destination and so are not
# updated by message passing. This is expected and already handled
# correctly via the per-node-type Linear fallback in HeteroGraphSAGE
# below (self.lin1 / self.lin2), so the warning is cosmetic only.
warnings.filterwarnings(
    "ignore",
    message="There exist node types .* whose representations do not get updated"
)
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData
import numpy as np
from neo4j import GraphDatabase
 
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Kailas@123")
 
EMBED_DIM  = 160
HIDDEN_DIM = 320
OUT_DIM    = 160
 
 
# ------------------------------------------------------------------
# Load all three datasets from Neo4j into a single HeteroData object.
# Each node type gets its own deterministic index mapping; each edge
# type gets its own edge_index tensor.
# ------------------------------------------------------------------
def load_hetero_graph_from_neo4j():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    id_maps = {"Entity": {}, "Transaction": {}, "Company": {}, "Director": {}}
    edges   = {
        ("Entity", "RELATION", "Entity"):           ([], []),
        ("Transaction", "TRANSFERS_TO", "Transaction"): ([], []),
        ("Director", "DIRECTOR_OF", "Company"):      ([], []),
    }
 
    relation_to_idx  = {}
    relation_edge_types = []  # parallel array to RELATION src/dst lists
 
    with driver.session() as session:
        # ---- Node id maps (deterministic order) ----
        for rec in session.run("MATCH (e:Entity) RETURN e.name AS k ORDER BY e.name"):
            id_maps["Entity"][rec["k"]] = len(id_maps["Entity"])
        for rec in session.run("MATCH (t:Transaction) RETURN t.txId AS k ORDER BY t.txId"):
            id_maps["Transaction"][rec["k"]] = len(id_maps["Transaction"])
        for rec in session.run("MATCH (c:Company) RETURN c.number AS k ORDER BY c.number"):
            id_maps["Company"][rec["k"]] = len(id_maps["Company"])
        for rec in session.run("MATCH (d:Director) RETURN d.name AS k ORDER BY d.name"):
            id_maps["Director"][rec["k"]] = len(id_maps["Director"])
 
        print(f"[M2b] Node counts — Entity: {len(id_maps['Entity'])}, "
              f"Transaction: {len(id_maps['Transaction'])}, "
              f"Company: {len(id_maps['Company'])}, "
              f"Director: {len(id_maps['Director'])}")
 
        # ---- Edges: Entity-RELATION->Entity (NOW also capturing r.type
        # for relation-aware DistMult scoring, same fix that improved
        # the standalone FB15k model's MRR by 94%) ----
        src, dst = edges[("Entity", "RELATION", "Entity")]
        for rec in session.run(
            "MATCH (h:Entity)-[r:RELATION]->(t:Entity) RETURN h.name AS h, t.name AS t, r.type AS rtype"
        ):
            if rec["h"] in id_maps["Entity"] and rec["t"] in id_maps["Entity"]:
                rtype = rec["rtype"] or "UNKNOWN"
                if rtype not in relation_to_idx:
                    relation_to_idx[rtype] = len(relation_to_idx)
                src.append(id_maps["Entity"][rec["h"]])
                dst.append(id_maps["Entity"][rec["t"]])
                relation_edge_types.append(relation_to_idx[rtype])
 
        # ---- Edges: Transaction-TRANSFERS_TO->Transaction ----
        src, dst = edges[("Transaction", "TRANSFERS_TO", "Transaction")]
        for rec in session.run(
            "MATCH (a:Transaction)-[:TRANSFERS_TO]->(b:Transaction) RETURN a.txId AS a, b.txId AS b"
        ):
            if rec["a"] in id_maps["Transaction"] and rec["b"] in id_maps["Transaction"]:
                src.append(id_maps["Transaction"][rec["a"]])
                dst.append(id_maps["Transaction"][rec["b"]])
 
        # ---- Edges: Director-DIRECTOR_OF->Company ----
        src, dst = edges[("Director", "DIRECTOR_OF", "Company")]
        for rec in session.run(
            "MATCH (d:Director)-[:DIRECTOR_OF]->(c:Company) RETURN d.name AS d, c.number AS c"
        ):
            if rec["d"] in id_maps["Director"] and rec["c"] in id_maps["Company"]:
                src.append(id_maps["Director"][rec["d"]])
                dst.append(id_maps["Company"][rec["c"]])
 
    driver.close()
 
    data = HeteroData()
    torch.manual_seed(42)
    for ntype, idmap in id_maps.items():
        n = max(len(idmap), 1)  # avoid zero-sized tensors if a type is empty
        data[ntype].x = torch.randn(n, EMBED_DIM)
        data[ntype].num_nodes = n
 
    edge_counts = {}
    for etype, (src, dst) in edges.items():
        if len(src) == 0:
            # placeholder empty edge index so the model doesn't crash
            # if a dataset/edge-type is missing
            data[etype].edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            data[etype].edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_counts[etype] = len(src)
 
    print(f"[M2b] Edge counts — RELATION: {edge_counts[('Entity','RELATION','Entity')]}, "
          f"TRANSFERS_TO: {edge_counts[('Transaction','TRANSFERS_TO','Transaction')]}, "
          f"DIRECTOR_OF: {edge_counts[('Director','DIRECTOR_OF','Company')]}")
    print(f"[M2b] RELATION distinct relation types: {len(relation_to_idx)}")
 
    rel_edge_type_tensor = torch.tensor(relation_edge_types, dtype=torch.long) \
        if relation_edge_types else torch.empty((0,), dtype=torch.long)
 
    return data, id_maps, relation_to_idx, rel_edge_type_tensor
 
 
# ------------------------------------------------------------------
# Heterogeneous GraphSAGE encoder: one SAGEConv per edge type per
# layer, aggregated per node type via HeteroConv.
# ------------------------------------------------------------------
class HeteroGraphSAGE(torch.nn.Module):
    """
    FIXED: a node type that never appears as the DESTINATION of any
    edge type (e.g. 'Director', which is only ever the source of
    DIRECTOR_OF edges) never receives a message-passing update from
    HeteroConv, so it is silently dropped from HeteroConv's output
    dict. Passed into the next layer, that missing key becomes None
    and crashes SAGEConv ('NoneType has no attribute dim').
 
    Fix: every node type also has its own per-layer Linear fallback.
    After each HeteroConv call, any node type NOT present in the
    output (because it received no incoming messages this layer) is
    filled in via its own linear projection of its previous-layer
    representation instead, keeping every node type's embedding at a
    consistent dimensionality across both layers.
    """
    def __init__(self, node_types, edge_types, hidden=HIDDEN_DIM, out=OUT_DIM):
        super().__init__()
        self.conv1 = HeteroConv({
            etype: SAGEConv((-1, -1), hidden) for etype in edge_types
        }, aggr="sum")
        self.conv2 = HeteroConv({
            etype: SAGEConv((-1, -1), out) for etype in edge_types
        }, aggr="sum")
        self.node_types = node_types
        # Fallback per-node-type projections for types with no incoming edges
        self.lin1 = torch.nn.ModuleDict({nt: torch.nn.LazyLinear(hidden) for nt in node_types})
        self.lin2 = torch.nn.ModuleDict({nt: torch.nn.LazyLinear(out) for nt in node_types})
 
    def forward(self, x_dict, edge_index_dict):
        out1 = self.conv1(x_dict, edge_index_dict)
        for nt in self.node_types:
            if nt not in out1:
                out1[nt] = self.lin1[nt](x_dict[nt])
        out1 = {k: F.relu(v) for k, v in out1.items()}
        out1 = {k: F.dropout(v, p=0.3, training=self.training) for k, v in out1.items()}
 
        out2 = self.conv2(out1, edge_index_dict)
        for nt in self.node_types:
            if nt not in out2:
                out2[nt] = self.lin2[nt](out1[nt])
        return out2
 
    @staticmethod
    def decode(z_src, z_dst, edge_index):
        h = z_src[edge_index[0]]
        t = z_dst[edge_index[1]]
        return (h * t).sum(dim=-1)
 
 
def negative_edges(num_src, num_dst, num_neg):
    neg_src = torch.randint(0, max(num_src, 1), (num_neg,))
    neg_dst = torch.randint(0, max(num_dst, 1), (num_neg,))
    return torch.stack([neg_src, neg_dst], dim=0)
 
 
# ------------------------------------------------------------------
# Train the combined heterogeneous model
# ------------------------------------------------------------------
def train_combined_gnn(epochs: int = 600, lr: float = 0.01,
                        save_path: str = "models/graphsage_combined.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[M2b] Training on: {device}")
 
    data, id_maps, relation_to_idx, rel_edge_types = load_hetero_graph_from_neo4j()
    data = data.to(device)
    rel_edge_types = rel_edge_types.to(device)
 
    node_types = list(id_maps.keys())
    edge_types = [et for et in data.edge_types if data[et].edge_index.size(1) > 0]
    print(f"[M2b] Active edge types for training: {edge_types}")
 
    model = HeteroGraphSAGE(node_types, data.edge_types).to(device)
 
    # Relation embedding table for RELATION edges specifically (FIX: this
    # is the same relation-aware DistMult scoring that improved the
    # standalone FB15k model's MRR by 94% — added here for the RELATION
    # edge type only, since TRANSFERS_TO and DIRECTOR_OF each have a
    # single relation type and do not need it).
    num_relations = max(len(relation_to_idx), 1)
    r_emb = torch.nn.Embedding(num_relations, OUT_DIM).to(device)
 
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(r_emb.parameters()), lr=lr, weight_decay=1e-5
    )
    # Decay LR gradually so training stabilises in later epochs instead
    # of oscillating (earlier runs showed loss occasionally increasing
    # between logged epochs, e.g. epoch 220->240).
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
 
 
    best_loss = float("inf")
 
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
 
        x_dict = {k: data[k].x for k in node_types}
        z_dict = model(x_dict, data.edge_index_dict)
 
        total_loss = 0.0
        for etype in edge_types:
            src_type, rel_name, dst_type = etype
            ei = data[etype].edge_index
            if ei.size(1) == 0:
                continue
 
            if rel_name == "RELATION":
                # Relation-aware DistMult scoring: sum(h * r * t)
                h = z_dict[src_type][ei[0]]
                t = z_dict[dst_type][ei[1]]
                r = r_emb(rel_edge_types)
                pos_score = (h * r * t).sum(dim=-1)
 
                neg_ei = negative_edges(data[src_type].num_nodes, data[dst_type].num_nodes,
                                         pos_score.size(0)).to(device)
                h_neg = z_dict[src_type][neg_ei[0]]
                t_neg = z_dict[dst_type][neg_ei[1]]
                neg_score = (h_neg * r * t_neg).sum(dim=-1)
            else:
                pos_score = model.decode(z_dict[src_type], z_dict[dst_type], ei)
                neg_ei = negative_edges(data[src_type].num_nodes, data[dst_type].num_nodes,
                                         pos_score.size(0)).to(device)
                neg_score = model.decode(z_dict[src_type], z_dict[dst_type], neg_ei)
 
            pos_label = torch.ones(pos_score.size(0), device=device)
            neg_label = torch.zeros(neg_score.size(0), device=device)
            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([pos_label, neg_label])
            total_loss = total_loss + F.binary_cross_entropy_with_logits(scores, labels)
 
        total_loss.backward()
        optimizer.step()
        scheduler.step()
 
        loss_val = total_loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "model_state":     model.state_dict(),
                "r_emb_state":     r_emb.state_dict(),
                "relation_to_idx": relation_to_idx,
                "id_maps":         id_maps,
                "node_types":      node_types,
                "edge_types":      data.edge_types,
                "hidden":          HIDDEN_DIM,
                "out_dim":         OUT_DIM,
                "embed_dim":       EMBED_DIM,
            }, save_path)
 
        if epoch % 20 == 0:
            print(f"[M2b] Epoch {epoch:03d} | Combined Loss: {loss_val:.4f} | Best: {best_loss:.4f}")
 
    print(f"[M2b] Training complete. Combined model saved -> {save_path}")
    return model, r_emb, rel_edge_types, data, id_maps
 
 
# ------------------------------------------------------------------
# Evaluate MRR/Hits@k PER EDGE TYPE (separately, since datasets are
# not comparable to each other — reporting them jointly would be
# misleading given the size imbalance).
# ------------------------------------------------------------------
def evaluate_combined(model, r_emb, rel_edge_types, data, k_list=(1, 3, 10)):
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
 
    with torch.no_grad():
        x_dict = {k: data[k].x for k in data.node_types}
        z_dict = model(x_dict, data.edge_index_dict)
 
    results = {}
    for etype in data.edge_types:
        src_type, rel_name, dst_type = etype
        ei = data[etype].edge_index
        if ei.size(1) == 0:
            print(f"[M2b] {rel_name}: no edges, skipped")
            continue
 
        num_edges = ei.size(1)
        sample    = min(500, num_edges)
        indices   = torch.randperm(num_edges)[:sample]
        test_ei   = ei[:, indices]
 
        z_src, z_dst = z_dict[src_type], z_dict[dst_type]
        ranks = []
        with torch.no_grad():
            if rel_name == "RELATION":
                # Relation-aware evaluation: score every candidate tail
                # against the TRUE relation of each test edge (matches
                # the DistMult scoring used during training).
                test_rel_ids = rel_edge_types[indices]
                for i in range(sample):
                    s   = test_ei[0, i].item()
                    d   = test_ei[1, i].item()
                    rv  = r_emb.weight[test_rel_ids[i]]
                    src_emb = (z_src[s] * rv).unsqueeze(0)
                    scores  = (src_emb * z_dst).sum(dim=-1)
                    true_score = scores[d].item()
                    rank = (scores > true_score).sum().item() + 1
                    ranks.append(rank)
            else:
                for i in range(sample):
                    s = test_ei[0, i].item()
                    d = test_ei[1, i].item()
                    scores = (z_src[s].unsqueeze(0) * z_dst).sum(dim=-1)
                    true_score = scores[d].item()
                    rank = (scores > true_score).sum().item() + 1
                    ranks.append(rank)
 
        ranks = np.array(ranks)
        mrr   = float(np.mean(1.0 / ranks))
        hits  = {k: float(np.mean(ranks <= k)) for k in k_list}
 
        print(f"[M2b] {rel_name} ({src_type}->{dst_type}, n={sample}): "
              f"MRR={mrr:.4f}, " + ", ".join(f"Hits@{k}={v:.4f}" for k, v in hits.items()))
        results[rel_name] = {"mrr": mrr, "hits": hits, "n": sample}
 
    return results
 
 
# ------------------------------------------------------------------
# Link prediction for a single entity, using the COMBINED heterogeneous
# model. Used by Module 3's GNNPredict fallback tool when a direct
# Cypher query returns no result.
#
# Unlike the earlier single-dataset predict_links(), this version
# first determines WHICH of the four node types (Entity, Transaction,
# Company, Director) the given entity_name belongs to by checking
# each dataset's id_map, then predicts links only against the node
# type(s) that entity_name's dataset is actually connected to in the
# graph schema -- so an FB15k-237 entity is scored against other
# Entity nodes, a Transaction id against other Transaction nodes, and
# a Director/Company name against its counterpart type, rather than
# meaninglessly comparing embeddings across unrelated datasets.
# ------------------------------------------------------------------
def predict_links(entity_name: str, top_k: int = 5,
                   model_path: str = "models/graphsage_combined.pt"):
    # weights_only=False: PyTorch 2.6+ defaults to weights_only=True,
    # which rejects this checkpoint because it contains custom
    # parameter types (UninitializedParameter from HeteroGraphSAGE's
    # LazyLinear fallback layers). Safe here since this is our own
    # locally-trained checkpoint, not a downloaded/untrusted file.
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    id_maps    = checkpoint["id_maps"]
    node_types = checkpoint["node_types"]
    edge_types = checkpoint["edge_types"]
 
    # Work out which dataset/node type this entity belongs to
    src_type = None
    for nt, idmap in id_maps.items():
        if entity_name in idmap:
            src_type = nt
            break
    if src_type is None:
        return []  # entity not found in any of the three datasets
 
    # Rebuild the graph (deterministic, seeded) and the trained model
    data, _, _, _ = load_hetero_graph_from_neo4j()
    model = HeteroGraphSAGE(node_types, edge_types)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
 
    with torch.no_grad():
        x_dict = {nt: data[nt].x for nt in node_types}
        z_dict = model(x_dict, data.edge_index_dict)
 
    src_idx = id_maps[src_type][entity_name]
    z_src   = z_dict[src_type][src_idx]
 
    # Only compare against node types this src_type actually connects
    # to in the schema (e.g. Entity<->Entity, Director<->Company),
    # never across unrelated datasets.
    candidate_types = set()
    for (s, _rel, d) in edge_types:
        if s == src_type:
            candidate_types.add(d)
        if d == src_type:
            candidate_types.add(s)
    if not candidate_types:
        candidate_types = {src_type}
 
    idx_to_name = {nt: {v: k for k, v in idmap.items()} for nt, idmap in id_maps.items()}
    results = []
    for tgt_type in candidate_types:
        z_tgt  = z_dict[tgt_type]
        scores = (z_src.unsqueeze(0) * z_tgt).sum(dim=-1)
        k      = min(top_k, scores.size(0))
        top_scores, top_idx = scores.topk(k)
        for idx, score in zip(top_idx.tolist(), top_scores.tolist()):
            name = idx_to_name[tgt_type].get(idx, f"{tgt_type}_{idx}")
            if name != entity_name:
                results.append({
                    "entity": name,
                    "type":   tgt_type,
                    "score":  round(score, 4)
                })
 
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]
 
 
if __name__ == "__main__":
    model, r_emb, rel_edge_types, data, id_maps = train_combined_gnn(epochs=600)
    print("\n[M2b] === Per-edge-type evaluation (datasets are NOT directly comparable) ===")
    evaluate_combined(model, r_emb, rel_edge_types, data)