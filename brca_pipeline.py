#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RTK–NRTK Redundancy Pipeline — BRCA focused, SQLite backend
============================================================
Data source: SQLite (Data/tcga_brca.db)
Embeddings:  ESM2 (HuggingFace, batched)
Vectors:     Qdrant (local)
"""

import logging
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from scipy.stats import fisher_exact
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DB_PATH       = Path("Data/tcga_brca.db")
DATA_DIR      = Path("Data/")
EMBED_MODEL   = "facebook/esm2_t33_650M_UR50D"
EMB_CACHE     = DATA_DIR / "esm2_embeddings.npy"
EMBED_BATCH   = 4
THRESH        = 0.75
WEIGHTS       = dict(emb=1/3, expr=1/3, mut=1/3)
CANCER_TYPE   = "BRCA"

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA FROM SQLITE
# ─────────────────────────────────────────────────────────────
def load_data():
    log.info(f"Loading BRCA data from {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)

    # Expression: pivot to genes × samples
    expr_long = pd.read_sql(
        "SELECT sample_id, gene, expression FROM expression_raw WHERE cancer_type='BRCA'",
        conn
    )
    expr = expr_long.pivot(index="gene", columns="sample_id", values="expression")
    expr = expr.dropna(how="all")
    log.info(f"  Expression: {expr.shape[0]} genes × {expr.shape[1]} samples")

    # Mutation: pivot to genes × samples
    mut_long = pd.read_sql(
        "SELECT sample_id, gene, mutated FROM mutation_raw WHERE cancer_type='BRCA'",
        conn
    )
    mut = mut_long.pivot(index="gene", columns="sample_id", values="mutated").fillna(0).astype(int)
    log.info(f"  Mutations:  {mut.shape[0]} genes × {mut.shape[1]} samples")

    # Sequences
    seqs = pd.read_sql(
        "SELECT gene, sequence FROM kinase_meta WHERE sequence IS NOT NULL",
        conn
    ).set_index("gene")
    log.info(f"  Sequences:  {len(seqs)} kinases")

    # RTK/NRTK pairs
    pairs = pd.read_sql("SELECT * FROM rtk_nrtk_pairs", conn)
    conn.close()

    # Get RTK/NRTK sets — handle both upper and lower column names
    rtk_col  = "RTK"  if "RTK"  in pairs.columns else "rtk"
    nrtk_col = "NRTK" if "NRTK" in pairs.columns else "nrtk"
    rtk_set  = set(pairs[rtk_col])
    nrtk_set = set(pairs[nrtk_col])

    # Common genes across all data types
    genes = sorted(set(expr.index) & set(mut.index) & set(seqs.index))
    log.info(f"  Common genes: {len(genes)}")

    expr = expr.loc[genes]
    mut  = mut.loc[genes]
    seqs = seqs.loc[genes]

    kinases = pd.DataFrame({
        "gene": genes,
        "type": ["RTK" if g in rtk_set else "NRTK" for g in genes],
    })
    log.info(f"  RTKs: {kinases[kinases.type=='RTK'].shape[0]}, "
             f"NRTKs: {kinases[kinases.type=='NRTK'].shape[0]}")

    return genes, expr, mut, seqs, kinases


# ─────────────────────────────────────────────────────────────
# 2. ESM-2 EMBEDDINGS
# ─────────────────────────────────────────────────────────────
def generate_embeddings(sequences: list[str]) -> np.ndarray:
    if EMB_CACHE.exists():
        log.info(f"Loading cached embeddings: {EMB_CACHE}")
        emb = np.load(EMB_CACHE)
        if emb.shape[0] == len(sequences):
            return emb
        log.warning("Cache size mismatch — regenerating.")

    log.info(f"Loading ESM2 model: {EMBED_MODEL}")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"  Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model     = AutoModel.from_pretrained(EMBED_MODEL).to(device)
    model.eval()

    def _embed_batch(seqs):
        tokens = tokenizer(
            seqs, return_tensors="pt", add_special_tokens=True,
            padding=True, truncation=True, max_length=1024,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            out = model(**tokens)
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        reps = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        return reps.cpu().numpy()

    results = []
    for i in tqdm(range(0, len(sequences), EMBED_BATCH), desc="ESM2 embeddings"):
        results.append(_embed_batch(sequences[i : i + EMBED_BATCH]))

    embeddings = np.vstack(results)
    np.save(EMB_CACHE, embeddings)
    log.info(f"Embeddings saved → {EMB_CACHE}  shape={embeddings.shape}")
    return embeddings


# ─────────────────────────────────────────────────────────────
# 3. SIMILARITY MATRICES
# ─────────────────────────────────────────────────────────────
def embedding_similarity(embeddings: np.ndarray) -> np.ndarray:
    log.info("Computing embedding similarity (cosine)...")
    return cosine_similarity(embeddings)


def expression_similarity(expr: pd.DataFrame) -> np.ndarray:
    log.info("Computing expression similarity (Pearson)...")
    arr  = expr.values.astype(float)
    stds = arr.std(axis=1, keepdims=True)
    safe = np.where(stds == 0, 1.0, stds)
    z    = (arr - arr.mean(axis=1, keepdims=True)) / safe
    corr = z @ z.T / arr.shape[1]
    np.fill_diagonal(corr, 1.0)
    return corr


def mutation_similarity(mut: pd.DataFrame) -> np.ndarray:
    log.info("Computing mutation co-occurrence (Fisher exact)...")
    arr = mut.values
    n   = arr.shape[0]
    scores = np.zeros((n, n), dtype=float)
    for i in tqdm(range(n), desc="Fisher exact"):
        for j in range(i, n):
            a = int(((arr[i]==1) & (arr[j]==1)).sum())
            b = int(((arr[i]==1) & (arr[j]==0)).sum())
            c = int(((arr[i]==0) & (arr[j]==1)).sum())
            d = int(((arr[i]==0) & (arr[j]==0)).sum())
            _, p = fisher_exact([[a,b],[c,d]])
            v = -np.log10(p + 1e-300)
            scores[i,j] = scores[j,i] = v
    return scores


def normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if np.isclose(lo, hi):
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def composite_redundancy(sim_emb, sim_expr, sim_mut) -> np.ndarray:
    log.info("Computing composite redundancy scores...")
    return (
        WEIGHTS["emb"]  * normalize(sim_emb)  +
        WEIGHTS["expr"] * normalize(sim_expr) +
        WEIGHTS["mut"]  * normalize(sim_mut)
    )


# ─────────────────────────────────────────────────────────────
# 4. SAVE RESULTS BACK TO SQLITE
# ─────────────────────────────────────────────────────────────
def save_results(genes, redundancy, sim_emb, sim_expr, sim_mut):
    log.info("Saving redundancy scores to SQLite...")
    conn = sqlite3.connect(DB_PATH)

    rows = []
    for i, g1 in enumerate(genes):
        for j, g2 in enumerate(genes):
            if j <= i:
                continue
            rows.append({
                "gene1":           g1,
                "gene2":           g2,
                "score_emb":       float(sim_emb[i,j]),
                "score_expr":      float(sim_expr[i,j]),
                "score_mut":       float(sim_mut[i,j]),
                "score_composite": float(redundancy[i,j]),
            })

    df = pd.DataFrame(rows)
    df.to_sql("redundancy_scores", conn, if_exists="replace", index=False)
    log.info(f"  Saved {len(df)} gene pairs to redundancy_scores")
    conn.close()


# ─────────────────────────────────────────────────────────────
# 5. NETWORK + VISUALISATION
# ─────────────────────────────────────────────────────────────
def build_network(genes, kinases, redundancy) -> nx.Graph:
    log.info(f"Building redundancy network (threshold={THRESH})...")
    G = nx.Graph()
    for _, row in kinases.iterrows():
        G.add_node(row["gene"], type=row["type"])
    for i, g1 in enumerate(genes):
        for j, g2 in enumerate(genes):
            if j <= i:
                continue
            if redundancy[i,j] >= THRESH:
                G.add_edge(g1, g2, weight=float(redundancy[i,j]))
    log.info(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G


def _draw_subgraph(subgraph, title, color, outpath):
    if len(subgraph.nodes()) == 0:
        log.warning(f"  Skipping {title} — no nodes")
        return
    fig, ax = plt.subplots(figsize=(10, 10))
    pos     = nx.spring_layout(subgraph, seed=42, k=2)
    degrees = dict(subgraph.degree())
    sizes   = [400 + 200 * degrees.get(n, 0) for n in subgraph.nodes()]
    widths  = [subgraph[u][v].get("weight", 0.5) * 3 for u, v in subgraph.edges()]
    nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_color=color,
                           node_size=sizes, alpha=0.85)
    nx.draw_networkx_edges(subgraph, pos, ax=ax, width=widths,
                           alpha=0.5, edge_color="#444444")
    nx.draw_networkx_labels(subgraph, pos, ax=ax, font_size=8,
                            font_weight="bold", font_color="white")
    ax.set_title(f"BRCA — {title}", fontsize=15, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {outpath}")


def visualise(G, kinases, redundancy, genes):
    RTK_COLOR  = "#E63946"   # red  for RTKs
    NRTK_COLOR = "#457B9D"   # blue for NRTKs
    EDGE_COLOR = "#A8DADC"

    rtks  = kinases[kinases.type=="RTK"]["gene"].tolist()
    nrtks = kinases[kinases.type=="NRTK"]["gene"].tolist()

    _draw_subgraph(G.subgraph(rtks).copy(),
                   "RTK–RTK Redundancy", RTK_COLOR,
                   DATA_DIR / "brca_network_rtk_rtk.png")

    _draw_subgraph(G.subgraph(nrtks).copy(),
                   "NRTK–NRTK Redundancy", NRTK_COLOR,
                   DATA_DIR / "brca_network_nrtk_nrtk.png")

    # Bipartite RTK–NRTK
    cross = [(u,v) for u,v in G.edges()
             if G.nodes[u]["type"] != G.nodes[v]["type"]]

    fig, ax = plt.subplots(figsize=(12, max(12, max(len(rtks), len(nrtks)) * 0.7)))
    pos = {g: (0, i*1.5) for i, g in enumerate(rtks)}
    pos.update({g: (3, i*1.5) for i, g in enumerate(nrtks)})

    nx.draw_networkx_nodes(G, pos, nodelist=rtks,  node_color=RTK_COLOR,
                           node_size=500, ax=ax, alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=nrtks, node_color=NRTK_COLOR,
                           node_size=500, ax=ax, alpha=0.9)

    if cross:
        edge_w = [G[u][v].get("weight",0.5)*3 for u,v in cross]
        nx.draw_networkx_edges(G, pos, edgelist=cross, edge_color=EDGE_COLOR,
                               width=edge_w, ax=ax, alpha=0.7)

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                            font_weight="bold", font_color="white")

    ax.legend(handles=[
        mpatches.Patch(facecolor=RTK_COLOR,  label="RTK"),
        mpatches.Patch(facecolor=NRTK_COLOR, label="NRTK"),
    ], fontsize=11, loc="upper center", ncol=2)
    ax.set_title("BRCA — RTK–NRTK Redundancy Network", fontsize=15,
                 fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(DATA_DIR / "brca_network_rtk_nrtk.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {DATA_DIR / 'brca_network_rtk_nrtk.png'}")

    # Heatmap of top redundancy scores
    log.info("Plotting redundancy heatmap...")
    red_df = pd.DataFrame(redundancy, index=genes, columns=genes)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(red_df.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(genes)))
    ax.set_yticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=90, fontsize=7)
    ax.set_yticklabels(genes, fontsize=7)
    plt.colorbar(im, ax=ax, label="Redundancy Score")
    ax.set_title("BRCA — Kinase Redundancy Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(DATA_DIR / "brca_redundancy_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {DATA_DIR / 'brca_redundancy_heatmap.png'}")


# ─────────────────────────────────────────────────────────────
# 6. SUMMARY REPORT
# ─────────────────────────────────────────────────────────────
def print_summary(G, genes, kinases, redundancy):
    print("\n" + "="*60)
    print("  BRCA RTK–NRTK REDUNDANCY ANALYSIS — SUMMARY")
    print("="*60)

    print(f"\n  Genes analysed : {len(genes)}")
    print(f"  RTKs           : {kinases[kinases.type=='RTK'].shape[0]}")
    print(f"  NRTKs          : {kinases[kinases.type=='NRTK'].shape[0]}")
    print(f"  Network edges  : {G.number_of_edges()} (threshold={THRESH})")

    # Top 10 redundant pairs
    rows = []
    for i, g1 in enumerate(genes):
        for j, g2 in enumerate(genes):
            if j <= i:
                continue
            rows.append((g1, g2, redundancy[i,j]))

    top = sorted(rows, key=lambda x: x[2], reverse=True)[:10]
    print(f"\n  Top 10 most redundant BRCA kinase pairs:")
    print(f"  {'Gene1':<12} {'Gene2':<12} {'Score':>8}")
    print(f"  {'-'*34}")
    for g1, g2, score in top:
        print(f"  {g1:<12} {g2:<12} {score:>8.4f}")

    # Most connected nodes
    if G.number_of_edges() > 0:
        degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  Most connected kinases in BRCA network:")
        for gene, deg in degree_sorted:
            gtype = G.nodes[gene]["type"]
            print(f"  {gene:<12} ({gtype})  —  {deg} connections")

    print("\n  Output files saved to Data/:")
    print("  • brca_network_rtk_rtk.png")
    print("  • brca_network_nrtk_nrtk.png")
    print("  • brca_network_rtk_nrtk.png")
    print("  • brca_redundancy_heatmap.png")
    print("  • redundancy_scores (SQLite table)")
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # Load
    genes, expr, mut, seqs, kinases = load_data()

    # Embeddings
    embeddings = generate_embeddings(seqs["sequence"].tolist())

    # Similarities
    sim_emb  = embedding_similarity(embeddings)
    sim_expr = expression_similarity(expr)
    sim_mut  = mutation_similarity(mut)

    # Composite score
    redundancy = composite_redundancy(sim_emb, sim_expr, sim_mut)

    # Save to SQLite
    save_results(genes, redundancy, sim_emb, sim_expr, sim_mut)

    # Network
    G = build_network(genes, kinases, redundancy)

    # Visualise
    visualise(G, kinases, redundancy, genes)

    # Summary
    print_summary(G, genes, kinases, redundancy)

    log.info("BRCA pipeline complete!")


if __name__ == "__main__":
    main()
