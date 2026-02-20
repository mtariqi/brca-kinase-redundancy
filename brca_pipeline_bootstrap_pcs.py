#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RTK-NRTK Redundancy Pipeline — BRCA + Bootstrap Statistics
===========================================================
New in this version:
  - Bootstrap confidence intervals (95% CI) on all redundancy scores
  - Statistically significant edges only (p < 0.05 by permutation)
  - Enhanced visualisations: CI error bars, significance-coded heatmap,
    force-directed network with edge confidence, volcano plot
"""

import logging
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from scipy.stats import pearsonr, fisher_exact
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
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

DB_PATH        = Path("Data/tcga_brca.db")
DATA_DIR       = Path("Data/")
RESULTS_DIR    = Path("Results/")
EMBED_MODEL    = "facebook/esm2_t33_650M_UR50D"
EMB_CACHE      = DATA_DIR / "esm2_embeddings.npy"
EMBED_BATCH    = 4
THRESH         = 0.50        # main network threshold
CROSS_THRESH   = 0.40        # RTK-NRTK cross-type threshold
WEIGHTS        = dict(emb=1/3, expr=1/3, mut=1/3)
N_BOOTSTRAP    = 1000        # bootstrap iterations
CI             = 95          # confidence interval %
ALPHA          = 0.05        # significance level
RANDOM_SEED    = 42

RESULTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA FROM SQLITE
# ─────────────────────────────────────────────────────────────
def load_data():
    log.info(f"Loading BRCA data from {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)

    expr_long = pd.read_sql(
        "SELECT sample_id, gene, expression FROM expression_raw WHERE cancer_type='BRCA'",
        conn
    )
    expr = expr_long.pivot(index="gene", columns="sample_id", values="expression")
    expr = expr.dropna(how="all")
    log.info(f"  Expression: {expr.shape[0]} genes x {expr.shape[1]} samples")

    mut_long = pd.read_sql(
        "SELECT sample_id, gene, mutated FROM mutation_raw WHERE cancer_type='BRCA'",
        conn
    )
    mut = mut_long.pivot(index="gene", columns="sample_id", values="mutated").fillna(0).astype(int)
    log.info(f"  Mutations:  {mut.shape[0]} genes x {mut.shape[1]} samples")

    seqs = pd.read_sql(
        "SELECT gene, sequence FROM kinase_meta WHERE sequence IS NOT NULL",
        conn
    ).set_index("gene")
    log.info(f"  Sequences:  {len(seqs)} kinases")

    pairs   = pd.read_sql("SELECT * FROM rtk_nrtk_pairs", conn)
    conn.close()

    rtk_col  = "RTK"  if "RTK"  in pairs.columns else "rtk"
    nrtk_col = "NRTK" if "NRTK" in pairs.columns else "nrtk"
    rtk_set  = set(pairs[rtk_col])
    nrtk_set = set(pairs[nrtk_col])

    genes = sorted(set(expr.index) & set(mut.index) & set(seqs.index))
    log.info(f"  Common genes: {len(genes)}")
    if not genes:
        raise RuntimeError("No common genes found. Check database contents.")

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
def generate_embeddings(sequences):
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
        tokens = tokenizer(seqs, return_tensors="pt", add_special_tokens=True,
                           padding=True, truncation=True, max_length=1024)
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
    log.info(f"Embeddings saved -> {EMB_CACHE}  shape={embeddings.shape}")
    return embeddings


# ─────────────────────────────────────────────────────────────
# 3. SIMILARITY MATRICES
# ─────────────────────────────────────────────────────────────
def embedding_similarity(embeddings):
    log.info("Computing embedding similarity (cosine)...")
    return cosine_similarity(embeddings)


def expression_similarity(expr):
    log.info("Computing expression similarity (Pearson)...")
    arr  = expr.values.astype(float)
    stds = arr.std(axis=1, keepdims=True)
    safe = np.where(stds == 0, 1.0, stds)
    z    = (arr - arr.mean(axis=1, keepdims=True)) / safe
    corr = z @ z.T / arr.shape[1]
    np.fill_diagonal(corr, 1.0)
    return corr


def mutation_similarity(mut):
    log.info("Computing mutation co-occurrence (Fisher exact)...")
    arr    = mut.values
    n      = arr.shape[0]
    scores = np.zeros((n, n), dtype=float)
    for i in tqdm(range(n), desc="Fisher exact"):
        for j in range(i, n):
            a = int(((arr[i]==1) & (arr[j]==1)).sum())
            b = int(((arr[i]==1) & (arr[j]==0)).sum())
            c = int(((arr[i]==0) & (arr[j]==1)).sum())
            d = int(((arr[i]==0) & (arr[j]==0)).sum())
            _, p  = fisher_exact([[a, b], [c, d]])
            v     = -np.log10(p + 1e-300)
            scores[i, j] = scores[j, i] = v
    return scores


def normalize(arr):
    lo, hi = arr.min(), arr.max()
    if np.isclose(lo, hi):
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def composite_redundancy(sim_emb, sim_expr, sim_mut):
    log.info("Computing composite redundancy scores...")
    return (
        WEIGHTS["emb"]  * normalize(sim_emb)  +
        WEIGHTS["expr"] * normalize(sim_expr) +
        WEIGHTS["mut"]  * normalize(sim_mut)
    )


# ─────────────────────────────────────────────────────────────
# 4. BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────
def bootstrap_redundancy(expr, mut, embeddings, n_bootstrap=N_BOOTSTRAP):
    """
    Bootstrap CI on composite redundancy scores.

    Strategy:
      - Resample columns (samples) with replacement for expression & mutation
      - Recompute Pearson + Fisher + cosine each iteration
      - CI derived from percentiles of bootstrap distribution
      - p-value = fraction of null iterations where score >= observed

    Returns
    -------
    mean_scores  : (n, n) mean across bootstrap iterations
    ci_lower     : (n, n) lower CI bound
    ci_upper     : (n, n) upper CI bound
    p_values     : (n, n) empirical p-values
    """
    log.info(f"Running bootstrap (n={n_bootstrap} iterations)...")
    rng      = np.random.default_rng(RANDOM_SEED)
    n_genes  = len(expr)
    n_samples = expr.shape[1]

    expr_arr = expr.values.astype(float)
    mut_arr  = mut.values.astype(int)
    boot_scores = np.zeros((n_bootstrap, n_genes, n_genes), dtype=float)

    for b in tqdm(range(n_bootstrap), desc="Bootstrap"):
        idx = rng.integers(0, n_samples, size=n_samples)

        # Resampled expression similarity
        e   = expr_arr[:, idx]
        std = e.std(axis=1, keepdims=True)
        sfe = np.where(std == 0, 1.0, std)
        z   = (e - e.mean(axis=1, keepdims=True)) / sfe
        sim_e = z @ z.T / n_samples
        np.fill_diagonal(sim_e, 1.0)

        # Resampled mutation similarity (Fisher)
        m = mut_arr[:, idx]
        sim_m = np.zeros((n_genes, n_genes), dtype=float)
        for i in range(n_genes):
            for j in range(i, n_genes):
                a = int(((m[i]==1) & (m[j]==1)).sum())
                b_ = int(((m[i]==1) & (m[j]==0)).sum())
                c = int(((m[i]==0) & (m[j]==1)).sum())
                d = int(((m[i]==0) & (m[j]==0)).sum())
                _, p = fisher_exact([[a, b_], [c, d]])
                v = -np.log10(p + 1e-300)
                sim_m[i, j] = sim_m[j, i] = v

        # Embedding similarity unchanged (sequence doesn't resample)
        sim_emb_b = cosine_similarity(embeddings)

        comp = (
            WEIGHTS["emb"]  * normalize(sim_emb_b) +
            WEIGHTS["expr"] * normalize(sim_e)     +
            WEIGHTS["mut"]  * normalize(sim_m)
        )
        boot_scores[b] = comp

    lo_pct = (100 - CI) / 2
    hi_pct = 100 - lo_pct

    mean_scores = boot_scores.mean(axis=0)
    ci_lower    = np.percentile(boot_scores, lo_pct, axis=0)
    ci_upper    = np.percentile(boot_scores, hi_pct, axis=0)

    # Empirical p-value: how often does the bootstrap score exceed the observed?
    # Observed = mean_scores; null = each bootstrap sample
    # p-value for pair (i,j) = fraction of boots where score <= null mean
    null_mean   = boot_scores.mean(axis=0)
    p_values    = np.zeros((n_genes, n_genes), dtype=float)
    for i in range(n_genes):
        for j in range(n_genes):
            if i == j:
                p_values[i, j] = 0.0
                continue
            obs = mean_scores[i, j]
            p_values[i, j] = (boot_scores[:, i, j] >= obs).mean()

    log.info(f"  Bootstrap complete. Mean score range: "
             f"{mean_scores.min():.3f} — {mean_scores.max():.3f}")
    sig_count = ((p_values < ALPHA) & (np.triu(np.ones_like(p_values), 1).astype(bool))).sum()
    log.info(f"  Significant pairs (p<{ALPHA}): {sig_count}")

    return mean_scores, ci_lower, ci_upper, p_values


# ─────────────────────────────────────────────────────────────
# 5. SAVE RESULTS TO SQLITE
# ─────────────────────────────────────────────────────────────
def save_results(genes, redundancy, ci_lower, ci_upper, p_values,
                 sim_emb, sim_expr, sim_mut):
    log.info("Saving results to SQLite...")
    conn = sqlite3.connect(DB_PATH)
    rows = []
    for i, g1 in enumerate(genes):
        for j, g2 in enumerate(genes):
            if j <= i:
                continue
            rows.append({
                "gene1":           g1,
                "gene2":           g2,
                "score_emb":       float(sim_emb[i, j]),
                "score_expr":      float(sim_expr[i, j]),
                "score_mut":       float(sim_mut[i, j]),
                "score_composite": float(redundancy[i, j]),
                "ci_lower":        float(ci_lower[i, j]),
                "ci_upper":        float(ci_upper[i, j]),
                "p_value":         float(p_values[i, j]),
                "significant":     int(p_values[i, j] < ALPHA),
            })
    df = pd.DataFrame(rows).sort_values("score_composite", ascending=False)
    df.to_sql("redundancy_scores", conn, if_exists="replace", index=False)
    # Also export top significant pairs as CSV
    sig = df[df.significant == 1]
    sig.to_csv(RESULTS_DIR / "significant_pairs.csv", index=False)
    log.info(f"  Saved {len(df)} pairs | {len(sig)} significant")
    conn.close()
    return df


# ─────────────────────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────────────────────

# ── 6a. Score distribution with CI overlay ──────────────────
def plot_score_distribution(genes, redundancy, ci_lower, ci_upper, p_values):
    log.info("Plotting score distribution...")
    n = len(genes)
    scores, cis_lo, cis_hi, pvals = [], [], [], []
    for i in range(n):
        for j in range(i+1, n):
            scores.append(redundancy[i, j])
            cis_lo.append(ci_lower[i, j])
            cis_hi.append(ci_upper[i, j])
            pvals.append(p_values[i, j])

    scores = np.array(scores)
    pvals  = np.array(pvals)
    ci_w   = np.array(cis_hi) - np.array(cis_lo)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("BRCA — Bootstrap Redundancy Score Statistics",
                 fontsize=14, fontweight="bold", y=1.02)

    # Panel 1: score histogram
    ax = axes[0]
    sig_mask = pvals < ALPHA
    ax.hist(scores[~sig_mask], bins=35, color="#457B9D", alpha=0.7,
            label=f"Non-significant (n={( ~sig_mask).sum()})")
    ax.hist(scores[sig_mask],  bins=35, color="#E63946", alpha=0.85,
            label=f"Significant p<{ALPHA} (n={sig_mask.sum()})")
    ax.axvline(THRESH, color="black", linewidth=1.5, linestyle="--",
               label=f"Threshold={THRESH}")
    ax.set_xlabel("Composite Redundancy Score", fontsize=11)
    ax.set_ylabel("Number of Gene Pairs", fontsize=11)
    ax.set_title("Score Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel 2: CI width distribution
    ax = axes[1]
    ax.hist(ci_w, bins=35, color="#2A9D8F", alpha=0.85, edgecolor="white")
    ax.axvline(ci_w.mean(), color="#E63946", linewidth=2,
               label=f"Mean CI width = {ci_w.mean():.3f}")
    ax.set_xlabel(f"{CI}% CI Width", fontsize=11)
    ax.set_ylabel("Number of Gene Pairs", fontsize=11)
    ax.set_title("Bootstrap Confidence Interval Width", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel 3: p-value distribution
    ax = axes[2]
    ax.hist(pvals, bins=35, color="#F4A261", alpha=0.85, edgecolor="white")
    ax.axvline(ALPHA, color="#E63946", linewidth=2, linestyle="--",
               label=f"alpha={ALPHA}")
    ax.set_xlabel("Empirical p-value", fontsize=11)
    ax.set_ylabel("Number of Gene Pairs", fontsize=11)
    ax.set_title("p-value Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "brca_bootstrap_stats.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: brca_bootstrap_stats.png")


# ── 6b. Volcano plot ─────────────────────────────────────────
def plot_volcano(genes, kinases, redundancy, p_values):
    log.info("Plotting volcano plot...")
    n = len(genes)
    records = []
    for i in range(n):
        for j in range(i+1, n):
            g1, g2 = genes[i], genes[j]
            t1 = kinases[kinases.gene==g1]["type"].values[0]
            t2 = kinases[kinases.gene==g2]["type"].values[0]
            records.append({
                "gene1": g1, "gene2": g2,
                "pair_type": f"{t1}-{t2}",
                "score": redundancy[i, j],
                "neg_log_p": -np.log10(p_values[i, j] + 1e-10),
                "significant": p_values[i, j] < ALPHA and redundancy[i, j] >= THRESH,
            })

    df = pd.DataFrame(records)
    colors = {
        "RTK-RTK":  "#E63946",
        "NRTK-RTK": "#F4A261",
        "RTK-NRTK": "#F4A261",
        "NRTK-NRTK":"#457B9D",
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    for ptype, grp in df.groupby("pair_type"):
        sig   = grp[grp.significant]
        nonsig = grp[~grp.significant]
        c = colors.get(ptype, "#888888")
        ax.scatter(nonsig.score, nonsig.neg_log_p, c=c, alpha=0.3,
                   s=30, label=f"{ptype} (n={len(grp)})")
        ax.scatter(sig.score,    sig.neg_log_p,    c=c, alpha=0.9,
                   s=80, edgecolors="black", linewidths=0.5)

    # Label top significant pairs
    top = df[df.significant].nlargest(10, "score")
    for _, row in top.iterrows():
        ax.annotate(f"{row.gene1}-{row.gene2}",
                    xy=(row.score, row.neg_log_p),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=7, color="#1D3557",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

    ax.axvline(THRESH,        color="black",   linewidth=1.5, linestyle="--",
               label=f"Score threshold={THRESH}")
    ax.axhline(-np.log10(ALPHA), color="#E63946", linewidth=1.5, linestyle=":",
               label=f"p={ALPHA}")
    ax.set_xlabel("Composite Redundancy Score", fontsize=12)
    ax.set_ylabel("-log10(p-value)", fontsize=12)
    ax.set_title("BRCA — Kinase Redundancy Volcano Plot\n"
                 "(filled = significant & above threshold)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "brca_volcano.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: brca_volcano.png")


# ── 6c. Significance-coded heatmap ───────────────────────────
def plot_heatmap(genes, kinases, redundancy, p_values):
    log.info("Plotting significance heatmap...")
    n   = len(genes)
    red = redundancy.copy()

    # Mask non-significant pairs (grey them out)
    masked = np.where(p_values < ALPHA, red, np.nan)

    # Sort genes: RTKs first, then NRTKs, each sorted by mean score
    rtks  = kinases[kinases.type=="RTK"]["gene"].tolist()
    nrtks = kinases[kinases.type=="NRTK"]["gene"].tolist()
    order = rtks + nrtks
    idx   = [genes.index(g) for g in order]
    red_s = red[np.ix_(idx, idx)]
    msk_s = masked[np.ix_(idx, idx)]

    fig, axes = plt.subplots(1, 2, figsize=(22, 9),
                             gridspec_kw={"width_ratios": [1, 1]})
    fig.suptitle("BRCA — Kinase Redundancy Heatmap",
                 fontsize=15, fontweight="bold")

    kw = dict(aspect="auto", vmin=0, vmax=1)

    # Left: all scores
    ax = axes[0]
    im = ax.imshow(red_s, cmap="RdYlGn", **kw)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticklabels(order, fontsize=7)
    # Draw divider between RTK/NRTK blocks
    div = len(rtks) - 0.5
    ax.axhline(div, color="black", linewidth=1.5)
    ax.axvline(div, color="black", linewidth=1.5)
    plt.colorbar(im, ax=ax, fraction=0.03, label="Redundancy Score")
    ax.set_title("All Scores", fontsize=12, fontweight="bold")

    # Right: significant only (grey = non-sig)
    ax = axes[1]
    # Background: full grey
    ax.imshow(red_s, cmap="Greys", vmin=0, vmax=1, aspect="auto", alpha=0.25)
    # Overlay significant scores
    cmap_sig = plt.cm.RdYlGn.copy()
    cmap_sig.set_bad(color="#EEEEEE")
    im2 = ax.imshow(msk_s, cmap=cmap_sig, **kw)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticklabels(order, fontsize=7)
    ax.axhline(div, color="black", linewidth=1.5)
    ax.axvline(div, color="black", linewidth=1.5)
    plt.colorbar(im2, ax=ax, fraction=0.03, label="Redundancy Score (sig. only)")
    ax.set_title(f"Significant Only (p<{ALPHA})", fontsize=12, fontweight="bold")

    # Annotate quadrants
    for ax_ in axes:
        ax_.text(len(rtks)/2 - 0.5, -1.8, "RTKs", ha="center",
                 fontsize=9, fontweight="bold", color="#E63946")
        ax_.text(len(rtks) + len(nrtks)/2 - 0.5, -1.8, "NRTKs", ha="center",
                 fontsize=9, fontweight="bold", color="#457B9D")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "brca_redundancy_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: brca_redundancy_heatmap.png")


# ── 6d. Network with CI-coded edges ──────────────────────────
def build_and_plot_networks(genes, kinases, redundancy, ci_lower,
                            ci_upper, p_values):
    RTK_COLOR  = "#E63946"
    NRTK_COLOR = "#457B9D"

    rtks  = kinases[kinases.type=="RTK"]["gene"].tolist()
    nrtks = kinases[kinases.type=="NRTK"]["gene"].tolist()

    # Build significant-only graph
    G = nx.Graph()
    for _, row in kinases.iterrows():
        G.add_node(row["gene"], type=row["type"])

    for i, g1 in enumerate(genes):
        for j, g2 in enumerate(genes):
            if j <= i:
                continue
            if (redundancy[i, j] >= THRESH and p_values[i, j] < ALPHA):
                ci_w = float(ci_upper[i, j] - ci_lower[i, j])
                G.add_edge(g1, g2,
                           weight=float(redundancy[i, j]),
                           ci_width=ci_w,
                           p_value=float(p_values[i, j]))

    log.info(f"  Significant network: {G.number_of_nodes()} nodes, "
             f"{G.number_of_edges()} edges")

    # ── RTK-RTK ──
    _draw_sig_subgraph(G.subgraph(rtks).copy(),
                       "RTK-RTK Redundancy (Bootstrap Significant)",
                       RTK_COLOR, RESULTS_DIR / "brca_network_rtk_rtk.png")

    # ── NRTK-NRTK ──
    _draw_sig_subgraph(G.subgraph(nrtks).copy(),
                       "NRTK-NRTK Redundancy (Bootstrap Significant)",
                       NRTK_COLOR, RESULTS_DIR / "brca_network_nrtk_nrtk.png")

    # ── Combined RTK-NRTK force directed ──
    G_cross = nx.Graph()
    for _, row in kinases.iterrows():
        G_cross.add_node(row["gene"], type=row["type"])
    for i, g1 in enumerate(genes):
        for j, g2 in enumerate(genes):
            if j <= i:
                continue
            if (redundancy[i, j] >= CROSS_THRESH and p_values[i, j] < ALPHA):
                ci_w = float(ci_upper[i, j] - ci_lower[i, j])
                G_cross.add_edge(g1, g2,
                                 weight=float(redundancy[i, j]),
                                 ci_width=ci_w)

    cross_edges = [(u, v) for u, v in G_cross.edges()
                   if G_cross.nodes[u]["type"] != G_cross.nodes[v]["type"]]
    same_edges  = [(u, v) for u, v in G_cross.edges()
                   if G_cross.nodes[u]["type"] == G_cross.nodes[v]["type"]]
    log.info(f"  RTK-NRTK cross edges: {len(cross_edges)}")

    fig, ax = plt.subplots(figsize=(18, 16))
    pos = nx.spring_layout(G_cross, seed=RANDOM_SEED, k=2.8, iterations=200)

    degrees   = dict(G_cross.degree())
    rtk_sizes  = [700 + 300 * degrees.get(n, 0) for n in rtks  if n in G_cross]
    nrtk_sizes = [700 + 300 * degrees.get(n, 0) for n in nrtks if n in G_cross]

    # Same-type edges (faint)
    if same_edges:
        nx.draw_networkx_edges(G_cross, pos, edgelist=same_edges,
                               edge_color="#CCCCCC", width=0.8, alpha=0.4, ax=ax)

    # Cross-type edges: width = score, alpha = CI confidence
    if cross_edges:
        cross_w  = [G_cross[u][v]["weight"] * 6 for u, v in cross_edges]
        cross_al = [max(0.3, 1 - G_cross[u][v]["ci_width"] * 3)
                    for u, v in cross_edges]
        for (u, v), w, a in zip(cross_edges, cross_w, cross_al):
            nx.draw_networkx_edges(G_cross, pos, edgelist=[(u, v)],
                                   edge_color="#F4A261", width=w,
                                   alpha=a, ax=ax)

    nx.draw_networkx_nodes(G_cross, pos,
                           nodelist=[n for n in rtks  if n in G_cross],
                           node_color=RTK_COLOR, node_size=rtk_sizes,
                           ax=ax, alpha=0.92)
    nx.draw_networkx_nodes(G_cross, pos,
                           nodelist=[n for n in nrtks if n in G_cross],
                           node_color=NRTK_COLOR, node_size=nrtk_sizes,
                           ax=ax, alpha=0.92)
    nx.draw_networkx_labels(G_cross, pos, ax=ax,
                            font_size=8, font_weight="bold", font_color="white")

    ax.legend(handles=[
        mpatches.Patch(facecolor=RTK_COLOR,  label=f"RTK (n={len(rtks)})"),
        mpatches.Patch(facecolor=NRTK_COLOR, label=f"NRTK (n={len(nrtks)})"),
        mpatches.Patch(facecolor="#F4A261",  label=f"RTK-NRTK edge (n={len(cross_edges)}, p<{ALPHA})"),
        mpatches.Patch(facecolor="#CCCCCC",  label="Same-type edge"),
    ], fontsize=10, loc="lower right", framealpha=0.9)

    ax.set_title(
        f"BRCA — Bootstrap-Validated RTK-NRTK Redundancy Network\n"
        f"(Edge width = redundancy score | "
        f"Edge opacity = CI confidence | threshold={CROSS_THRESH}, p<{ALPHA})",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "brca_network_rtk_nrtk.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: brca_network_rtk_nrtk.png")

    return G


def _draw_sig_subgraph(subgraph, title, node_color, outpath):
    if len(subgraph.nodes()) == 0:
        log.warning(f"  Skipping {title} — no nodes")
        return

    fig, ax = plt.subplots(figsize=(14, 14))
    has_edges = subgraph.number_of_edges() > 0
    pos = nx.spring_layout(subgraph, seed=RANDOM_SEED, k=3.0, iterations=150) \
          if has_edges else nx.circular_layout(subgraph)

    degrees = dict(subgraph.degree())
    sizes   = [900 + 350 * degrees.get(n, 0) for n in subgraph.nodes()]

    nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_color=node_color,
                           node_size=sizes, alpha=0.9)

    if has_edges:
        # Width = score, opacity = 1 - ci_width (narrower CI = more opaque)
        for u, v, data in subgraph.edges(data=True):
            w  = data.get("weight",   0.5) * 5
            ci = data.get("ci_width", 0.1)
            a  = max(0.3, 1 - ci * 3)
            nx.draw_networkx_edges(subgraph, pos, edgelist=[(u, v)],
                                   width=w, alpha=a,
                                   edge_color="#333333", ax=ax)

    nx.draw_networkx_labels(subgraph, pos, ax=ax,
                            font_size=9, font_weight="bold", font_color="white")

    edge_info = f"{subgraph.number_of_edges()} sig. edges" if has_edges \
                else "no significant edges"
    ax.set_title(f"BRCA — {title}\n({edge_info}, p<{ALPHA})",
                 fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {outpath}")


# ── 6e. Top pairs CI bar chart ───────────────────────────────
def plot_ci_bars(genes, kinases, redundancy, ci_lower, ci_upper, p_values):
    log.info("Plotting top pairs CI chart...")
    n = len(genes)
    records = []
    for i in range(n):
        for j in range(i+1, n):
            if p_values[i, j] >= ALPHA:
                continue
            g1, g2 = genes[i], genes[j]
            t1 = kinases[kinases.gene==g1]["type"].values[0]
            t2 = kinases[kinases.gene==g2]["type"].values[0]
            records.append({
                "pair":      f"{g1}–{g2}",
                "type":      f"{t1}/{t2}",
                "score":     redundancy[i, j],
                "ci_lo":     ci_lower[i, j],
                "ci_hi":     ci_upper[i, j],
                "p_value":   p_values[i, j],
            })

    if not records:
        log.warning("No significant pairs for CI chart.")
        return

    df = pd.DataFrame(records).sort_values("score", ascending=False).head(25)
    err_lo = df.score - df.ci_lo
    err_hi = df.ci_hi - df.score

    color_map = {"RTK/RTK": "#E63946", "NRTK/NRTK": "#457B9D",
                 "RTK/NRTK": "#F4A261", "NRTK/RTK": "#F4A261"}
    colors = [color_map.get(t, "#888888") for t in df.type]

    fig, ax = plt.subplots(figsize=(12, 9))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df.score, xerr=[err_lo, err_hi],
            color=colors, alpha=0.85, capsize=4,
            error_kw={"elinewidth": 1.5, "ecolor": "#333333"})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.pair, fontsize=9)
    ax.axvline(THRESH, color="black", linewidth=1.5,
               linestyle="--", label=f"Threshold={THRESH}")
    ax.set_xlabel("Composite Redundancy Score ± 95% CI", fontsize=11)
    ax.set_title(f"BRCA — Top 25 Significant Kinase Pairs\n"
                 f"(Bootstrap {CI}% CI, p<{ALPHA})",
                 fontsize=13, fontweight="bold")

    legend_handles = [
        mpatches.Patch(facecolor="#E63946",  label="RTK/RTK"),
        mpatches.Patch(facecolor="#457B9D",  label="NRTK/NRTK"),
        mpatches.Patch(facecolor="#F4A261",  label="RTK/NRTK"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "brca_top_pairs_ci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: brca_top_pairs_ci.png")


def plot_pca(genes, kinases, mean_scores, sim_emb, sim_expr, sim_mut, embeddings, expr, mut):
    """
    Plot PCA visualizations for the analysis
    
    Parameters:
    genes: list of gene names
    kinases: list of kinase names
    mean_scores: mean scores from analysis
    sim_emb: similarity matrix for embeddings
    sim_expr: similarity matrix for expression
    sim_mut: similarity matrix for mutations
    embeddings: embedding data
    expr: expression data
    mut: mutation data
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Example PCA plots (customize based on your data structure)
    # Plot 1: PCA of embeddings
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings.T)  # Adjust based on your data orientation
    axes[0, 0].scatter(embeddings_pca[:, 0], embeddings_pca[:, 1])
    axes[0, 0].set_title('PCA of Embeddings')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    
    # Plot 2: PCA of expression data
    expr_pca = pca.fit_transform(expr.T)
    axes[0, 1].scatter(expr_pca[:, 0], expr_pca[:, 1])
    axes[0, 1].set_title('PCA of Expression Data')
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    
    # Plot 3: PCA of mutation data
    mut_pca = pca.fit_transform(mut.T)
    axes[0, 2].scatter(mut_pca[:, 0], mut_pca[:, 1])
    axes[0, 2].set_title('PCA of Mutation Data')
    axes[0, 2].set_xlabel('PC1')
    axes[0, 2].set_ylabel('PC2')
    
    # Plot 4: Similarity matrix heatmap for embeddings
    im = axes[1, 0].imshow(sim_emb, cmap='viridis', aspect='auto')
    axes[1, 0].set_title('Similarity Matrix - Embeddings')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Plot 5: Similarity matrix for expression
    im = axes[1, 1].imshow(sim_expr, cmap='viridis', aspect='auto')
    axes[1, 1].set_title('Similarity Matrix - Expression')
    plt.colorbar(im, ax=axes[1, 1])
    
    # Plot 6: Similarity matrix for mutations
    im = axes[1, 2].imshow(sim_mut, cmap='viridis', aspect='auto')
    axes[1, 2].set_title('Similarity Matrix - Mutation')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('pca_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    print("\nPCA Analysis Summary:")
    print(f"Number of genes analyzed: {len(genes)}")
    print(f"Number of kinases: {len(kinases)}")
    print(f"Mean scores shape: {mean_scores.shape if hasattr(mean_scores, 'shape') else 'N/A'}")

# ─────────────────────────────────────────────────────────────
# 7. SUMMARY REPORT
# ─────────────────────────────────────────────────────────────
def print_summary(G, genes, kinases, redundancy, p_values, df_results):
    sig = df_results[df_results.significant == 1]
    print("\n" + "="*65)
    print("  BRCA RTK-NRTK REDUNDANCY — BOOTSTRAP SUMMARY")
    print("="*65)
    print(f"\n  Genes analysed     : {len(genes)}")
    print(f"  RTKs               : {kinases[kinases.type=='RTK'].shape[0]}")
    print(f"  NRTKs              : {kinases[kinases.type=='NRTK'].shape[0]}")
    print(f"  Bootstrap iters    : {N_BOOTSTRAP}")
    print(f"  Confidence interval: {CI}%")
    print(f"  Significance alpha : {ALPHA}")
    print(f"  Total pairs tested : {len(df_results)}")
    print(f"  Significant pairs  : {len(sig)}")
    print(f"  Network edges      : {G.number_of_edges()}")

    print(f"\n  Top 15 significant pairs (sorted by score):")
    print(f"  {'Gene1':<10} {'Type1':<6} {'Gene2':<10} {'Type2':<6} "
          f"{'Score':>7} {'CI Lo':>7} {'CI Hi':>7} {'p-val':>8}")
    print(f"  {'-'*62}")
    for _, row in sig.head(15).iterrows():
        t1 = kinases[kinases.gene==row.gene1]["type"].values[0]
        t2 = kinases[kinases.gene==row.gene2]["type"].values[0]
        print(f"  {row.gene1:<10} {t1:<6} {row.gene2:<10} {t2:<6} "
              f"{row.score_composite:>7.4f} {row.ci_lower:>7.4f} "
              f"{row.ci_upper:>7.4f} {row.p_value:>8.4f}")

    if G.number_of_edges() > 0:
        print(f"\n  Most connected kinases (significant network):")
        for gene, deg in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {gene:<12} ({G.nodes[gene]['type']})  {deg} connections")

    print(f"\n  Output files saved to Results/:")
    for f in ["brca_bootstrap_stats.png", "brca_volcano.png",
              "brca_redundancy_heatmap.png", "brca_top_pairs_ci.png",
              "brca_network_rtk_rtk.png", "brca_network_nrtk_nrtk.png",
              "brca_network_rtk_nrtk.png", "significant_pairs.csv"]:
        print(f"  • {f}")
    print("="*65 + "\n")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    genes, expr, mut, seqs, kinases = load_data()
    embeddings = generate_embeddings(seqs["sequence"].tolist())

    sim_emb  = embedding_similarity(embeddings)
    sim_expr = expression_similarity(expr)
    sim_mut  = mutation_similarity(mut)
    redundancy = composite_redundancy(sim_emb, sim_expr, sim_mut)

    # Bootstrap
    mean_scores, ci_lower, ci_upper, p_values = bootstrap_redundancy(
        expr, mut, embeddings, n_bootstrap=N_BOOTSTRAP
    )

    # Use bootstrap mean as the final score
    df_results = save_results(
        genes, mean_scores, ci_lower, ci_upper, p_values,
        sim_emb, sim_expr, sim_mut
    )

    # Visualisations
    plot_score_distribution(genes, mean_scores, ci_lower, ci_upper, p_values)
    plot_volcano(genes, kinases, mean_scores, p_values)
    plot_heatmap(genes, kinases, mean_scores, p_values)
    plot_ci_bars(genes, kinases, mean_scores, ci_lower, ci_upper, p_values)
    G = build_and_plot_networks(
        genes, kinases, mean_scores, ci_lower, ci_upper, p_values
    )
    plot_pca(genes, kinases, mean_scores, sim_emb, sim_expr, sim_mut, embeddings, expr, mut)
    print_summary(G, genes, kinases, mean_scores, p_values, df_results)
    log.info("BRCA bootstrap pipeline complete!")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────
# PCA ANALYSIS
# ─────────────────────────────────────────────────────────────
def plot_pca(genes, kinases, redundancy, sim_emb, sim_expr, sim_mut,
             embeddings, expr, mut):
    """
    Multi-panel PCA analysis of kinase redundancy in BRCA.

    Panel layout:
      1. PCA on composite redundancy matrix (each kinase's redundancy profile)
      2. PCA on raw ESM-2 embeddings
      3. PCA on expression profiles
      4. Explained variance scree plot (composite)
      5. PC1 vs PC3 biplot (composite)
      6. Per-modality variance explained comparison bar chart
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    log.info("Running PCA analysis...")

    RTK_COLOR  = "#E63946"
    NRTK_COLOR = "#457B9D"

    types  = kinases.set_index("gene")["type"]
    colors = [RTK_COLOR if types[g] == "RTK" else NRTK_COLOR for g in genes]
    markers = ["o" if types[g] == "RTK" else "s" for g in genes]

    # ── Prepare data matrices ──────────────────────────────────
    # 1. Composite redundancy profile (43 x 43 → each row = kinase's profile)
    scaler = StandardScaler()
    R_scaled = scaler.fit_transform(redundancy)
    pca_comp = PCA(n_components=min(10, len(genes)))
    pc_comp  = pca_comp.fit_transform(R_scaled)
    var_comp = pca_comp.explained_variance_ratio_

    # 2. ESM-2 embeddings (43 x 1280)
    emb_scaled = scaler.fit_transform(embeddings)
    pca_emb  = PCA(n_components=min(10, embeddings.shape[1]))
    pc_emb   = pca_emb.fit_transform(emb_scaled)
    var_emb  = pca_emb.explained_variance_ratio_

    # 3. Expression profiles (43 x 1082)
    expr_scaled = scaler.fit_transform(expr.values)
    pca_expr = PCA(n_components=min(10, expr.shape[1]))
    pc_expr  = pca_expr.fit_transform(expr_scaled)
    var_expr = pca_expr.explained_variance_ratio_

    # 4. Mutation profiles (43 x 1082)
    mut_scaled = scaler.fit_transform(mut.values.astype(float))
    pca_mut  = PCA(n_components=min(10, mut.shape[1]))
    pc_mut   = pca_mut.fit_transform(mut_scaled)
    var_mut  = pca_mut.explained_variance_ratio_

    log.info(f"  Composite PCA: PC1={var_comp[0]:.1%}, PC2={var_comp[1]:.1%}, "
             f"PC3={var_comp[2]:.1%}")
    log.info(f"  Embedding PCA: PC1={var_emb[0]:.1%}, PC2={var_emb[1]:.1%}")
    log.info(f"  Expression PCA: PC1={var_expr[0]:.1%}, PC2={var_expr[1]:.1%}")

    # ─────────────────────────────────────────────────────────
    # FIGURE 1 — 6-panel overview
    # ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle("BRCA Kinase Redundancy — PCA Analysis",
                 fontsize=17, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.32)
    axes = [
        fig.add_subplot(gs[0, 0]),  # 0: composite PC1 vs PC2
        fig.add_subplot(gs[0, 1]),  # 1: ESM-2 PC1 vs PC2
        fig.add_subplot(gs[0, 2]),  # 2: expression PC1 vs PC2
        fig.add_subplot(gs[1, 0]),  # 3: mutation PC1 vs PC2
        fig.add_subplot(gs[1, 1]),  # 4: scree composite
        fig.add_subplot(gs[1, 2]),  # 5: PC1 vs PC3 composite
        fig.add_subplot(gs[2, :]),  # 6: variance comparison bar
    ]

    legend_handles = [
        mpatches.Patch(facecolor=RTK_COLOR,  label="RTK"),
        mpatches.Patch(facecolor=NRTK_COLOR, label="NRTK"),
    ]

    def scatter_pca(ax, pcs, var, title, annotate=True):
        for i, g in enumerate(genes):
            ax.scatter(pcs[i, 0], pcs[i, 1],
                       c=colors[i], marker=markers[i],
                       s=120, alpha=0.85, edgecolors="white", linewidths=0.5)
            if annotate:
                ax.annotate(g, (pcs[i, 0], pcs[i, 1]),
                            fontsize=6.5, ha="left", va="bottom",
                            xytext=(3, 3), textcoords="offset points",
                            color="#1D3557")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_xlabel(f"PC1 ({var[0]:.1%} var)", fontsize=10)
        ax.set_ylabel(f"PC2 ({var[1]:.1%} var)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    # Panel 0: composite redundancy
    scatter_pca(axes[0], pc_comp, var_comp,
                "Composite Redundancy\nPCA (PC1 vs PC2)")

    # Panel 1: ESM-2
    scatter_pca(axes[1], pc_emb, var_emb,
                "ESM-2 Sequence Embeddings\nPCA (PC1 vs PC2)")

    # Panel 2: expression
    scatter_pca(axes[2], pc_expr, var_expr,
                "Expression Profiles\nPCA (PC1 vs PC2)")

    # Panel 3: mutation
    scatter_pca(axes[3], pc_mut, var_mut,
                "Mutation Profiles\nPCA (PC1 vs PC2)")

    # Panel 4: scree plot
    ax = axes[4]
    cumvar = np.cumsum(var_comp)
    ax.bar(range(1, len(var_comp)+1), var_comp * 100,
           color="#457B9D", alpha=0.8, label="Individual")
    ax.plot(range(1, len(cumvar)+1), cumvar * 100,
            color="#E63946", marker="o", markersize=5,
            linewidth=2, label="Cumulative")
    ax.axhline(80, color="grey", linewidth=1, linestyle="--",
               label="80% threshold")
    ax.set_xlabel("Principal Component", fontsize=10)
    ax.set_ylabel("Variance Explained (%)", fontsize=10)
    ax.set_title("Composite PCA\nScree Plot", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xticks(range(1, len(var_comp)+1))

    # Panel 5: PC1 vs PC3
    ax = axes[5]
    for i, g in enumerate(genes):
        ax.scatter(pc_comp[i, 0], pc_comp[i, 2],
                   c=colors[i], marker=markers[i],
                   s=120, alpha=0.85, edgecolors="white", linewidths=0.5)
        ax.annotate(g, (pc_comp[i, 0], pc_comp[i, 2]),
                    fontsize=6.5, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points",
                    color="#1D3557")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"PC1 ({var_comp[0]:.1%} var)", fontsize=10)
    ax.set_ylabel(f"PC3 ({var_comp[2]:.1%} var)", fontsize=10)
    ax.set_title("Composite Redundancy\nPCA (PC1 vs PC3)", fontsize=11, fontweight="bold")
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    # Panel 6: modality variance comparison
    ax = axes[6]
    n_pc = min(5, len(var_comp))
    x  = np.arange(n_pc)
    w  = 0.2
    mods = [
        ("Composite Redundancy", var_comp[:n_pc], "#2A9D8F"),
        ("ESM-2 Embeddings",     var_emb[:n_pc],  "#E76F51"),
        ("Expression Profiles",  var_expr[:n_pc], "#457B9D"),
        ("Mutation Profiles",    var_mut[:n_pc],  "#E63946"),
    ]
    for k, (label, var, col) in enumerate(mods):
        ax.bar(x + k*w, var*100, width=w, label=label,
               color=col, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Principal Component", fontsize=11)
    ax.set_ylabel("Variance Explained (%)", fontsize=11)
    ax.set_title("Per-Modality Variance Explained by First 5 PCs",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels([f"PC{i+1}" for i in range(n_pc)], fontsize=10)
    ax.legend(fontsize=9, loc="upper right")

    plt.savefig(RESULTS_DIR / "brca_pca_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: brca_pca_overview.png")

    # ─────────────────────────────────────────────────────────
    # FIGURE 2 — Biplot with loadings (composite PCA)
    # ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle("BRCA — Composite Redundancy PCA Biplot",
                 fontsize=15, fontweight="bold")

    for ax_idx, (pc_x, pc_y) in enumerate([(0, 1), (0, 2)]):
        ax = axes[ax_idx]

        # Score scatter
        for i, g in enumerate(genes):
            ax.scatter(pc_comp[i, pc_x], pc_comp[i, pc_y],
                       c=colors[i], marker=markers[i],
                       s=160, alpha=0.9, edgecolors="white", linewidths=0.8,
                       zorder=3)
            ax.annotate(g, (pc_comp[i, pc_x], pc_comp[i, pc_y]),
                        fontsize=7.5, fontweight="bold",
                        ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points",
                        color="#1D3557", zorder=4)

        # Loadings — top 8 most influential genes per axis
        loadings = pca_comp.components_   # shape: (n_pcs, n_genes)
        top_idx = np.argsort(
            np.abs(loadings[pc_x]) + np.abs(loadings[pc_y])
        )[-8:]

        scale = max(np.abs(pc_comp[:, pc_x]).max(),
                    np.abs(pc_comp[:, pc_y]).max()) * 0.6

        for idx in top_idx:
            lx = loadings[pc_x, idx] * scale
            ly = loadings[pc_y, idx] * scale
            ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>",
                                        color="#F4A261", lw=1.5))
            ax.text(lx * 1.08, ly * 1.08, genes[idx],
                    fontsize=7, color="#E76F51", fontweight="bold",
                    ha="center", va="center")

        ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
        ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
        ax.set_xlabel(f"PC{pc_x+1} ({var_comp[pc_x]:.1%} variance)", fontsize=11)
        ax.set_ylabel(f"PC{pc_y+1} ({var_comp[pc_y]:.1%} variance)", fontsize=11)
        ax.set_title(f"PC{pc_x+1} vs PC{pc_y+1}  |  Orange arrows = top loading genes",
                     fontsize=11, fontweight="bold")
        ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "brca_pca_biplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: brca_pca_biplot.png")

    # ─────────────────────────────────────────────────────────
    # FIGURE 3 — RTK/NRTK separation + per-PC loadings bar
    # ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("BRCA — PCA Class Separation and PC Loadings",
                 fontsize=15, fontweight="bold")

    # Left: confidence ellipses per class
    ax = axes[0]
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    def confidence_ellipse(x, y, ax, color, n_std=1.5):
        if len(x) < 3:
            return
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        rx = np.sqrt(1 + pearson)
        ry = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=rx*2, height=ry*2,
                          facecolor=color, alpha=0.12,
                          edgecolor=color, linewidth=2, linestyle="--")
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_x, mean_y = np.mean(x), np.mean(y)
        t = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
        ellipse.set_transform(t + ax.transData)
        ax.add_patch(ellipse)

    rtk_idx  = [i for i, g in enumerate(genes) if types[g] == "RTK"]
    nrtk_idx = [i for i, g in enumerate(genes) if types[g] == "NRTK"]

    ax.scatter(pc_comp[rtk_idx, 0],  pc_comp[rtk_idx, 1],
               c=RTK_COLOR,  s=150, alpha=0.9, marker="o",
               edgecolors="white", linewidths=0.8, label="RTK", zorder=3)
    ax.scatter(pc_comp[nrtk_idx, 0], pc_comp[nrtk_idx, 1],
               c=NRTK_COLOR, s=150, alpha=0.9, marker="s",
               edgecolors="white", linewidths=0.8, label="NRTK", zorder=3)

    confidence_ellipse(pc_comp[rtk_idx, 0],  pc_comp[rtk_idx, 1],  ax, RTK_COLOR)
    confidence_ellipse(pc_comp[nrtk_idx, 0], pc_comp[nrtk_idx, 1], ax, NRTK_COLOR)

    for i, g in enumerate(genes):
        ax.annotate(g, (pc_comp[i, 0], pc_comp[i, 1]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points", color="#1D3557")

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"PC1 ({var_comp[0]:.1%} var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_comp[1]:.1%} var)", fontsize=11)
    ax.set_title("RTK vs NRTK Separation\n(shaded = 1.5 SD confidence ellipse)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)

    # Right: top 15 PC1 loadings bar chart
    ax = axes[1]
    loadings_pc1 = pca_comp.components_[0]
    sorted_idx   = np.argsort(loadings_pc1)
    top15_idx    = np.concatenate([sorted_idx[:8], sorted_idx[-7:]])
    top15_genes  = [genes[i] for i in top15_idx]
    top15_vals   = loadings_pc1[top15_idx]
    bar_colors   = [RTK_COLOR if types[g] == "RTK" else NRTK_COLOR
                    for g in top15_genes]

    ax.barh(range(len(top15_genes)), top15_vals,
            color=bar_colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(top15_genes)))
    ax.set_yticklabels(top15_genes, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("PC1 Loading", fontsize=11)
    ax.set_title("Top PC1 Loadings\n(which kinases drive PC1 separation)",
                 fontsize=11, fontweight="bold")
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "brca_pca_separation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: brca_pca_separation.png")
    log.info("PCA analysis complete — 3 figures saved.")
