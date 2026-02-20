<div align="center">

<!-- BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=BRCA%20Kinase%20Redundancy&fontSize=40&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Multi-Modal%20RTK%20%E2%80%A2%20NRTK%20Analysis%20in%20Breast%20Cancer&descAlignY=55&descSize=18" width="100%"/>

<!-- BADGES ROW 1 — Status -->
[![Active](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/mtariqi/brca-kinase-redundancy)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/mtariqi/brca-kinase-redundancy?style=for-the-badge&logo=git&logoColor=white&color=orange)](https://github.com/mtariqi/brca-kinase-redundancy/commits/main)
[![Repo Size](https://img.shields.io/github/repo-size/mtariqi/brca-kinase-redundancy?style=for-the-badge&logo=github&logoColor=white&color=purple)](https://github.com/mtariqi/brca-kinase-redundancy)

<!-- BADGES ROW 2 — Languages & Tools -->
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ESM--2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org/)

<!-- BADGES ROW 3 — Infrastructure -->
[![Apache NiFi](https://img.shields.io/badge/Apache-NiFi-728E9B?style=for-the-badge&logo=apache&logoColor=white)](https://nifi.apache.org/)
[![Apache Doris](https://img.shields.io/badge/Apache-Doris-4479A1?style=for-the-badge&logo=apache&logoColor=white)](https://doris.apache.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC143C?style=for-the-badge&logo=databricks&logoColor=white)](https://qdrant.tech/)
[![CLI](https://img.shields.io/badge/CLI-Bash%20%2F%20Fish-4EAA25?style=for-the-badge&logo=gnubash&logoColor=white)](https://fishshell.com/)

<!-- BADGES ROW 4 — Data & Science -->
[![TCGA](https://img.shields.io/badge/Data-TCGA--BRCA-red?style=for-the-badge&logo=databricks&logoColor=white)](https://portal.gdc.cancer.gov/)
[![Samples](https://img.shields.io/badge/Cohort-1%2C082%20Samples-blueviolet?style=for-the-badge&logo=microsoftexcel&logoColor=white)](https://portal.gdc.cancer.gov/)
[![Kinases](https://img.shields.io/badge/Kinases-43%20(RTK%20%2B%20NRTK)-ff69b4?style=for-the-badge&logo=molecule&logoColor=white)](https://github.com/mtariqi/brca-kinase-redundancy)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-95%25%20CI-success?style=for-the-badge&logo=scipy&logoColor=white)](https://github.com/mtariqi/brca-kinase-redundancy)

<!-- BADGES ROW 5 — Libraries -->
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graphs-orange?style=for-the-badge&logo=python&logoColor=white)](https://networkx.org/)

</div>

---

## 🧬 Overview

> **Kinase redundancy** is a primary driver of therapeutic resistance in breast cancer. When one kinase is inhibited, a functionally redundant kinase compensates — sustaining oncogenic signalling. This pipeline **systematically quantifies** that redundancy across the full tyrosine kinome.

<div align="center">

```
TCGA-BRCA (1,082 samples · 43 kinases)
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
 ESM-2     Pearson    Fisher
Sequence  Co-express  Mutation
Embedding  Pearson   Co-occur
    │         │         │
    └─────────┼─────────┘
              ▼
     Composite Redundancy Score
              │
    ┌─────────┼──────────┐
    ▼         ▼          ▼
Bootstrap  Network     PCA
  95% CI   Graphs   Analysis
```

</div>

---

## 🏗️ Infrastructure Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                        │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Apache NiFi  │  │Apache Doris  │  │   Qdrant     │  │
│  │   (ETL)      │→ │  (OLAP DB)   │  │ (Vector DB)  │  │
│  │  Port 8443   │  │  Port 9030   │  │  Port 6333   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│          │                │                  │          │
│          └────────────────┼──────────────────┘          │
│                           ▼                             │
│              ┌─────────────────────┐                    │
│              │  Python Pipeline    │                    │
│              │  brca_pipeline.py   │                    │
│              └─────────────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Results Gallery

<div align="center">

| Heatmap | Volcano Plot |
|:---:|:---:|
| ![Heatmap](Results/brca_redundancy_heatmap.png) | ![Volcano](Results/brca_volcano.png) |

| RTK-RTK Network | NRTK-NRTK Network |
|:---:|:---:|
| ![RTK](Results/brca_network_rtk_rtk.png) | ![NRTK](Results/brca_network_nrtk_nrtk.png) |

| RTK-NRTK Cross-Type Network | Bootstrap Statistics |
|:---:|:---:|
| ![Cross](Results/brca_network_rtk_nrtk.png) | ![Bootstrap](Results/brca_bootstrap_stats.png) |

</div>

---

## ⚡ Quick Start

### Prerequisites

```bash
# System requirements
sudo swapoff -a                              # Disable swap (required for Doris)
sudo sysctl -w vm.max_map_count=2000000      # Set memory map limit
```

### 1. Clone & Setup

```bash
git clone https://github.com/mtariqi/brca-kinase-redundancy.git
cd brca-kinase-redundancy
```

### 2. Install Python Dependencies

```bash
pip install pandas numpy scipy scikit-learn networkx matplotlib \
            torch transformers tqdm qdrant-client anthropic pymysql
```

### 3. Start Infrastructure

```bash
docker-compose up -d
# Wait ~90 seconds for Doris to initialise

# Start Doris Backend Engine
docker exec -it doris bash -c "ulimit -n 655350 && \
  /opt/apache-doris/be/bin/start_be.sh --daemon"
```

### 4. Load Data

```bash
# Place TCGA-BRCA CSVs in Data/ folder, then:
python -c "
import pandas as pd, sqlite3
conn = sqlite3.connect('Data/tcga_brca.db')
pd.read_csv('Data/expression_data_processed.csv',
            index_col=0).T \
  .reset_index().melt(id_vars='index',
                      var_name='gene',
                      value_name='expression') \
  .rename(columns={'index':'sample_id'}) \
  .assign(cancer_type='BRCA') \
  .to_sql('expression_raw', conn,
          if_exists='replace', index=False)
conn.close()
print('Done')
"
```

### 5. Run Pipeline

```bash
python brca_pipeline_bootstrap_pcs.py
```

---

## 🔬 Methods

### Modality 1 — ESM-2 Protein Embeddings

```python
# facebook/esm2_t33_650M_UR50D (650M parameters, 33 layers)
# Mean-pooled last hidden state → cosine similarity
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
```

### Modality 2 — Transcriptomic Co-expression

```python
# Vectorised Pearson correlation (z-score method)
z = (arr - arr.mean(axis=1, keepdims=True)) / std
corr = z @ z.T / n_samples        # O(n²) not O(n³)
```

### Modality 3 — Mutational Co-occurrence

```python
# Fisher's exact test on 2×2 contingency tables
# Score = -log10(p-value) → higher = more co-mutated
_, p = fisher_exact([[a, b], [c, d]])
score = -np.log10(p + 1e-300)
```

### Composite Score

```
R(i,j) = ⅓ × norm(S_emb) + ⅓ × norm(S_expr) + ⅓ × norm(S_mut)
```

### Bootstrap Validation

```python
# Non-parametric bootstrap (n=50 pilot, n=1000 recommended)
# 95% CI from 2.5th–97.5th percentiles
# Empirical p-value = fraction of boots ≥ observed score
```

---

## 📁 Repository Structure

```
brca-kinase-redundancy/
│
├── 📄 brca_pipeline_bootstrap_pcs.py   # Main pipeline
├── 📄 docker-compose.yml               # Infrastructure
├── 📄 README.md                        # This file
├── 📄 requirements.txt                 # Python deps
│
├── 📂 Results/
│   ├── 🖼️  brca_redundancy_heatmap.png
│   ├── 🖼️  brca_network_rtk_rtk.png
│   ├── 🖼️  brca_network_nrtk_nrtk.png
│   ├── 🖼️  brca_network_rtk_nrtk.png
│   ├── 🖼️  brca_volcano.png
│   ├── 🖼️  brca_bootstrap_stats.png
│   ├── 🖼️  brca_score_distribution.png
│   ├── 📊  significant_pairs.csv
│   └── 📝  BRCA_Kinase_Redundancy_Report.docx
│
└── 📂 Data/                            # Not tracked (too large)
    ├── expression_data_processed.csv
    ├── mutation_data_processed.csv
    ├── kinase_sequences.csv
    └── tcga_brca.db
```

---

## 📦 Docker Services

| Service | Image | Port | Role |
|---------|-------|------|------|
| **Doris** | `dyrnq/doris:2.1.7` | 9030, 8030 | Analytical storage |
| **NiFi** | `apache/nifi:1.25.0` | 8443, 8888 | ETL orchestration |
| **Qdrant** | `qdrant/qdrant:v1.9.2` | 6333, 6334 | Vector search |

---

## 📈 Key Findings

| Finding | Detail |
|---------|--------|
| **Kinase pairs evaluated** | 903 unique pairs |
| **Score range** | 0.08 – 0.65 |
| **75th percentile** | 0.418 |
| **90th percentile** | 0.458 |
| **Mean 95% CI width** | 0.039 (stable estimates) |
| **Top RTK cluster** | ERBB2 · ERBB3 · ERBB4 (HER family) |
| **Top NRTK cluster** | LCK · LYN · HCK · FGR · FYN (SRC family) |
| **Top RTK hub** | PDGFRB (connected to FLT1, FLT4, KDR) |

---

## 🛠️ Troubleshooting

<details>
<summary><b>Doris BE not starting</b></summary>

```bash
# Disable swap first
sudo swapoff -a
# Set file descriptor limit and start
docker exec -it doris bash -c \
  "ulimit -n 655350 && /opt/apache-doris/be/bin/start_be.sh --daemon"
# Verify
docker exec -it doris mysql -h 127.0.0.1 -P 9030 -u root \
  -e "SHOW BACKENDS\G" | grep Alive
```
</details>

<details>
<summary><b>ESM-2 out of memory</b></summary>

```python
# Reduce batch size in pipeline
EMBED_BATCH = 2   # default is 4
# Or use smaller ESM-2 model
EMBED_MODEL = "facebook/esm2_t12_35M_UR50D"
```
</details>

<details>
<summary><b>Fish shell compatibility</b></summary>

```fish
# Fish shell: use set -x instead of export
set -x ANTHROPIC_API_KEY "your-key-here"
# Run commands separately (no && chaining)
docker exec -it doris /opt/apache-doris/be/bin/start_be.sh --daemon
sleep 20
```
</details>

---

## 📚 References

- TCGA Network (2012). *Nature*, 490, 61–70.
- Lin et al. (2023). ESM-2. *Science*, 379, 1123–1130.
- Efron & Tibshirani (1994). *An Introduction to the Bootstrap*.
- Yeatman (2004). A renaissance for SRC. *Nature Reviews Cancer*, 4, 470–480.

---

## 👤 Author

**Md Tariqul Islam**

[![GitHub](https://img.shields.io/badge/GitHub-mtariqi-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mtariqi)

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>
</div>
