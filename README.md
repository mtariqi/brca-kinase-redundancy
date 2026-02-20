# RTK-NRTK Kinase Redundancy Analysis — BRCA

A multi-modal computational pipeline to identify functionally redundant 
receptor and non-receptor tyrosine kinase pairs in breast cancer (TCGA-BRCA).

## Overview

Kinase redundancy drives therapeutic resistance. This pipeline quantifies 
redundancy across 43 tyrosine kinases (23 RTKs, 20 NRTKs) in 1,082 
BRCA tumour samples using three complementary modalities:

- **ESM-2** protein language model embeddings (sequence similarity)
- **Pearson correlation** of RNA-seq expression profiles (co-expression)
- **Fisher's exact test** on somatic mutation co-occurrence

Scores are validated with bootstrap resampling (95% CI, empirical p-values).

## Infrastructure

| Component     | Role                          |
|---------------|-------------------------------|
| Apache NiFi   | ETL pipeline for TCGA data    |
| Apache Doris  | Analytical columnar database  |
| Qdrant        | Vector search (ESM-2 embeddings) |
| SQLite        | Lightweight local storage     |
| Docker Compose| Container orchestration       |

## Quick Start

### 1. Prerequisites
```bash
# Disable swap (required for Doris BE)
sudo swapoff -a
sudo sysctl -w vm.max_map_count=2000000

# Start containers
docker-compose up -d
```

### 2. Install Python dependencies
```bash
pip install pandas numpy scipy scikit-learn networkx matplotlib \
            torch transformers tqdm qdrant-client anthropic pymysql
```

### 3. Load data
```bash
python -c "
import pandas as pd, sqlite3
conn = sqlite3.connect('Data/tcga_brca.db')
# Load your TCGA CSVs here
conn.close()
"
```

### 4. Run pipeline
```bash
python brca_pipeline.py
```

## Output Files (Results/)

| File | Description |
|------|-------------|
| `brca_redundancy_heatmap.png` | 43×43 kinase redundancy matrix |
| `brca_network_rtk_rtk.png` | RTK-RTK redundancy network |
| `brca_network_nrtk_nrtk.png` | NRTK-NRTK redundancy network |
| `brca_network_rtk_nrtk.png` | Cross-type redundancy network |
| `brca_volcano.png` | Volcano plot (score vs p-value) |
| `brca_bootstrap_stats.png` | Bootstrap CI distribution |
| `brca_pca_overview.png` | PCA across all modalities |
| `brca_pca_biplot.png` | PCA biplot with loadings |
| `brca_pca_separation.png` | RTK/NRTK class separation |
| `significant_pairs.csv` | Statistically significant pairs |

## Data

Data sourced from [TCGA-BRCA](https://portal.gdc.cancer.gov/).  
**Note:** Raw data files are not included due to size. Download from GDC portal.

## Citation

If you use this pipeline, please cite:
- TCGA Network (2012). Nature, 490, 61–70.
- Lin et al. (2023). ESM-2. Science, 379, 1123–1130.

