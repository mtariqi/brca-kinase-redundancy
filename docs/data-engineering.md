# Data engineering workflow

The repository now has a small, dependency-free data layer in `engineering/`.
It validates declared CSV/TSV files and keeps a SQLite catalog with the source
path, SHA-256 checksum, source version, cohort, modality, row count, schema,
contract checksum, and ingestion time. A dataset is unchanged on repeat runs.
Changed data or contract requires an explicit `--replace`. An invalid file
rolls back its own ingest; previously accepted datasets remain intact.

## Data flow

### Local GDC STAR and UMich exports

For the local folders reported on 2026-09-27, convert into private `Data/raw/`
without copying original GDC files into Git. These commands default to primary
tumor samples and `tpm_unstranded` (TPM is not raw counts):

```bash
python -m engineering.source_converters gdc \
  --root /home/mtariq/rtk_nrtk_tnbc/data/raw/tcga_brca/TCGA-BRCA \
  --sample-sheet /home/mtariq/rtk_nrtk_tnbc/data/raw/tcga_brca/gdc_sample_sheet.tsv \
  --output Data/raw/tcga/expression_long.csv --sample-key file-id
python -m engineering.source_converters umich \
  --input /home/mtariq/breast_cancer_proteogenomics/clean_cptac_data/BRCA_UMICH_proteomics_RTK_NRTK.csv \
  --output Data/raw/cptac/protein_long.csv
```

The STAR converter uses `File ID` and `File Name` from the GDC sample sheet,
excludes STAR summary rows, and rejects duplicate gene symbols or mismatched
gene sets. When multiple GDC files have the same
`Sample ID`, `--sample-key file-id` preserves them as distinct technical files;
the sample sheet is the required file-to-biological-sample mapping. Do not
interpret file IDs as distinct patients or materialize against mutation sample
IDs until you choose one file per biological sample using a recorded rule.
The UMich converter selects columns matching the observed `11BR047` style
sample identifiers and ignores annotation fields such as gene symbols.
Check that all intended sample columns match this convention before ingest.
Check `Sample Type` values in the sheet if you need another subset, and record
the GDC release. Raw counts
can be selected with `--measurement unstranded`; do not mix counts and TPM in
one analysis. UMich `Gene` is treated as an Ensembl identifier and the
normalized abundance is preserved in its source scale. The converter refuses
repeated gene IDs, which require a documented protein-group aggregation rule.
The full `BRCA_UMICH_proteomics.csv` can be selected instead of the kinase
subset when appropriate. Neither conversion implies a matched TCGA/CPTAC
patient cohort or independently reproduces published results.

Copy `config/datasets.example.json` to `Data/datasets.json`, record the real
source release/version, and ingest the generated files as described below.
The separate mutation, sequence and pair inputs must be prepared and validated
before running `engineering.materialize`.

1. Keep raw source files outside Git under `Data/raw/`. Record each source
   release, permitted use, and normalization in your run notes.
2. Convert matrices to the *declared long format* outside this ingest step.
   Never label peptide intensity as gene abundance without an explicit
   aggregation step and documented method.
   For a gene-by-sample matrix, list the exact sample columns in a text file
   (one per line), then run:

   ```bash
   python -m engineering.matrix_to_long --input Data/raw/tcga/expression_matrix.csv \
     --output Data/raw/tcga/expression_long.csv --id-column gene \
     --sample-columns-file Data/tcga_sample_columns.txt --value-column expression
   ```

   Missing matrix cells are skipped and counted; the converter refuses
   duplicate gene rows. Use `--tsv` for tab-delimited matrices.
3. Copy `config/datasets.example.json` to `Data/datasets.json`; adjust paths,
   column names, keys, cohorts, and source versions to match the real files.
   The example cohort names are placeholders, not assertions that releases match.
4. Ingest and validate:

   ```bash
   python -m engineering.ingest --manifest Data/datasets.json \
     --input-root Data/raw --database Data/lineage.db
   ```

5. If the four Python-pipeline inputs pass validation and refer to the same
   declared cohort, and RNA/mutation sample IDs match exactly, materialize
   the legacy SQLite schema:

   ```bash
   python -m engineering.materialize --source Data/lineage.db \
     --destination Data/tcga_brca.db --cohort tcga_brca_release_1 \
     --expression tcga_brca_expression_v1 \
     --mutation tcga_brca_mutation_v1 \
     --sequences kinase_sequences_v1 --pairs rtk_nrtk_pairs_v1
   ```

6. Run `python brca_pipeline_bootstrap_pcs.py` only after confirming the input
   cohort and scientific settings. ESM-2 model weights may require a separate
   download. The data layer does not reproduce or validate historical figures.
   The materializer expects kinase-focused long-form input; filter a full
   genome-wide matrix with a documented kinase list before this step.

## Contracts and joins

| Input | Required key | Measurement |
| --- | --- | --- |
| TCGA RNA | `sample_id, gene` | finite numeric `expression` |
| TCGA mutation | `sample_id, gene` | boolean `mutated` |
| Kinase reference | `gene` | `sequence` |
| Pair list | `RTK, NRTK` | pair membership |
| CPTAC protein | `cptac_sample_id, ensembl_gene_id` | finite numeric `abundance` |

The validated CPTAC dataset stays separate from TCGA. Patient-level RNA–protein
analysis requires a documented, one-to-one or explicitly adjudicated sample
map; shared gene names alone do not justify pairing samples. Do not combine
the 1,082-sample Python cohort, 1,224-sample R report, or TNBC subset by row
number or by assuming their sample IDs match.

The uploaded analyses include exploratory scripts with fixed paths and a
potential RNA–protein merge error. They are not invoked automatically. Run
them only after adapting input paths, verifying mappings, and reviewing their
statistical methods. Docker Compose offers NiFi, Doris, and Qdrant as optional
services; this SQLite workflow does not imply those services were used to
produce the reported findings.

## Inspection

```bash
sqlite3 Data/lineage.db 'SELECT dataset_id, cohort, modality, rows, sha256 FROM datasets;'
python -m engineering.audit --database Data/lineage.db --strict
python -m unittest discover -s tests -v
```

The catalog holds parsed rows as JSON for audit and export. Keep `Data/`
private and backed up according to the data use agreement. Do not commit raw
patient-level data or identifiers to Git.

## Optional services

The existing Compose stack is not required for validation or local analysis.
Before starting it, copy `.env.example` to `.env` and set a unique NiFi
password. Its published ports are bound to localhost. NiFi can access
`Data/` through its existing mount, but no NiFi flow or Doris/Qdrant load is
shipped or claimed as part of the historical analysis. Review security and
data access controls before using these services with patient-level records.
