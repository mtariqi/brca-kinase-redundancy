# TCGA-BRCA RTK/NRTK co-expression analysis (September 2026)

This page records results from **RTK_Redundancy_Full_Report.html**, dated
September 21, 2026. The report describes an R analysis of 1,224 TCGA-BRCA
samples. It is a separate analysis from the Python ESM-2/expression/mutation
pipeline documented in the main README (1,082 samples and a composite score).
The scores and cohort sizes should not be compared as if they came from one run.

## Reported results

The report ranks RTKs by a BH-filtered co-expression redundancy score. These
are *expression association* rankings, not measurements of functional
compensation, therapeutic resistance, or drug synergy.

| RTK | BH redundancy score | Mean correlation | Significant partners (BH) |
| --- | ---: | ---: | ---: |
| PDGFRA | 2.436 | 0.271 | 9 |
| PDGFRB | 2.411 | 0.301 | 8 |
| KDR | 2.068 | 0.295 | 7 |
| EGFR | 2.005 | 0.251 | 8 |
| FLT1 | 1.863 | 0.266 | 7 |

In the combined RTK/NRTK network, the leading nodes by degree were LYN (24),
FYN (22), JAK1 (22), PDGFRA (21), MERTK (21), and ABL1 (20).
Degree counts depend on the network's edge definition and do not establish
causal signaling relationships.

The high and low redundancy survival groups each contained 546 patients.
Their log-rank comparison was **p = 0.7297**; the report therefore provides no
evidence of a survival difference under this grouping. Subtype analysis was
skipped, so these results should not be described as TNBC-specific.

## Validation status and interpretation

The report applies BH, Bonferroni, Storey q-value, permutation, Monte Carlo,
and bootstrap procedures. Some validation outputs need correction before
being used to claim calibrated error rates:

- The Monte Carlo table reports FDR around 0.73 and FWER of 1.00 across its
  four rules. Inspection of the corresponding
  `08_comprehensive_empirical_statistics.R` script shows that simulated null
  and alternative p-values are **shuffled**, but the error-rate function
  labels the *last* columns as nulls. Those labels no longer identify the
  generated nulls after shuffling. Consequently, its false-positive,
  true-positive, FDR, and FWER summaries are invalid as estimates of the
  reported discoveries' error rates. The 0.73 value should not be interpreted
  as an observed false-discovery fraction for the network.
- The same script splits samples at the median of the sum of their RTK
  expression, then tests RTK expression between those groups with mt.maxT.
  Because the grouping is constructed from the tested expression variables,
  that permutation result is not independent validation of co-expression
  edges.
- Bootstrap intervals quantify variation under the implemented resampling
  scheme; they do not demonstrate that one kinase replaces another when
  inhibited.

**Next checks:** retain null/alternative labels through simulation and
recalculate error rates; define independent phenotypes for permutation tests;
assess expression effects from cell composition and tumor subtype; then
validate prioritized pairs with perturbation or drug-response data.

### Provenance

Source: *RTK/NRTK Redundancy Analysis — TCGA BRCA*, Md Tariqul Islam,
September 21, 2026, sections 1–8; implementation reviewed in
`08_comprehensive_empirical_statistics.R`. The source report and R workflow
are not included in this repository revision. The values above are
transcribed from that report and have not been recomputed here.
