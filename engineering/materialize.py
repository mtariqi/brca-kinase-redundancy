"""Build the legacy Python pipeline's SQLite tables from validated datasets."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def build(source: Path, destination: Path, cohort: str,
          expression: str, mutation: str, sequences: str, pairs: str) -> dict:
    names = {
        "expression": expression, "mutation": mutation,
        "sequences": sequences, "pairs": pairs,
    }
    if len(set(names.values())) != 4:
        raise ValueError("Four distinct dataset ids are required")
    source_db = sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = sqlite3.connect(destination)
    try:
        expected_modality = {
            "expression": "rna", "mutation": "mutation",
            "sequences": "sequence", "pairs": "pairs",
        }
        for role, dataset_id in names.items():
            row = source_db.execute(
                "SELECT cohort, modality FROM datasets WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
            if row is None or row[0] != cohort:
                raise ValueError(f"{role}: missing dataset or cohort mismatch ({dataset_id})")
            if row[1] != expected_modality[role]:
                raise ValueError(f"{role}: expected modality {expected_modality[role]}, got {row[1]}")
        def records(dataset_id):
            for (payload,) in source_db.execute(
                "SELECT record_json FROM observations WHERE dataset_id = ? ORDER BY row_number",
                (dataset_id,),
            ):
                yield json.loads(payload)

        expr = list(records(expression))
        mut = list(records(mutation))
        seq = list(records(sequences))
        prs = list(records(pairs))
        for role, rows, fields in (
            ("expression", expr, {"sample_id", "gene", "expression"}),
            ("mutation", mut, {"sample_id", "gene", "mutated"}),
            ("sequences", seq, {"gene", "sequence"}),
            ("pairs", prs, {"RTK", "NRTK"}),
        ):
            if not rows or not fields.issubset(rows[0]):
                raise ValueError(f"{role}: expected columns {sorted(fields)}")
        expr_samples = {r["sample_id"] for r in expr}
        mut_samples = {r["sample_id"] for r in mut}
        if expr_samples != mut_samples:
            raise ValueError("Expression and mutation sample IDs must match exactly")
        genes = {r["gene"] for r in expr} & {r["gene"] for r in mut} & {r["gene"] for r in seq}
        if not genes:
            raise ValueError("No genes overlap expression, mutation, and sequences")
        rtk_genes = {r["RTK"] for r in prs}
        nrtk_genes = {r["NRTK"] for r in prs}
        if rtk_genes & nrtk_genes or not genes <= rtk_genes | nrtk_genes:
            raise ValueError("Pair list must classify every shared kinase unambiguously")
        with target:
            target.executescript("""
                CREATE TABLE IF NOT EXISTS expression_raw (
                    sample_id TEXT, gene TEXT, expression REAL, cancer_type TEXT,
                    PRIMARY KEY (sample_id, gene)
                );
                CREATE TABLE IF NOT EXISTS mutation_raw (
                    sample_id TEXT, gene TEXT, mutated INTEGER, cancer_type TEXT,
                    PRIMARY KEY (sample_id, gene)
                );
                CREATE TABLE IF NOT EXISTS kinase_meta (
                    gene TEXT PRIMARY KEY, sequence TEXT
                );
                CREATE TABLE IF NOT EXISTS rtk_nrtk_pairs (
                    RTK TEXT, NRTK TEXT, PRIMARY KEY (RTK, NRTK)
                );
            """)
            for table in ("expression_raw", "mutation_raw", "kinase_meta", "rtk_nrtk_pairs"):
                target.execute(f"DELETE FROM {table}")
            target.executemany("INSERT INTO expression_raw VALUES (?, ?, ?, ?)",
                               ((r["sample_id"], r["gene"], r["expression"], "BRCA") for r in expr))
            target.executemany("INSERT INTO mutation_raw VALUES (?, ?, ?, ?)",
                               ((r["sample_id"], r["gene"], int(r["mutated"]), "BRCA") for r in mut))
            target.executemany("INSERT INTO kinase_meta VALUES (?, ?)",
                               ((r["gene"], r["sequence"]) for r in seq))
            target.executemany("INSERT INTO rtk_nrtk_pairs VALUES (?, ?)",
                               ((r["RTK"], r["NRTK"]) for r in prs))
        return {"cohort": cohort, "tables": {k: len(v) for k, v in
                (("expression_raw", expr), ("mutation_raw", mut),
                 ("kinase_meta", seq), ("rtk_nrtk_pairs", prs))}}
    finally:
        source_db.close()
        target.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Data/lineage.db"))
    parser.add_argument("--destination", type=Path, default=Path("Data/tcga_brca.db"))
    parser.add_argument("--cohort", required=True)
    for role in ("expression", "mutation", "sequences", "pairs"):
        parser.add_argument(f"--{role}", required=True, help=f"dataset id for {role}")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.destination, args.cohort,
                           args.expression, args.mutation, args.sequences, args.pairs), indent=2))


if __name__ == "__main__":
    main()
