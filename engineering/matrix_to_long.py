"""Convert a gene-by-sample matrix to long CSV with explicit sample columns.

This is for one row per gene. Peptide-level CPTAC input must first be
aggregated to gene level with a separately documented method.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path


def convert(source: Path, destination: Path, id_column: str,
            sample_columns: list[str], value_column: str,
            separator: str = ",") -> dict:
    if separator not in {",", "\t"} or not sample_columns:
        raise ValueError("Specify a supported separator and at least one sample")
    if len(sample_columns) != len(set(sample_columns)):
        raise ValueError("Sample column names must be unique")
    if value_column not in {"expression", "abundance"}:
        raise ValueError("value column must be expression or abundance")
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample_id_column = "sample_id" if value_column == "expression" else "cptac_sample_id"
    gene_id_column = "gene" if value_column == "expression" else "ensembl_gene_id"
    temp_path = None
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as src:
            reader = csv.DictReader(src, delimiter=separator)
            headers = reader.fieldnames or []
            if len(headers) != len(set(headers)):
                raise ValueError("Duplicate matrix headers")
            missing = set([id_column] + sample_columns) - set(headers)
            if missing:
                raise ValueError(f"Missing matrix columns: {sorted(missing)}")
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=destination.parent,
                prefix=".matrix-", suffix=".csv", delete=False
            ) as dest:
                temp_path = Path(dest.name)
                writer = csv.writer(dest)
                writer.writerow([sample_id_column, gene_id_column, value_column])
                genes = set()
                rows = 0
                missing_values = 0
                for line, record in enumerate(reader, 2):
                    if None in record or any(record[s] is None for s in sample_columns):
                        raise ValueError(f"Malformed matrix line {line}")
                    gene = (record[id_column] or "").strip()
                    if not gene or gene in genes:
                        raise ValueError(f"Empty or duplicate gene id on line {line}: {gene!r}")
                    genes.add(gene)
                    for sample in sample_columns:
                        value = record[sample].strip()
                        if not value:
                            missing_values += 1
                            continue
                        try:
                            numeric = float(value)
                        except ValueError as exc:
                            raise ValueError(f"Non-numeric value on line {line}, {sample}") from exc
                        if not math.isfinite(numeric):
                            raise ValueError(f"Non-finite value on line {line}, {sample}")
                        writer.writerow([sample, gene, numeric])
                        rows += 1
        if not rows:
            raise ValueError("No measurements in matrix")
        os.replace(temp_path, destination)
        return {"genes": len(genes), "samples": len(sample_columns),
                "measurements": rows, "missing_values": missing_values}
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--sample-columns-file", type=Path, required=True,
                        help="One exact sample column name per line")
    parser.add_argument("--value-column", choices=["expression", "abundance"], required=True)
    parser.add_argument("--tsv", action="store_true")
    args = parser.parse_args()
    sample_columns = [
        line.strip() for line in args.sample_columns_file.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    print(convert(args.input, args.output, args.id_column, sample_columns,
                  args.value_column, "\t" if args.tsv else ","))


if __name__ == "__main__":
    main()
