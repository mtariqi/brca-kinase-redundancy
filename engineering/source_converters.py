"""Convert local GDC STAR and UMich protein exports to declared long CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from pathlib import Path


def _atomic_csv(destination: Path, columns: list[str], write_rows):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=destination.parent, prefix=".convert-",
                                         suffix=".csv", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(columns)
            result = write_rows(writer)
        os.replace(temporary, destination)
        return result
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def convert_gdc(root: Path, sample_sheet: Path, destination: Path,
                measurement: str = "tpm_unstranded", sample_type: str = "Primary Tumor",
                sample_key: str = "sample-id") -> dict:
    """Require unique GDC file/sample mappings; retain gene symbols for kinase analyses."""
    if measurement not in {"unstranded", "tpm_unstranded", "fpkm_unstranded", "fpkm_uq_unstranded"}:
        raise ValueError("Unsupported STAR measurement")
    if destination.resolve() in (root.resolve(), sample_sheet.resolve()):
        raise ValueError("Output would overwrite input")

    with sample_sheet.open(newline="", encoding="utf-8-sig") as handle:
        sheet = csv.DictReader(handle, delimiter="\t")
        required = {"File ID", "File Name", "Sample ID", "Sample Type"}
        if not required.issubset(sheet.fieldnames or []):
            raise ValueError(f"Missing sample sheet columns: {required - set(sheet.fieldnames or [])}")
        records = [r for r in sheet if r["Sample Type"] == sample_type
                   and r["File Name"].endswith(".rna_seq.augmented_star_gene_counts.tsv")]
    if not records:
        raise ValueError(f"No STAR files with sample type {sample_type!r}")
    ids = [r["Sample ID"] for r in records]
    filenames = [r["File ID"] for r in records]
    if sample_key not in {"sample-id", "file-id"}:
        raise ValueError("sample_key must be sample-id or file-id")
    if any(not x for x in ids + filenames) or len(set(filenames)) != len(filenames):
        raise ValueError("Empty ID or repeated file ID in sample sheet")
    if sample_key == "sample-id" and len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample ID: use --sample-key file-id to preserve distinct files")
    paths = {}
    for path in root.rglob("*.rna_seq.augmented_star_gene_counts.tsv"):
        paths.setdefault((path.parent.name, path.name), []).append(path)

    def write(writer):
        total = 0
        expected_genes = None
        for record in sorted(records, key=lambda r: (r["Sample ID"], r["File ID"])):
            matches = paths.get((record["File ID"], record["File Name"]), [])
            if len(matches) != 1 or matches[0].resolve() == destination.resolve():
                raise ValueError(f"Expected one GDC file for {record['File ID']}: found {len(matches)}")
            with matches[0].open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t")
                required = {"gene_id", "gene_name", measurement}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f"Missing STAR columns in {matches[0]}")
                genes = set()
                for row in reader:
                    gene = row["gene_name"].strip()
                    if row["gene_id"].startswith("N_") or not gene:
                        continue
                    if gene in genes:
                        raise ValueError(f"Duplicate gene symbol {gene!r} in {matches[0]}; resolve gene mapping")
                    genes.add(gene)
                    value = float(row[measurement])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"Invalid {measurement} for {gene!r}")
                    writer.writerow((record["Sample ID"] if sample_key == "sample-id" else record["File ID"], gene, value))
                    total += 1
                if not genes or (expected_genes is not None and genes != expected_genes):
                    raise ValueError(f"Empty or inconsistent gene set in {matches[0]}")
                expected_genes = genes
        return {"samples": len(records), "genes_per_sample": len(expected_genes), "measurements": total,
                "unit": measurement, "sample_type": sample_type, "sample_key": sample_key,
                "duplicate_sample_ids": len(ids) - len(set(ids))}

    return _atomic_csv(destination, ["sample_id", "gene", "expression"], write)


def convert_umich(source: Path, destination: Path, *, delimiter: str = ",") -> dict:
    """Melt protein-group abundance; reject duplicate identifiers instead of aggregating."""
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output must differ")

    def write(writer):
        with source.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            fields = reader.fieldnames or []
            metadata = {"Index", "NumberPSM", "Gene", "MaxPepProb", "ReferenceIntensity"}
            if len(fields) != len(set(fields)) or not metadata.issubset(fields):
                raise ValueError("Unexpected or duplicate UMich headers")
            # Other annotation columns, including gene symbols, can be present.
            # UMich sample identifiers in the supplied export have the form 11BR047.
            samples = [f for f in fields if re.fullmatch(r"[0-9]{2}BR[0-9]{3}", f)]
            if not samples:
                raise ValueError("No abundance sample columns")
            genes = set()
            total = missing = 0
            for row in reader:
                gene = row["Gene"].strip()
                if not gene or gene in genes:
                    raise ValueError(f"Missing or repeated gene ID {gene!r}; aggregate protein groups explicitly")
                genes.add(gene)
                for sample in samples:
                    value = (row[sample] or "").strip()
                    if not value:
                        missing += 1
                        continue
                    abundance = float(value)
                    if not math.isfinite(abundance):
                        raise ValueError(f"Non-finite abundance for {sample}, {gene}")
                    writer.writerow((sample, gene, abundance))
                    total += 1
            if not total:
                raise ValueError("No protein abundances")
            return {"samples": len(samples), "genes": len(genes), "measurements": total,
                    "missing_values": missing, "unit": "source normalized abundance"}

    return _atomic_csv(destination, ["cptac_sample_id", "ensembl_gene_id", "abundance"], write)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="source", required=True)
    gdc = sub.add_parser("gdc")
    gdc.add_argument("--root", type=Path, required=True)
    gdc.add_argument("--sample-sheet", type=Path, required=True)
    gdc.add_argument("--output", type=Path, required=True)
    gdc.add_argument("--measurement", default="tpm_unstranded",
                     choices=["unstranded", "tpm_unstranded", "fpkm_unstranded", "fpkm_uq_unstranded"])
    gdc.add_argument("--sample-type", default="Primary Tumor")
    gdc.add_argument("--sample-key", choices=["sample-id", "file-id"], default="sample-id",
                     help="Use file-id to retain multiple files per biological sample without collapsing them")
    umich = sub.add_parser("umich")
    umich.add_argument("--input", type=Path, required=True)
    umich.add_argument("--output", type=Path, required=True)
    umich.add_argument("--tsv", action="store_true")
    args = parser.parse_args()
    if args.source == "gdc":
        print(convert_gdc(args.root, args.sample_sheet, args.output, args.measurement,
                          args.sample_type, args.sample_key))
    else:
        print(convert_umich(args.input, args.output, delimiter="\t" if args.tsv else ","))


if __name__ == "__main__":
    main()
