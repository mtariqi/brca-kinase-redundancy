"""Stream a declared kinase subset from a GDC-derived long CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


def extract(source: Path, sheet: Path, genes_file: Path, output: Path,
            report: Path, selected_file_ids: Path | None = None) -> dict:
    if len({p.resolve() for p in (source, sheet, genes_file, output, report)}) != 5:
        raise ValueError("Input, output and report paths must be distinct")
    genes_list = [line.strip() for line in genes_file.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
    if not genes_list or len(genes_list) != len(set(genes_list)):
        raise ValueError("Provide a nonempty, unique gene-symbol list")
    genes = set(genes_list)
    with sheet.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"File ID", "Sample ID", "Case ID", "Sample Type"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing GDC sample-sheet columns: {required - set(reader.fieldnames or [])}")
        rows = [row for row in reader if row["Sample Type"] == "Primary Tumor"
                and row.get("File Name", "").endswith(".rna_seq.augmented_star_gene_counts.tsv")]
    by_file = {}
    by_sample = defaultdict(list)
    for row in rows:
        file_id = row["File ID"]
        if not file_id or file_id in by_file or not row["Sample ID"]:
            raise ValueError("Missing or duplicate file ID, or missing sample ID")
        by_file[file_id] = row
        by_sample[row["Sample ID"]].append(file_id)
    duplicates = {key: sorted(values) for key, values in by_sample.items() if len(values) > 1}
    if selected_file_ids is None:
        selected = set(by_file)
    else:
        choices = [x.strip() for x in selected_file_ids.read_text().splitlines()
                   if x.strip() and not x.lstrip().startswith("#")]
        if not choices or len(choices) != len(set(choices)) or set(choices) - set(by_file):
            raise ValueError("Selected file IDs must be unique and present in the sample sheet")
        selected = set(choices)
        retained_samples = [by_file[x]["Sample ID"] for x in selected]
        if len(retained_samples) != len(set(retained_samples)):
            raise ValueError("Selection still includes multiple files for one biological sample")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=output.parent,
                                         prefix=".kinase-", suffix=".csv", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "gene", "expression"])
            counts = Counter()
            observed_files = set()
            with source.open(newline="", encoding="utf-8-sig") as input_handle:
                reader = csv.DictReader(input_handle)
                if reader.fieldnames != ["sample_id", "gene", "expression"]:
                    raise ValueError("Expected GDC long CSV columns: sample_id,gene,expression")
                for row in reader:
                    file_id = row["sample_id"]
                    if file_id not in by_file:
                        raise ValueError(f"File ID absent from GDC sheet: {file_id}")
                    observed_files.add(file_id)
                    if file_id in selected and row["gene"] in genes:
                        writer.writerow((by_file[file_id]["Sample ID"] if selected_file_ids else file_id,
                                         row["gene"], row["expression"]))
                        counts[row["gene"]] += 1
            if selected - observed_files:
                raise ValueError(f"Selected GDC files absent from input: {len(selected - observed_files)}")
            if not counts:
                raise ValueError("No listed genes found in expression data")
        result = {"requested_genes": len(genes), "found_genes": len(counts),
                  "missing_genes": sorted(genes - set(counts)), "measurements": sum(counts.values()),
                  "input_files": len(observed_files), "selected_files": len(selected),
                  "duplicate_sample_ids": duplicates,
                  "output_id_type": "sample-id" if selected_file_ids else "file-id",
                  "gene_measurements": dict(sorted(counts.items()))}
        # Write the audit report before publishing the extract; both stay private in Data/.
        report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        return result
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--sample-sheet", required=True, type=Path)
    parser.add_argument("--genes-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--selected-file-ids", type=Path,
                        help="One chosen file ID per line, at most one per biological sample")
    args = parser.parse_args()
    result = extract(args.input, args.sample_sheet, args.genes_file, args.output,
                     args.report, args.selected_file_ids)
    print({key: result[key] for key in ("requested_genes", "found_genes", "missing_genes",
                                       "measurements", "input_files", "selected_files",
                                       "output_id_type")})
    print(f"Review duplicate IDs and per-gene counts in {args.report}")


if __name__ == "__main__":
    main()
