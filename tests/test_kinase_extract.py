import csv
import tempfile
import unittest
from pathlib import Path

from engineering.kinase_extract import extract


class KinaseExtractTests(unittest.TestCase):
    def test_duplicate_files_are_reported_then_resolved_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sheet = root / "sheet.tsv"
            sheet.write_text("File ID\tFile Name\tSample ID\tCase ID\tSample Type\n" + "".join(
                f"{file_id}\ta.rna_seq.augmented_star_gene_counts.tsv\tS1\tC1\tPrimary Tumor\n"
                for file_id in ("F1", "F2")))
            source = root / "long.csv"
            source.write_text("sample_id,gene,expression\nF1,EGFR,2\nF1,TP53,1\n"
                              "F2,EGFR,3\nF2,TP53,2\n")
            genes = root / "kinases.txt"
            genes.write_text("EGFR\nPDGFRA\n")
            output = root / "subset.csv"
            report = root / "report.json"
            result = extract(source, sheet, genes, output, report)
            self.assertEqual(result["duplicate_sample_ids"], {"S1": ["F1", "F2"]})
            self.assertEqual(result["missing_genes"], ["PDGFRA"])
            self.assertEqual(result["measurements"], 2)
            choices = root / "selected.txt"
            choices.write_text("F2\n")
            result = extract(source, sheet, genes, output, report, choices)
            self.assertEqual(result["output_id_type"], "sample-id")
            with output.open() as handle:
                self.assertEqual(list(csv.reader(handle))[1:], [["S1", "EGFR", "3"]])
            choices.write_text("F1\nF2\n")
            with self.assertRaisesRegex(ValueError, "multiple files"):
                extract(source, sheet, genes, output, report, choices)


if __name__ == "__main__":
    unittest.main()
