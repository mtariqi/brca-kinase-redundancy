import csv
import tempfile
import unittest
from pathlib import Path

from engineering.source_converters import convert_gdc, convert_umich


class SourceConverterTests(unittest.TestCase):
    def test_gdc_sample_mapping_and_star_comments(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "counts" / "file-1"
            source.mkdir(parents=True)
            (source / "counts.rna_seq.augmented_star_gene_counts.tsv").write_text(
                "# gene-model: GENCODE v36\n"
                "gene_id\tgene_name\tunstranded\ttpm_unstranded\n"
                "N_unmapped\t\t100\t0\nENSG1.1\tEGFR\t10\t2.5\n")
            sheet = root / "sheet.tsv"
            sheet.write_text("File ID\tFile Name\tSample ID\tSample Type\n"
                             "file-1\tcounts.rna_seq.augmented_star_gene_counts.tsv\tS1\tPrimary Tumor\n")
            output = root / "expression.csv"
            result = convert_gdc(root / "counts", sheet, output)
            self.assertEqual(result["measurements"], 1)
            with output.open() as handle:
                self.assertEqual(list(csv.reader(handle))[1], ["S1", "EGFR", "2.5"])

    def test_umich_repeated_gene_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "protein.csv"
            source.write_text("Index,NumberPSM,Gene,MaxPepProb,ReferenceIntensity,11BR047\n"
                              "a,1,ENSG1,1,2,27.1\nb,2,ENSG1,1,2,27.2\n")
            output = root / "protein_long.csv"
            with self.assertRaisesRegex(ValueError, "repeated gene"):
                convert_umich(source, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
