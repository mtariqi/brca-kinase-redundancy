import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from engineering.ingest import run
from engineering.materialize import build
from engineering.audit import audit
from engineering.matrix_to_long import convert


class EngineeringIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "raw"
        self.input.mkdir()
        self.db = self.root / "lineage.db"
        self.dest = self.root / "tcga.db"
        files = {
            "expression": (["sample_id", "gene", "expression"],
                           [["S1", "EGFR", "1.25"], ["S2", "EGFR", "2.0"]]),
            "mutation": (["sample_id", "gene", "mutated"],
                         [["S1", "EGFR", "1"], ["S2", "EGFR", "0"]]),
            "sequence": (["gene", "sequence"], [["EGFR", "MKK"]]),
            "pairs": (["RTK", "NRTK"], [["EGFR", "LYN"]]),
        }
        self.specs = []
        for name, (header, rows) in files.items():
            with (self.input / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(rows)
            types = {col: "str" for col in header}
            if name == "expression":
                types["expression"] = "float"
            if name == "mutation":
                types["mutated"] = "bool"
            key = {"expression": ["sample_id", "gene"],
                   "mutation": ["sample_id", "gene"],
                   "sequence": ["gene"], "pairs": ["RTK", "NRTK"]}[name]
            self.specs.append(dict(id=name, file=f"{name}.csv", cohort="TCGA-A",
                                   modality={"expression": "rna", "mutation": "mutation",
                                             "sequence": "sequence", "pairs": "pairs"}[name],
                                   source_version="fixture-1",
                                   columns=types, key=key))
        self.manifest = self.root / "manifest.json"
        self.save_manifest()

    def save_manifest(self):
        self.manifest.write_text(json.dumps({"datasets": self.specs}))

    def test_ingest_idempotency_and_materialization(self):
        first = run(self.manifest, self.input, self.db)
        self.assertEqual([r["status"] for r in first], ["ingested"] * 4)
        self.assertEqual([r["status"] for r in run(self.manifest, self.input, self.db)],
                         ["unchanged"] * 4)
        self.assertTrue(all(row["source_unchanged"] for row in audit(self.db)))
        result = build(self.db, self.dest, "TCGA-A",
                       "expression", "mutation", "sequence", "pairs")
        self.assertEqual(result["tables"]["expression_raw"], 2)
        with sqlite3.connect(self.dest) as connection:
            self.assertEqual(connection.execute(
                "SELECT sample_id, gene, expression FROM expression_raw ORDER BY sample_id"
            ).fetchall(), [("S1", "EGFR", 1.25), ("S2", "EGFR", 2.0)])
            self.assertEqual(connection.execute(
                "SELECT mutated FROM mutation_raw ORDER BY sample_id"
            ).fetchall(), [(1,), (0,)])

    def test_rejects_duplicate_keys_without_destroying_previous_data(self):
        run(self.manifest, self.input, self.db)
        with (self.input / "expression.csv").open("a") as stream:
            stream.write("S1,EGFR,3\n")
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            run(self.manifest, self.input, self.db, replace=True)
        with sqlite3.connect(self.db) as connection:
            self.assertEqual(connection.execute(
                "SELECT rows FROM datasets WHERE dataset_id = 'expression'"
            ).fetchone()[0], 2)

    def test_rejects_cohort_mismatch_and_invalid_values(self):
        run(self.manifest, self.input, self.db)
        with self.assertRaisesRegex(ValueError, "cohort mismatch"):
            build(self.db, self.dest, "CPTAC-B",
                  "expression", "mutation", "sequence", "pairs")
        with (self.input / "expression.csv").open("a") as stream:
            stream.write("S3,EGFR,NaN\n")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            run(self.manifest, self.input, self.db, replace=True)

    def test_matrix_conversion_keeps_declared_samples_only(self):
        matrix = self.input / "matrix.csv"
        matrix.write_text("gene,S1,S2,clinical_label\nEGFR,1.0,,tumor\nLYN,2.0,3.5,normal\n")
        output = self.input / "long.csv"
        report = convert(matrix, output, "gene", ["S1", "S2"], "expression")
        self.assertEqual(report["missing_values"], 1)
        self.assertEqual(report["measurements"], 3)
        self.assertEqual(output.read_text().splitlines()[1:], [
            "S1,EGFR,1.0", "S1,LYN,2.0", "S2,LYN,3.5",
        ])


if __name__ == "__main__":
    unittest.main()
