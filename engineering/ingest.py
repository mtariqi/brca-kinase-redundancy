"""Validate delimited datasets and atomically register them in SQLite.

Only explicitly declared files are read. This module never infers that TCGA and
CPTAC samples are paired, and never transforms scientific measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

TYPES = {"str", "float", "int", "bool"}
SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS datasets (
  dataset_id TEXT PRIMARY KEY,
  cohort TEXT NOT NULL,
  modality TEXT NOT NULL,
  source_path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  rows INTEGER NOT NULL,
  columns_json TEXT NOT NULL,
  ingested_at TEXT NOT NULL,
  source_version TEXT NOT NULL,
  config_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
  dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
  row_number INTEGER NOT NULL,
  record_json TEXT NOT NULL,
  PRIMARY KEY (dataset_id, row_number)
);
CREATE INDEX IF NOT EXISTS observations_dataset ON observations(dataset_id);
"""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_contract(spec: dict) -> None:
    required = {"id", "file", "cohort", "modality", "source_version", "columns", "key"}
    missing = required - spec.keys()
    if missing:
        raise ValueError(f"Dataset contract missing: {sorted(missing)}")
    if not spec["id"] or not isinstance(spec["columns"], dict) or not spec["columns"]:
        raise ValueError("Dataset id and columns must be nonempty")
    if not spec["cohort"] or not spec["source_version"]:
        raise ValueError("Cohort and source_version must be nonempty")
    if not isinstance(spec["key"], list) or not spec["key"]:
        raise ValueError("key must be a nonempty list of column names")
    if set(spec["key"]) - set(spec["columns"]):
        raise ValueError("Every key field must be declared in columns")
    if set(spec["columns"].values()) - TYPES:
        raise ValueError(f"Supported types are {sorted(TYPES)}")
    if spec.get("delimiter", ",") not in {",", "\t"}:
        raise ValueError("delimiter must be comma or tab")


def value_as_type(value: str, kind: str, column: str, line: int):
    value = value.strip()
    if value == "":
        return None
    try:
        if kind == "str":
            return value
        if kind == "float":
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("non-finite value")
            return result
        if kind == "int":
            if not value.lstrip("+-").isdigit():
                raise ValueError("not an integer")
            return int(value)
        if value.lower() in {"1", "true", "yes"}:
            return True
        if value.lower() in {"0", "false", "no"}:
            return False
        raise ValueError("not a boolean")
    except ValueError as exc:
        raise ValueError(f"line {line}, {column}: {exc}") from exc


def ingest_one(connection: sqlite3.Connection, spec: dict, root: Path,
               config_hash: str, replace: bool = False) -> dict:
    validate_contract(spec)
    path = (root / spec["file"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{spec['id']}: file escapes input root")
    if not path.is_file():
        raise FileNotFoundError(path)
    source_hash = digest(path)
    previous = connection.execute(
        "SELECT sha256, config_sha256 FROM datasets WHERE dataset_id = ?",
        (spec["id"],),
    ).fetchone()
    if previous and not replace:
        if previous == (source_hash, config_hash):
            return {"id": spec["id"], "status": "unchanged"}
        raise ValueError(f"{spec['id']}: source or contract changed; pass --replace")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=spec.get("delimiter", ","))
        header = reader.fieldnames
        if not header or len(set(header)) != len(header):
            raise ValueError(f"{spec['id']}: missing or duplicate header")
        missing = set(spec["columns"]) - set(header)
        if missing:
            raise ValueError(f"{spec['id']}: missing columns {sorted(missing)}")
        count = 0
        with connection:
            # Keep uniqueness state on disk; TCGA long-form matrices may have
            # millions of sample-gene rows.
            connection.execute("CREATE TEMP TABLE IF NOT EXISTS ingest_keys (key TEXT PRIMARY KEY)")
            connection.execute("DELETE FROM ingest_keys")
            if previous:
                connection.execute("DELETE FROM datasets WHERE dataset_id = ?", (spec["id"],))
            connection.execute(
                """INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (spec["id"], spec["cohort"], spec["modality"], str(path),
                 source_hash, 0, json.dumps(header), datetime.now(timezone.utc).isoformat(),
                 spec["source_version"], config_hash),
            )
            batch = []
            for line, raw in enumerate(reader, 2):
                if None in raw:
                    raise ValueError(f"{spec['id']}: extra fields on line {line}")
                if any(raw[col] is None for col in spec["columns"]):
                    raise ValueError(f"{spec['id']}: missing fields on line {line}")
                record = {
                    col: value_as_type(raw[col], typ, col, line)
                    for col, typ in spec["columns"].items()
                }
                key = tuple(record[col] for col in spec["key"])
                if any(v is None for v in key):
                    raise ValueError(f"{spec['id']}: empty key on line {line}")
                try:
                    connection.execute("INSERT INTO ingest_keys VALUES (?)", (json.dumps(key),))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        f"{spec['id']}: duplicate key {key} on line {line}"
                    ) from exc
                count += 1
                batch.append((spec["id"], count, json.dumps(record, sort_keys=True)))
                if len(batch) >= 1000:
                    connection.executemany("INSERT INTO observations VALUES (?, ?, ?)", batch)
                    batch.clear()
            if batch:
                connection.executemany("INSERT INTO observations VALUES (?, ?, ?)", batch)
            if count == 0:
                raise ValueError(f"{spec['id']}: no records")
            if "expected_rows" in spec and count != spec["expected_rows"]:
                raise ValueError(f"{spec['id']}: expected {spec['expected_rows']} rows, found {count}")
            connection.execute("UPDATE datasets SET rows = ? WHERE dataset_id = ?",
                               (count, spec["id"]))
    return {"id": spec["id"], "status": "ingested", "rows": count, "sha256": source_hash}


def run(manifest: Path, input_root: Path, database: Path, replace: bool = False) -> list[dict]:
    config = json.loads(manifest.read_text(encoding="utf-8"))
    specs = config.get("datasets")
    if not isinstance(specs, list) or not specs:
        raise ValueError("manifest must contain a nonempty datasets list")
    ids = [s.get("id") for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset ids must be unique")
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    try:
        connection.executescript(SCHEMA)
        config_hash = digest(manifest)
        return [ingest_one(connection, s, input_root, config_hash, replace) for s in specs]
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=Path("Data/lineage.db"))
    parser.add_argument("--replace", action="store_true",
                        help="Replace changed datasets only after validation")
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.input_root, args.database, args.replace), indent=2))


if __name__ == "__main__":
    main()
