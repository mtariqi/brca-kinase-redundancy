"""Check source drift and export a metadata-only lineage inventory."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from .ingest import digest


def audit(database: Path) -> list[dict]:
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT dataset_id, cohort, modality, source_path, sha256, rows, "
            "source_version, ingested_at FROM datasets ORDER BY dataset_id"
        ).fetchall()
        result = []
        for dataset_id, cohort, modality, source, saved_hash, count, version, ingested in rows:
            path = Path(source)
            current = digest(path) if path.is_file() else None
            result.append({
                "dataset_id": dataset_id, "cohort": cohort, "modality": modality,
                "rows": count, "source_version": version, "ingested_at": ingested,
                "source_exists": current is not None,
                "source_unchanged": current == saved_hash if current is not None else None,
                "sha256": saved_hash,
            })
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("Data/lineage.db"))
    parser.add_argument("--output", type=Path, help="Optional metadata-only JSON output")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if a source changed or disappeared")
    args = parser.parse_args()
    result = audit(args.database)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.strict and any(item["source_unchanged"] is not True for item in result):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
