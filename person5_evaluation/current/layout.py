"""Resolve pre-reorganisation provenance paths without rewriting frozen records."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "person5_evaluation/archive/structure_20261021"


@lru_cache(maxsize=1)
def migration_record():
    return json.loads((AUDIT / "migration.json").read_text(encoding="utf-8"))


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_recorded_path(recorded):
    """Map a historical project-relative path to its current physical location."""
    name = str(recorded).replace("\\", "/")
    current_prefixes = ("person5_evaluation/current", "person5_evaluation/archive",
                        "person5_evaluation/archive_v1", "person3_handoff",
                        "person2_handoff/archives", "person4_handoff/archives")
    if not any(name == prefix or name.startswith(prefix + "/") for prefix in current_prefixes):
        for previous, current in sorted(migration_record()["path_mappings"], key=lambda item: len(item[0]), reverse=True):
            if name == previous or name.startswith(previous + "/"):
                name = current + name[len(previous):]
                break
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Recorded path escapes the project: {recorded}")
    return path


def verify_recorded_hash(recorded, expected):
    """Require original bytes, or an explicitly recorded path-only amendment."""
    path = resolve_recorded_path(recorded)
    actual = file_sha256(path)
    if actual == expected:
        return
    amendment = migration_record()["code_amendments"].get(str(recorded))
    if amendment is None or amendment["before_sha256"] != expected or amendment["after_sha256"] != actual:
        raise ValueError(f"Frozen source changed outside the recorded path migration: {recorded}")
    if file_sha256(ROOT / amendment["original_copy"]) != expected:
        raise ValueError(f"Original frozen source backup changed: {recorded}")


def verify_migration_code():
    """Check the path adapter and all recorded code edits against the migration record."""
    record = migration_record()
    for name, expected in record["path_adapter_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"Path migration adapter changed: {name}")
    for name, amendment in record["code_amendments"].items():
        verify_recorded_hash(name, amendment["before_sha256"])
