"""BlackWing dataset split helpers; only the knowledge-base split may be indexed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "BLACKWING_DATA"
KNOWLEDGE_PREFIX = "01_RAG_KNOWLEDGE_BASE"
MANIFEST_RELATIVE = Path(KNOWLEDGE_PREFIX) / "BW-MF-001-SOURCE-MANIFEST.json"
UNSEEN_PREFIX = Path("03_UNSEEN_INCIDENT_TEST_CASES")
GOLDEN_PROMPT = Path("02_GOLD_EVALUATION_PROTECTED/GOLD_80_QUESTIONS/BW-SP-003-sample-prompt.json")


def load_demo_inputs(split: str, *, dataset_root: Path = DATASET_ROOT) -> list[dict[str, Any]]:
    """Return sanitized incident prompts only; evaluator notes and answer files never leave disk."""
    root = dataset_root.resolve()
    if split == "unseen":
        paths = sorted((root / UNSEEN_PREFIX).rglob("*.json"))
    elif split == "golden":
        paths = [root / GOLDEN_PROMPT]
    else:
        raise ValueError("split must be 'unseen' or 'golden'")

    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        relative = path.resolve().relative_to(root).as_posix()
        if split == "unseen" and not relative.startswith(UNSEEN_PREFIX.as_posix() + "/"):
            raise ValueError("Unseen input escaped its protected dataset directory")
        if split == "golden" and relative != GOLDEN_PROMPT.as_posix():
            raise ValueError("Golden demo input path is not the approved sample prompt")
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        narrative = str(payload.get("narrative") or payload.get("input_text") or "").strip()
        # Do not include evaluator_only_note, answer keys, rubrics, or final reports.
        rows.append({
            "case_id": str(payload.get("case_id") or payload.get("case_label") or path.stem),
            "case_name": str(payload.get("case_name") or payload.get("case_label") or path.stem),
            "jurisdiction": str(payload.get("jurisdiction") or payload.get("country") or "").strip() or None,
            "incident_date": str(payload.get("incident_date") or payload.get("event_date") or "").strip() or None,
            "narrative": narrative,
            "dataset_split": split,
            "input_file": relative,
            "validation_status": "ready" if narrative else "skipped_missing_narrative",
        })
    return rows
