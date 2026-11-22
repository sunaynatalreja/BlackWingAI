from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

SOURCE_TYPES = {"official_factual", "official_preliminary", "official_final", "voluntary_asrs",
                "regulation", "operational_order", "advisory", "user_submitted"}
GUARD = re.compile(
    r"(^|[/_\\-])(GOLD(?:EN)?|EVALUATION|HELDOUT|HOLDOUT|UNSEEN|VALIDATION|"
    r"DO_NOT_INDEX|REPORT_DRAFT)([/_\\.-]|$)", re.I
)


@dataclass
class CorpusAudit:
    discovered_sources: int = 0
    approved_unique_files: int = 0
    indexed_source_records: int = 0
    approved_document_count: int = 0
    chunks: int = 0
    source_type_counts: dict[str, int] = field(default_factory=dict)
    exclusions: list[dict[str, str]] = field(default_factory=list)
    asrs_record_count: int = 0
    asrs_iit_counting: str = "NOT VERIFIED: institute approval needed for individual ASRS records"

    @property
    def meets_iit_document_count(self) -> bool:
        return 500 <= self.approved_document_count <= 2000

    def record_exclusion(self, source: str, reason: str) -> None:
        self.exclusions.append({"source": source, "reason": reason})


def _inside_root(root: Path, relative: str) -> Path:
    p = (root / relative).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError("Manifest paths cannot escape the approved source directory")
    return p


def _text_units(path: Path, entry: dict[str, Any]) -> list[tuple[int | None, str, str]]:
    """Returns (page_number, record_id, text); no image/OCR claims for scanned PDFs."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf from requirements-rag.txt to ingest PDFs") from exc
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except Exception as exc:
                raise RuntimeError(f"Cannot open encrypted PDF {path.name} without a password") from exc
            if not unlocked:
                raise RuntimeError(f"PDF {path.name} requires a password; it was not indexed")
        units = []
        fallback_pdf = None
        try:
            try:
                pages = reader.pages
                page_count = len(pages)
            except Exception as exc:
                raise RuntimeError(f"Cannot read PDF page tree in {path.name}: {type(exc).__name__}") from exc
            for page_number in range(1, page_count + 1):
                page = pages[page_number - 1]
                try:
                    page_text = page.extract_text() or ""
                except Exception as primary_error:
                    try:
                        import pymupdf
                        if fallback_pdf is None:
                            fallback_pdf = pymupdf.open(str(path))
                        page_text = fallback_pdf[page_number - 1].get_text("text") or ""
                        print(f"PDF extraction fallback: {path.name} page {page_number} "
                              f"({type(primary_error).__name__})")
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"Cannot extract {path.name}, page {page_number}: "
                            f"pypdf and PyMuPDF both failed"
                        ) from fallback_error
                units.append((page_number, "", page_text))
            return units
        finally:
            if fallback_pdf is not None:
                fallback_pdf.close()
    if suffix in {".txt", ".md"}:
        return [(None, "", path.read_text(encoding="utf-8-sig"))]
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            data = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        else:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        if isinstance(data, list):
            items = []
            for i, row in enumerate(data, 1):
                if not isinstance(row, dict):
                    continue
                rec = str(row.get(entry.get("record_id_column", "record_id")) or i)
                items.append((None, rec, str(row.get(entry.get("text_column", "text")) or row.get("narrative") or "")))
            return items
        if isinstance(data, dict):
            return [(None, "", str(data.get("content") or data.get("text") or ""))]
        return []
    if suffix == ".csv":
        if not entry.get("record_id_column") or not entry.get("text_column"):
            raise ValueError("CSV sources require record_id_column and text_column for provenance")
        with path.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        return [(None, str(row.get(entry["record_id_column"], "")).strip(),
                 str(row.get(entry["text_column"], "")).strip()) for row in rows]
    raise ValueError(f"Unsupported source format {suffix}")


def load_manifest(
    manifest_path: str | Path,
    root: str | Path,
    *,
    allowed_path_prefixes: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], CorpusAudit]:
    """Load source pages/records; never silently approve or count ASRS narratives as IIT documents."""
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    entries = m["sources"] if isinstance(m, dict) and "sources" in m else m
    if not isinstance(entries, list):
        raise ValueError("Manifest needs a 'sources' array")
    root = Path(root).resolve()
    audit = CorpusAudit(discovered_sources=len(entries))
    seen_hashes: set[str] = set()
    source_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for entry in entries:
        source_id = str(entry.get("source_id", "")).strip()
        rel = str(entry.get("path", "")).replace("\\", "/")
        if not source_id or not rel:
            raise ValueError("Every manifest entry needs source_id and path")
        if source_id in source_ids:
            audit.record_exclusion(source_id, "duplicate source ID")
            continue
        source_ids.add(source_id)
        if allowed_path_prefixes and not any(
            rel.startswith(prefix.rstrip("/") + "/") for prefix in allowed_path_prefixes
        ):
            audit.record_exclusion(source_id, "outside the configured knowledge-base source directory")
            continue
        if GUARD.search(rel) or GUARD.search(source_id) or entry.get("evaluation_exclusion"):
            audit.record_exclusion(source_id, "protected evaluation or generated material")
            continue
        if entry.get("approved") is not True:
            audit.record_exclusion(source_id, "not explicitly approved")
            continue
        if entry.get("trial_index_eligible") is False:
            audit.record_exclusion(source_id, "manifest marks source as not eligible for trial indexing")
            continue
        kind = entry.get("source_type")
        if kind not in SOURCE_TYPES:
            audit.record_exclusion(source_id, "unsupported or missing source_type")
            continue
        if kind == "voluntary_asrs" and entry.get("pii_reviewed") is not True:
            audit.record_exclusion(source_id, "ASRS requires explicit PII review/redaction acknowledgement")
            continue
        if kind == "official_final" and entry.get("blind_evaluation_source"):
            audit.record_exclusion(source_id, "official final conclusion excluded from blind evaluation")
            continue
        try:
            for date_key in ("incident_date", "publication_date", "document_date", "effective_date", "effective_through"):
                date_value = entry.get(date_key)
                if date_value:
                    date.fromisoformat(str(date_value))
            if entry.get("effective_date") and entry.get("effective_through") and str(entry["effective_date"]) > str(entry["effective_through"]):
                raise ValueError("effective_date is later than effective_through")
            path = _inside_root(root, rel)
            if not path.is_file():
                raise FileNotFoundError("source file missing")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if sha in seen_hashes:
                audit.record_exclusion(source_id, "duplicate file SHA-256")
                continue
            units = _text_units(path, entry)
            usable = [(page, rid, text.strip()) for page, rid, text in units if len(text.strip()) >= 25]
            if not usable:
                audit.record_exclusion(source_id, "no extractable text; image/OCR review required")
                continue
            if path.suffix.lower() in {".csv", ".jsonl"} and any(not rid for _, rid, _ in usable):
                audit.record_exclusion(source_id, "missing record identifier")
                continue
            seen_hashes.add(sha)
            audit.approved_unique_files += 1
            audit.source_type_counts[kind] = audit.source_type_counts.get(kind, 0) + 1
            # Count each real approved source file once. ASRS row counting needs institute approval.
            audit.approved_document_count += 1
            if kind == "voluntary_asrs":
                audit.asrs_record_count += len({rid for _, rid, _ in usable})
            for page, rid, text in usable:
                document_id = source_id + (":" + rid if rid else "")
                role = ("historical_analogy" if kind == "official_final" else
                        "general_guidance" if kind == "advisory" else
                        "regulatory_reference" if kind in {"regulation", "operational_order"} else
                        "reported" if kind in {"voluntary_asrs", "user_submitted"} else "case_factual")
                rows.append({
                    "document_id": document_id, "source_id": source_id,
                    "record_id": rid or None, "title": entry.get("title", path.stem),
                    "original_filename": path.name, "source_path": rel, "sha256": sha,
                    "source_url": entry.get("source_url"), "document_type": entry.get("document_type") or (
                        "incident_report" if kind.startswith("official_") else
                        "regulation" if kind in {"regulation", "operational_order"} else "guidance"),
                    "source_type": kind, "authority_type": entry.get("authority_type") or entry.get("authority"),
                    "source_status": entry.get("source_status") or kind,
                    "evidence_role": role, "incident_id": entry.get("incident_id") or entry.get("case_id") or "",
                    "case_id": entry.get("incident_id") or entry.get("case_id") or "",
                    "incident_date": entry.get("incident_date"),
                    "document_date": entry.get("publication_date") or entry.get("document_date"),
                    "publication_date": entry.get("publication_date") or entry.get("document_date"),
                    "effective_date": entry.get("effective_date"),
                    "effective_through": entry.get("effective_through"),
                    "jurisdiction": entry.get("jurisdiction"),
                    "edition": entry.get("edition"), "version": entry.get("edition") or entry.get("version"),
                    "page_number": page, "section_reference": entry.get("section_reference"),
                    "provenance": "approved_manifest", "evidence_status": "RETRIEVED_NOT_VERIFIED",
                    "text": text,
                })
        except (ValueError, OSError, RuntimeError) as exc:
            audit.record_exclusion(source_id, f"ingestion issue: {type(exc).__name__}: {str(exc)[:180]}")
    audit.indexed_source_records = len(rows)
    return rows, audit
