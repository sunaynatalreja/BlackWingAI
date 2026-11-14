"""Single-process development repository; replace with PostgreSQL/object storage."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from threading import RLock
from typing import Any

from blackwing.schemas import CaseStatus, RunStatus
from blackwing.rag.retriever import load_seed_documents


def now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryRepository:
    """In-memory store for the first local vertical slice."""

    def __init__(self) -> None:
        self._lock = RLock()
        self.cases: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.reviews: dict[str, list[dict[str, Any]]] = {}
        # Seed the local knowledge base with the official NTSB reference case.
        # Ingested user documents remain in-memory for this development slice.
        self.documents: dict[str, dict[str, Any]] = {
            row["document_id"]: {**row, "created_at": now()} for row in load_seed_documents()
        }

    def create_case(self, case_id: str, data: dict[str, Any]) -> dict[str, Any]:
        timestamp = now()
        record = {
            **data,
            "case_id": case_id,
            "status": CaseStatus.DRAFT,
            "created_at": timestamp,
            "updated_at": timestamp,
            "evidence": data.get("evidence", []),
        }
        with self._lock:
            self.cases[case_id] = record
            self.reviews[case_id] = []
        return record

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self.cases.get(case_id)

    def list_cases(self, status: str | None, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            records = list(self.cases.values())
        if status:
            records = [row for row in records if row["status"] == status]
        records.sort(key=lambda row: row["created_at"], reverse=True)
        return records[offset:offset + limit], len(records)

    def update_case(self, case_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            record = self.cases.get(case_id)
            if record is None:
                return None
            record.update(fields)
            record["updated_at"] = now()
            return record

    def add_evidence(self, case_id: str, evidence: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            record = self.cases.get(case_id)
            if record is None:
                return None
            record["evidence"].append(evidence)
            record["updated_at"] = now()
            return evidence

    def create_run(self, run_id: str, case_id: str, workflow: str, model: str | None) -> dict[str, Any]:
        record = {
            "run_id": run_id,
            "case_id": case_id,
            "status": RunStatus.QUEUED,
            "workflow": workflow,
            "model": model,
            "progress": [],
            "trace": [],
            "result": None,
            "error_code": None,
            "error": None,
            "created_at": now(),
            "started_at": None,
            "completed_at": None,
        }
        with self._lock:
            self.runs[run_id] = record
            self.cases[case_id]["status"] = CaseStatus.ANALYSIS_RUNNING
            self.cases[case_id]["updated_at"] = now()
        return record

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self.runs.get(run_id)

    def list_case_runs(self, case_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [run for run in self.runs.values() if run["case_id"] == case_id]

    def patch_run(self, run_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            record = self.runs.get(run_id)
            if record is None:
                return None
            record.update(fields)
            return record

    def add_review(self, case_id: str, review: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            if case_id not in self.cases:
                return None
            self.reviews[case_id].append(review)
            self.cases[case_id]["status"] = CaseStatus.REVIEWED
            self.cases[case_id]["updated_at"] = now()
            return review

    def list_reviews(self, case_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.reviews.get(case_id, []))

    def add_document(self, document_id: str, document: dict[str, Any]) -> dict[str, Any]:
        record = {**document, "document_id": document_id, "created_at": now()}
        with self._lock:
            self.documents[document_id] = record
        return record

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self.documents.get(document_id)

    def list_documents(self, document_type: str | None = None, jurisdiction: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self.documents.values())
        return [
            row for row in records
            if (document_type is None or row["document_type"] == document_type)
            and (jurisdiction is None or row.get("jurisdiction", "").casefold() == jurisdiction.casefold())
        ]

    def search_documents(
        self,
        query: str,
        *,
        document_types: list[str] | None = None,
        jurisdiction: str | None = None,
        incident_date: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        terms = {term.casefold() for term in re.findall(r"[\w-]+", query) if len(term) > 2}
        scored: list[tuple[int, dict[str, Any]]] = []
        for row in self.list_documents(jurisdiction=jurisdiction):
            if document_types and row["document_type"] not in document_types:
                continue
            effective_date = row.get("effective_date")
            if incident_date and effective_date and str(effective_date) > incident_date:
                continue
            searchable = " ".join((row.get("title", ""), row.get("content", ""))).casefold()
            score = sum(searchable.count(term) for term in terms)
            if score:
                scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], item[1]["document_id"]))
        results = []
        for score, row in scored[:limit]:
            results.append({
                "document_id": row["document_id"],
                "title": row["title"],
                "document_type": row["document_type"],
                "authority_type": row.get("authority_type"),
                "source_status": row.get("source_status"),
                "version": row.get("version"),
                "source_id": row.get("source_id"),
                "source_url": row.get("source_url"),
                "jurisdiction": row.get("jurisdiction"),
                "document_date": row.get("document_date"),
                "effective_date": row.get("effective_date"),
                "page_number": row.get("page_number"),
                "section_reference": row.get("section_reference"),
                "excerpt": row["content"][:1500],
                "ranking_score": score,
            })
        return results


repository = MemoryRepository()
