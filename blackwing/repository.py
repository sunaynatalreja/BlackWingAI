"""Local repository interface with memory and durable SQLite implementations."""

from __future__ import annotations

from datetime import datetime, timezone
import re
import json
import os
import sqlite3
from pathlib import Path
from functools import wraps
from threading import RLock
from typing import Any

from blackwing.schemas import CaseStatus, RunStatus
from blackwing.diagnostics import lifecycle_event


def now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryRepository:
    """In-memory store for the first local vertical slice."""

    def __init__(self) -> None:
        self._lock = RLock()
        self.cases: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.reviews: dict[str, list[dict[str, Any]]] = {}
        # The approved dataset index is configured separately. Do not seed a
        # protected final report into in-memory retrieval documents.
        self.documents: dict[str, dict[str, Any]] = {}
        self.qa_answers: dict[str, dict[str, Any]] = {}
        self.qa_conversations: dict[str, list[dict[str, Any]]] = {}

    def list_conversation_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.qa_conversations.get(conversation_id, []))

    def append_conversation_message(self, conversation_id: str, message: dict[str, Any]) -> None:
        with self._lock:
            messages = self.qa_conversations.setdefault(conversation_id, [])
            messages.append(message)
            del messages[:-50]

    def save_qa_answer(self, answer_id: str, record: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.qa_answers[answer_id] = record
            return record

    def get_qa_answer(self, answer_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self.qa_answers.get(answer_id)

    def list_qa_answers(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(self.qa_answers.values(), key=lambda r: r["created_at"], reverse=True)

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


    def append_event(self, run_id, event):
        with self._lock:
            record = self.runs[run_id]
            record.setdefault("audit_events", []).append(event)
            if event.get("event_id"):
                summary = {key: event[key] for key in ("step", "status", "agent", "agent_key", "execution_id", "timestamp", "duration_ms", "error") if key in event}
                record.setdefault("progress", []).append(summary)
                record["execution_state"] = summary

def _encode(value):
    if isinstance(value, datetime):
        return {"__blackwing_datetime__": value.isoformat()}
    raise TypeError(type(value).__name__)


def _decode(value):
    if set(value) == {"__blackwing_datetime__"}:
        return datetime.fromisoformat(value["__blackwing_datetime__"])
    return value


class SQLiteRepository(MemoryRepository):
    """Single API worker, atomic SQLite transactions and write-through mutations.

    Keep the original interface and detached-on-restart JSON records. One owner
    process must use a database; a second application worker is unsupported.
    """
    def __init__(self, path=None):
        super().__init__()
        path = Path(path or os.getenv("BLACKWING_DATABASE_PATH") or Path(__file__).resolve().parent.parent / "work" / "blackwing.sqlite3")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, payload TEXT NOT NULL, PRIMARY KEY(kind,id))")
        self._db.execute("CREATE TABLE IF NOT EXISTS audit_events (run_id TEXT, sequence INTEGER, payload TEXT NOT NULL, PRIMARY KEY(run_id,sequence))")
        for kind in ("cases", "runs", "reviews", "documents", "qa_answers", "qa_conversations"):
            rows = self._db.execute("SELECT id,payload FROM records WHERE kind=?", (kind,)).fetchall()
            if rows:
                setattr(self, kind, {key: json.loads(payload, object_hook=_decode) for key, payload in rows})
        for run in self.runs.values():
            if run["status"] in {"queued", "running"}:
                run.update(status="failed", error_code="interrupted",
                           error="Application restarted before completion. Resume is available when a saved LangGraph checkpoint exists.",
                           completed_at=now(), resumable=bool(run.get("resume_payload")))
                self.cases[run["case_id"]].update(status="analysis_failed", updated_at=now())
                previous = run.get("execution_state") or {}
                active = {}
                for event in run.get("audit_events", []):
                    if event.get("execution_id"):
                        active[event["execution_id"]] = event
                unfinished = [e for e in active.values() if e["status"] not in {"agent_completed", "agent_failed", "execution_interrupted"}]
                last_agent = unfinished[-1] if unfinished else {}
                self.append_event(run["run_id"], lifecycle_event(
                    "agent_execution" if last_agent else "workflow_execution", "execution_interrupted", case_id=run["case_id"],
                    **{key: last_agent[key] for key in ("execution_id", "agent", "agent_key", "provider", "model") if key in last_agent},
                    error={"code": "interrupted", "type": "ApplicationRestart", "message": "Application restarted before this run completed; no checkpoint resumption was attempted."},
                    detail={"last_known_status": previous.get("status"), "agent_started_observed": bool(last_agent), "resumable": False}))
        self._flush()

    def _flush(self):
        with self._db:
            for kind in ("cases", "runs", "reviews", "documents", "qa_answers", "qa_conversations"):
                self._db.executemany("INSERT OR REPLACE INTO records VALUES (?,?,?)", [
                    (kind, key, json.dumps(value, default=_encode))
                    for key, value in getattr(self, kind).items()])

    def append_event(self, run_id, event):
        with self._lock:
            super().append_event(run_id, event)
            sequence = len(self.runs[run_id]["audit_events"])
            with self._db:
                self._db.execute("INSERT INTO audit_events VALUES (?,?,?)",
                                 (run_id, sequence, json.dumps(event, default=_encode)))
                self._flush()

    def close(self):
        with self._lock:
            self._db.close()


def _durable_mutation(name):
    @wraps(getattr(MemoryRepository, name))
    def mutate(self, *args, **kwargs):
        with self._lock:
            result = getattr(super(SQLiteRepository, self), name)(*args, **kwargs)
            self._flush()
            return result
    return mutate


for _method in ("create_case", "update_case", "add_evidence", "create_run", "patch_run", "add_review", "add_document", "save_qa_answer", "append_conversation_message"):
    setattr(SQLiteRepository, _method, _durable_mutation(_method))


class _DefaultRepository:
    """Open durable storage on first use, avoiding import-time filesystem writes."""
    def __init__(self):
        self._instance = None
        self._lock = RLock()

    def __getattr__(self, name):
        with self._lock:
            if self._instance is None:
                self._instance = SQLiteRepository()
            return getattr(self._instance, name)


repository = _DefaultRepository()
