"""Local SQLite persistence for evidence, investigation turns and finding review."""
import datetime as dt
import hashlib
import json
import re
import sqlite3
import unicodedata
import uuid
from contextlib import closing, contextmanager
from functools import lru_cache
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parent
DATABASE = ROOT / "output" / "copilot.sqlite3"
GOA = "GOA-VT-BXL"
MAX_NARRATIVE_CHARS = 12_000
PROTECTED_CASE_ALIASES = frozenset({"GOAVTBXL", "INDAAIB001", "INDAAIB001VTBXL"})


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _encode(data):
    return json.dumps(data, ensure_ascii=False)


@lru_cache(maxsize=1)
def library():
    from blackwing_app import INDEX, load_index

    return load_index() if INDEX.is_file() else []


def is_protected_source(record):
    from blackwing_app import normalized_case_id

    identifiers = {
        re.sub(r"[^A-Z0-9]", "", normalized_case_id(record.get(key)))
        for key in ("case_id", "document_id")
    }
    source_labels = " ".join(str(record.get(key) or "") for key in ("filename", "source_role", "source_status"))
    return bool(record.get("held_out") or identifiers & PROTECTED_CASE_ALIASES
                or re.search(r"(?:^|[^a-z])(?:final|gold|answer[ _-]?key)(?:[^a-z]|$)", source_labels, re.I))


@contextmanager
def _database():
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DATABASE, timeout=15)) as connection:
        with connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS cases (case_id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS evidence (
                    case_id TEXT NOT NULL, document_id TEXT NOT NULL, metadata TEXT NOT NULL,
                    chunks TEXT NOT NULL, PRIMARY KEY(case_id, document_id)
                );
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS findings (
                    finding_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, data TEXT NOT NULL
                );
            """)
            yield connection


def _presets():
    for file in ("JFK_DCA23LA125.json", "AUSTIN_DCA23FA149.json", "MUMBAI_SYNTHETIC.json"):
        item = json.loads((ROOT / "examples" / file).read_text(encoding="utf-8-sig"))
        if file.endswith("_SYNTHETIC.json"):
            item["synthetic"] = True
        yield item
    yield {"case_id": GOA, "airport": "GOI", "jurisdiction": "INDIA", "event_date": "2026-04-03",
           "narrative": "", "held_out": True, "new_case": True}


def list_cases():
    from blackwing_app import normalized_case_id

    with _database() as db:
        saved = db.execute("SELECT data FROM cases ORDER BY rowid").fetchall()
    presets = list(_presets())
    cases = [*presets[:2], presets[3], presets[2], *(json.loads(row[0]) for row in saved)]
    known_ids = {normalized_case_id(item["case_id"]) for item in cases}
    for record in library():
        case_id = record.get("case_id")
        normalized_id = normalized_case_id(case_id)
        if (not normalized_id or normalized_id in known_ids or record.get("source_role") != "evidence"
                or record.get("source_status") == "UNVERIFIED_VOLUNTARY_REPORT"
                or is_protected_source(record) or record.get("synthetic")):
            continue
        known_ids.add(normalized_id)
        cases.append({
            "case_id": case_id, "airport": record.get("airport") or "Unknown",
            "jurisdiction": record.get("jurisdiction") or "Unknown",
            "event_date": record.get("event_date"), "narrative": "",
        })
    return cases


def list_historical_cases():
    historical = []
    for incident in list_cases():
        if incident.get("held_out") or incident.get("synthetic"):
            continue
        try:
            event_date = dt.date.fromisoformat(str(incident.get("event_date")))
        except ValueError:
            continue
        if event_date <= dt.date.today():
            historical.append((event_date, incident))
    return [incident for _, incident in sorted(historical, key=lambda row: row[0], reverse=True)]


def create_case(airport, event_date, jurisdiction, narrative, *, aircraft_details="", files=(),
                provenance=""):
    from review_workflow import redact

    incident = {
        "case_id": "NEW-" + uuid.uuid4().hex[:12].upper(),
        "airport": redact(str(airport or "Unknown").strip())[:60],
        "event_date": dt.date.fromisoformat(str(event_date)).isoformat() if event_date else None,
        "jurisdiction": redact(str(jurisdiction or "Unknown").strip())[:60],
        "narrative": redact(str(narrative or "").strip()),
        "aircraft_details": redact(str(aircraft_details or "").strip()),
        "new_case": True,
    }
    if not incident["narrative"] or len(incident["narrative"]) > MAX_NARRATIVE_CHARS:
        raise ValueError(f"Enter a narrative of 1 to {MAX_NARRATIVE_CHARS:,} characters.")
    if len(incident["aircraft_details"]) > 1000:
        raise ValueError("Use up to 1,000 characters for aircraft details.")
    documents = _prepare_documents(incident, files, provenance=provenance) if files else []
    with _database() as db:
        db.execute("INSERT INTO cases VALUES (?, ?)", (incident["case_id"], _encode(incident)))
        _store_documents(db, incident["case_id"], documents)
    return incident


def get_case(case_id):
    incident = next((item for item in list_cases() if item["case_id"] == case_id), None)
    if incident is None:
        raise ValueError("Select an existing case.")
    with _database() as db:
        state = {"incident": incident}
        for table, column in (("evidence", "metadata"), ("findings", "data"), ("turns", "data")):
            rows = db.execute(f"SELECT {column} FROM {table} WHERE case_id = ? ORDER BY rowid",
                              (case_id,)).fetchall()
            state[table] = [json.loads(row[0]) for row in rows]
    return state


def case_chunks(case_id):
    with _database() as db:
        rows = db.execute("SELECT chunks FROM evidence WHERE case_id = ? ORDER BY rowid", (case_id,)).fetchall()
    return [chunk for row in rows for chunk in json.loads(row[0])]


def add_case_evidence(case_id, files, *, provenance, reviewer="", approved=False,
                      source_role="evidence"):
    incident = get_case(case_id)["incident"]
    documents = _prepare_documents(incident, files, provenance=provenance, reviewer=reviewer,
                                   approved=approved, source_role=source_role)
    with _database() as db:
        db.execute("BEGIN IMMEDIATE")
        _store_documents(db, case_id, documents)
    return [metadata for metadata, _ in documents]


def _prepare_documents(incident, files, *, provenance, reviewer="", approved=False,
                       source_role="evidence"):
    from pypdf.errors import PdfReadError
    from review_workflow import redact, uploaded_chunks

    if len(files) > 3:
        raise ValueError("Upload up to three supporting files at once.")
    provenance = redact(str(provenance or "").strip())
    reviewer = redact(str(reviewer or "").strip())
    if not provenance or len(provenance) > 500:
        raise ValueError("Provide source provenance in 1 to 500 characters.")
    if len(reviewer) > 100:
        raise ValueError("Use up to 100 characters for the reviewer name.")
    if (incident.get("held_out") or approved or source_role == "rule_candidate") and (not approved or not reviewer):
        raise ValueError("Record the independent reviewer's name and source approval.")
    if source_role not in {"evidence", "rule_candidate"}:
        raise ValueError("Choose factual evidence or candidate guidance.")
    if not files:
        raise ValueError("Choose at least one evidence file.")

    documents = []
    for name, data in files:
        filename = unicodedata.normalize("NFKC", PureWindowsPath(name).name)
        searchable_name = re.sub(r"[^a-z0-9]+", " ", filename.casefold())
        if re.search(r"\b(?:final(?: report| investigation| conclusions)?|gold(?: answers?)?"
                     r"|answer ?key|conclusions?|evaluation workbook)\b", searchable_name):
            raise ValueError("Use a factual source or guidance file cleared for investigation inputs.")
        digest = hashlib.sha256(data).hexdigest()
        try:
            records = uploaded_chunks([(filename, data)], incident["case_id"])
        except PdfReadError as exc:
            raise ValueError("The PDF could not be read. Upload a valid text PDF.") from exc
        except UnicodeDecodeError as exc:
            raise ValueError("Save text evidence in UTF-8 before uploading it.") from exc
        extracted = " ".join(row["text"] for row in records)
        protected = (r"\b(?:aviation|aircraft|accident|incident|independent|formal) "
                     r"(?:accident |incident )?investigation final report\b"
                     r"|\bfinal (?:aircraft |aviation |accident |incident )?investigation report\b"
                     r"|\bfinal report (?:on|of) (?:the |a )?(?:aircraft|aviation|accident|serious incident)\b"
                     r"|\b(?:gold standard answers?|gold answer key|approved gold answer"
                     r"|answer key for the (?:evaluation|test))\b")
        if re.search(protected, extracted, re.I):
            raise ValueError("The file contains a final report or evaluation answer-key marker.")
        if filename.lower().endswith(".txt"):
            text = redact(data.decode("utf-8-sig"))
            marker = r"(?m)^\[([^|\]\r\n]+) \| physical PDF (pp?\. [^|\]\r\n]+) \| ([^\]\r\n]+)\]"
            matches = list(re.finditer(marker, text))
            if matches:
                template = records[0]
                records = []
                for index, match in enumerate(matches):
                    end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                    locator = f"{match[1].strip()} {match[2].strip()}"
                    words = text[match.end():end].split()[:4000]
                    for offset in range(0, len(words), 140):
                        records.append({
                            **template, "chunk_id": f"{template['document_id']}-section{index + 1}-c{offset}",
                            "page": None, "locator": locator,
                            "text": " ".join(words[offset:offset+170]),
                        })
                if not records:
                    raise ValueError("The page markers contain no evidence text.")
        if incident.get("held_out") and source_role == "evidence":
            suspect = r"probable cause|contributing factors?|applicable standard|occurrence classification|safety recommendations"
            if re.search(suspect, extracted, re.I) or any(re.search(suspect, row["text"], re.I) for row in records):
                raise ValueError("A held-out file contains conclusion or classification language. Use a reviewed factual packet.")
        role = source_role if approved else "submitted"
        status = {"evidence": "REVIEWER_ATTESTED_FACTUAL_INPUT",
                  "rule_candidate": "CANDIDATE_RULE_REQUIRES_EDITION_REVIEW",
                  "submitted": "UNVERIFIED_UPLOAD"}[role]
        metadata = {
            "document_id": records[0]["document_id"], "filename": records[0]["filename"],
            "provenance": provenance, "sha256": digest, "source_role": role, "source_status": status,
            "reviewer": reviewer, "jurisdiction": incident["jurisdiction"], "uploaded_at": now(),
        }
        for row in records:
            row.update(source_role=role, source_status=status, jurisdiction=incident["jurisdiction"],
                       provenance=provenance, reviewer=reviewer)
        documents.append((metadata, records))
    return documents


def _store_documents(db, case_id, documents):
    for metadata, records in documents:
        existing = db.execute("SELECT metadata FROM evidence WHERE case_id=? AND document_id=?",
                              (case_id, metadata["document_id"])).fetchone()
        if existing:
            previous = json.loads(existing[0])
            if any(previous[key] != metadata[key] for key in ("sha256", "source_role", "provenance", "reviewer")):
                raise ValueError("This file is already registered with different content or provenance.")
            metadata.update(previous)
            continue
        db.execute("INSERT INTO evidence VALUES (?, ?, ?, ?)",
                   (case_id, metadata["document_id"], _encode(metadata), _encode(records)))


def save_turn(case_id, turn):
    with _database() as db:
        db.execute("INSERT INTO turns VALUES (?, ?, ?)", (turn["turn_id"], case_id, _encode(turn)))
        db.executemany("INSERT INTO findings VALUES (?, ?, ?)",
                       [(item["finding_id"], case_id, _encode(item)) for item in turn["findings"]])
    return turn


def review_finding(case_id, finding_id, decision, reviewer, note=""):
    from review_workflow import redact

    reviewer = redact(str(reviewer or "").strip())
    if decision not in {"accepted", "rejected"} or not reviewer or len(reviewer) > 100:
        raise ValueError("Choose accept or reject and enter the reviewer name (up to 100 characters).")
    with _database() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT data FROM findings WHERE case_id=? AND finding_id=?",
                         (case_id, finding_id)).fetchone()
        if row is None:
            raise ValueError("This finding is unavailable in the selected case.")
        item = json.loads(row[0])
        if item["status"] != "pending":
            raise ValueError("This finding has already been reviewed.")
        if decision == "accepted" and not item["verification"]["eligible"]:
            raise ValueError("The evidence checks flagged this finding. Reject it and ask a more focused question.")
        item.update(status=decision, reviewed_by=reviewer, reviewed_at=now(),
                    review_note=redact(str(note or "").strip())[:500])
        db.execute("UPDATE findings SET data=? WHERE case_id=? AND finding_id=?",
                   (_encode(item), case_id, finding_id))
    return item
