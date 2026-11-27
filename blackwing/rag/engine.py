from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .tokenization import ApproximateTokenizer, chunk_document, clip_query

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.I)

_NTSB_INCIDENT_YEAR = re.compile(r"\b[A-Z]{3}(\d{2})[A-Z]{2}\d{3,}\b", re.I)
_LABELED_ISO_DATE = re.compile(
    r"\b(?:incident\s+date|occurrence\s+date|date\s*&\s*time)\s*:\s*"
    r"(19\d{2}|20\d{2})-(\d{2})-(\d{2})\b", re.I,
)
_LABELED_MONTH_DATE = re.compile(
    r"\b(?:incident\s+date|occurrence\s+date|date\s*&\s*time)\s*:\s*"
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),?\s+(19\d{2}|20\d{2})\b", re.I,
)
_MONTHS = {name.casefold(): index for index, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"), 1)}


def infer_occurrence_date(row: Mapping[str, Any]) -> str | None:
    """Recover an occurrence date conservatively when source metadata omitted it.

    Prefer explicit metadata, then a date stated in the source text, then the
    two-digit year in a standard NTSB incident identifier. The identifier gives
    only a year, so callers must not infer an ordering within that year.
    """
    value = str(row.get("incident_date") or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    text = " ".join(str(row.get(key) or "") for key in
                    ("title", "excerpt", "content", "text"))
    match = _LABELED_ISO_DATE.search(text)
    if match:
        return "-".join(match.groups())
    match = _LABELED_MONTH_DATE.search(text)
    if match:
        month, day, year = match.groups()
        return f"{year}-{_MONTHS[month.casefold()]:02d}-{int(day):02d}"
    incident_id = " ".join(str(row.get(key) or "") for key in
                            ("incident_id", "case_id", "title"))
    match = _NTSB_INCIDENT_YEAR.search(incident_id)
    if match:
        year = int(match.group(1))
        year += 2000 if year <= 29 else 1900
        return f"{year:04d}"
    return None


def _terms(text: str) -> list[str]:
    return [item.casefold() for item in TOKEN_RE.findall(text) if len(item) >= 2]


def _signature(rows: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str,
                  separators=(",", ":")).encode("utf-8")).hexdigest()


def lexical_rank(query: str, records: list[dict[str, Any]]) -> list[tuple[float, int]]:
    if not records:
        return []
    terms = set(_terms(query))
    if not terms:
        return []
    tok = [_terms(str(row.get("excerpt", "")) + " " + str(row.get("title", ""))) for row in records]
    sizes = [len(t) for t in tok]
    avg = sum(sizes) / len(sizes) or 1.0
    df = Counter(term for row in tok for term in set(row) if term in terms)
    out = []
    for i, (t, size) in enumerate(zip(tok, sizes)):
        tf = Counter(t)
        score = 0.0
        for term in terms:
            if tf[term]:
                idf = math.log(1 + (len(records) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf[term] * 2.2 / (tf[term] + 1.2 * (0.25 + 0.75 * size / avg))
        if score:
            out.append((score, i))
    return sorted(out, key=lambda v: (-v[0], records[v[1]]["chunk_id"]))


class CrossEncoderReranker:
    """Opt-in real CrossEncoder; no silent substitution with keyword boosting."""
    identity = "cross-encoder/ms-marco-MiniLM-L6-v2"

    def __init__(self, model_name: str = identity) -> None:
        self.identity = model_name
        self._model = None

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.identity)
        return [float(v) for v in self._model.predict([(query, passage) for passage in passages],
                                                      show_progress_bar=False)]
def _is_protected_gold(row):
    import re

    identifiers = " ".join(
        str(row.get(key) or "")
        for key in ("source_id", "document_id", "source_path")
    )

    return bool(
        re.search(
            r"(?<![A-Za-z0-9])BW-HO-00[1-9](?![0-9])",
            identifiers,
            flags=re.IGNORECASE,
        )
    )

def _scope_filter(row: dict[str, Any], *, scope: str, case_id: str | None,
                  jurisdiction: str | None, incident_date: str | None,
                  include_retrospective_guidance: bool) -> bool:
    kind = str(row.get("source_type") or "").casefold()
    dtype = str(row.get("document_type") or "").casefold()
    case = str(row.get("incident_id") or row.get("case_id") or "").casefold()
    target = (case_id or "").casefold()
    region = (jurisdiction or "").casefold()
    doc_region = str(row.get("jurisdiction") or "").casefold()
    if scope not in {"all", "direct", "historical", "regulatory", "guidance"}:
        raise ValueError("scope must be all, direct, historical, regulatory or guidance")
    if scope == "direct":
        if not target or case != target or kind not in {"official_factual", "official_preliminary", "user_submitted"}:
            return False
    if scope == "historical":
        if kind not in {"official_factual", "official_preliminary", "official_final", "voluntary_asrs"}:
            return False
        if not case or (target and case == target):
            return False
    if scope == "regulatory":
        if kind not in {"regulation", "operational_order"} and dtype not in {"regulation", "procedure"}:
            return False
        # Binding-applicability retrieval is not meaningful without both filters.
        if not region or not incident_date or not doc_region or doc_region != region:
            return False
        start = row.get("effective_date") or row.get("publication_date") or row.get("document_date")
        if not start or str(start) > incident_date:
            return False
        if row.get("effective_through") and str(row["effective_through"]) < incident_date:
            return False
    if scope == "guidance":
        if kind != "advisory" and dtype != "guidance":
            return False
        # Cross-jurisdiction guidance is general context, never a binding local requirement.
        if incident_date and not include_retrospective_guidance:
            published = row.get("publication_date") or row.get("document_date")
            if published and str(published) > incident_date:
                return False
    if scope in {"all", "direct", "historical"}:
        if jurisdiction and doc_region and doc_region != region:
            return False
        if incident_date and row.get("effective_date") and str(row["effective_date"]) > incident_date:
            return False
        if incident_date and row.get("effective_through") and str(row["effective_through"]) < incident_date:
            return False
    if scope == "historical" and incident_date:
        source_date = infer_occurrence_date(row)
        if source_date:
            # NTSB identifiers only establish a year. Treat same-year cases as
            # non-preceding when the actual date cannot be established.
            if len(source_date) == 4:
                if source_date >= incident_date[:4]:
                    return False
            elif source_date >= incident_date:
                return False

        elif kind == "voluntary_asrs":
            month = str(row.get("incident_month") or "")

            if (len(month) != 6 or not month.isdigit()
                    or not 1 <= int(month[4:6]) <= 12):
                return False

            # Month-level evidence must precede the entire target month.
            # Never invent an exact incident day.
            target_month = incident_date[:4] + incident_date[5:7]

            if month >= target_month:
                return False

        else:
            return False
    return True


class HybridIndex:
    def __init__(self, chunks: Sequence[dict[str, Any]], *, embeddings: Any = None,
                 vectors: Any = None, strategy: str = "fixed", audit: dict | None = None) -> None:
        self.chunks = list(chunks)
        self.embeddings = embeddings
        self.strategy = strategy
        self.audit = audit or {}
        ids = [r["chunk_id"] for r in self.chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate chunk IDs")
        self.vectors = None
        if vectors is not None:
            arr = np.asarray(vectors, dtype=np.float32)
            if arr.shape != (len(self.chunks), 384):
                raise ValueError("Vector index shape mismatch")
            self.vectors = arr
        self.signature = _signature(self.chunks)

    @classmethod
    def from_documents(cls, documents: Sequence[dict[str, Any]], *, strategy: str = "fixed",
                       tokenizer: Any = None, embeddings: Any = None,
                       audit: dict | None = None) -> "HybridIndex":
        tokenizer = tokenizer or (embeddings.tokenizer if embeddings is not None else ApproximateTokenizer())
        chunks = [item for document in documents for item in chunk_document(document, strategy, tokenizer)]
        vectors = embeddings.encode([row["excerpt"] for row in chunks]) if embeddings and chunks else None
        return cls(chunks, embeddings=embeddings, vectors=vectors, strategy=strategy, audit=audit)

    def add_documents(self, documents: Sequence[dict[str, Any]]) -> None:
        """Add new approved/incident text without rewriting source documents or existing vectors."""
        existing = {row["document_id"] for row in self.chunks}
        fresh = [row for row in documents if row["document_id"] not in existing]
        if not fresh:
            return
        tokenizer = self.embeddings.tokenizer if self.embeddings else ApproximateTokenizer()
        new_chunks = [item for doc in fresh for item in chunk_document(doc, self.strategy, tokenizer)]
        if not new_chunks:
            return
        if self.embeddings and self.vectors is not None:
            additional = self.embeddings.encode([row["excerpt"] for row in new_chunks])
            self.vectors = np.vstack((self.vectors, additional))
        elif self.vectors is not None:
            raise ValueError("Cannot add passages without matching embedding provider")
        self.chunks.extend(new_chunks)
        self.signature = _signature(self.chunks)

    def save(self, directory: str | Path) -> dict[str, Any]:
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError("Refusing to overwrite prior RAG index; choose a new versioned directory")
        directory.mkdir(parents=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for row in self.chunks:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        model_id = self.embeddings.identity if self.vectors is not None and self.embeddings else None
        if self.vectors is not None:
            np.save(directory / "vectors.npy", self.vectors, allow_pickle=False)
        source_ids = {row.get("source_id") or row["document_id"] for row in self.chunks}
        manifest = {
            "schema_version": 2, "signature": self.signature, "strategy": self.strategy,
            "document_count_indexed": len(source_ids), "chunk_count": len(self.chunks),
            "embedding_model": model_id,
            "vector_sha256": hashlib.sha256(self.vectors.astype(np.float32).tobytes()).hexdigest() if self.vectors is not None else None,
            "tokenizer_verified": all(r.get("tokenizer_verified", False) for r in self.chunks),
            "crossencoder_status": "NOT VERIFIED: evaluate separately",
            "audit": self.audit,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return manifest

    @classmethod
    def load(cls, directory: str | Path, *, embeddings: Any = None) -> "HybridIndex":
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        # JSON strings may contain Unicode line separators (U+2028/U+0085).
        # str.splitlines() treats those as record boundaries even when escaped
        # JSON is otherwise valid; JSONL records are delimited by literal LF.
        chunks = [json.loads(line) for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
        if _signature(chunks) != manifest["signature"]:
            raise ValueError("Index content does not match manifest: refuse to search")
        vectors = None
        if (directory / "vectors.npy").is_file():
            if embeddings is not None:
                if manifest.get("embedding_model") != getattr(embeddings, "identity", None):
                    raise ValueError("Query and document embedding model identities differ; rebuild or choose matching backend")
                vectors = np.load(directory / "vectors.npy", allow_pickle=False)
                if manifest.get("vector_sha256") != hashlib.sha256(np.asarray(vectors, dtype=np.float32).tobytes()).hexdigest():
                    raise ValueError("Saved embedding vectors do not match the index manifest")
            # Without matching backend, degrade EXPLICITLY to lexical retrieval.
        return cls(chunks, embeddings=embeddings, vectors=vectors,
                   strategy=manifest["strategy"], audit=manifest.get("audit", {}))

    def search(self, query: str, *, top_k: int = 5, method: str = "hybrid",
               scope: str = "all", case_id: str | None = None,
               jurisdiction: str | None = None, incident_date: str | None = None,
               document_types: Sequence[str] | None = None,
               include_retrospective_guidance: bool = False,
               reranker: Any = None, rerank_candidates: int = 30) -> dict[str, Any]:
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if method not in {"bm25", "dense", "hybrid", "rerank"}:
            raise ValueError("method must be bm25, dense, hybrid or rerank")
        if scope == "regulatory" and (not jurisdiction or not incident_date):
            raise ValueError("Regulatory retrieval requires jurisdiction and incident_date")
        pool = [
    row for row in self.chunks
    if not _is_protected_gold(row)
    and (not document_types or row.get("document_type") in document_types)
    and _scope_filter(
        row,
        scope=scope,
        case_id=case_id,
        jurisdiction=jurisdiction,
        incident_date=incident_date,
        include_retrospective_guidance=include_retrospective_guidance,
    )
]
        if not query.strip() or not pool:
            return {"results": [], "retrieval_method": "no_matches", "limitations": ["No eligible evidence in selected scope"],
                    "metrics": {"eligible_chunks": len(pool), "reranker_status": "not_run"}}
        used_query, shortened = clip_query(query, self.embeddings.tokenizer if self.embeddings else ApproximateTokenizer())
        lexical = lexical_rank(used_query, pool)
        bm25 = dict((i, float(s)) for s, i in lexical)
        dense = []
        if method != "bm25" and self.embeddings is not None and self.vectors is not None:
            position = {r["chunk_id"]: i for i, r in enumerate(self.chunks)}
            idx = [position[row["chunk_id"]] for row in pool]
            vec = self.embeddings.encode([used_query], query=True)[0]
            scores = self.vectors[idx] @ vec
            dense = sorted([(float(score), i) for i, score in enumerate(scores)],
                           key=lambda v: (-v[0], pool[v[1]]["chunk_id"]))
        if method == "dense" and not dense:
            raise ValueError("Dense search requested but matching query/document BGE embeddings are unavailable")
        scores_dense = {i: score for score, i in dense}
        fused = {}
        if method == "bm25" or (not dense and method in {"hybrid", "rerank"}):
            order = [i for _, i in lexical]
            for rank, i in enumerate(order, 1):
                fused[i] = 1.0 / (60 + rank)
        elif method == "dense":
            order = [i for _, i in dense]
            for rank, i in enumerate(order, 1):
                fused[i] = 1.0 / (60 + rank)
        else:
            for rankings in (lexical[:100], dense[:100]):
                for rank, (_, i) in enumerate(rankings, 1):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
            order = sorted(fused, key=lambda i: (-fused[i], pool[i]["chunk_id"]))
        reranks: dict[int, float] = {}
        status = "not_requested"
        if method == "rerank":
            if reranker is None:
                status = "UNAVAILABLE: CrossEncoder not supplied; using hybrid candidate order"
            elif not order:
                status = "empty"
            else:
                candidates = order[:max(top_k, rerank_candidates)]
                values = reranker.score(used_query, [pool[i]["excerpt"] for i in candidates])
                if len(values) != len(candidates):
                    raise ValueError("Reranker score count mismatch")
                reranks = dict(zip(candidates, map(float, values)))
                order = sorted(candidates, key=lambda i: (-reranks[i], pool[i]["chunk_id"]))
                status = "crossencoder"
        results = []
        for i in order[:top_k]:
            row = pool[i]
            locator = f"p.{row['page_number']}" if row.get("page_number") else (
                f"record {row['record_id']}" if row.get("record_id") else row.get("section_reference") or "unpaginated")
            caution = ("Historical analogy: not direct evidence about this incident" if row.get("evidence_role") == "historical_analogy" or row.get("source_type") == "official_final" else
                       "General guidance: not binding local regulation" if row.get("evidence_role") == "general_guidance" or row.get("source_type") == "advisory" else
                       "Retrieved candidate: verify source and factual support")
            result = {**row,
                "ranking_score": round(fused.get(i, 0.0), 8),
                "score_bm25": bm25.get(i), "score_dense": scores_dense.get(i),
                "score_rrf": fused.get(i), "score_rerank": reranks.get(i),
                "retrieval_method": method if status != "UNAVAILABLE" else "hybrid",
                "citation": {"chunk_id": row["chunk_id"], "source_id": row.get("source_id"),
                             "page_number": row.get("page_number"), "record_id": row.get("record_id")},
                "citation_label": f"{row.get('source_id') or row['document_id']} | {locator}",
                "caution": caution,
            }
            results.append(result)
        limitations = ["Retrieved passages are candidates, not independently verified incident findings.",
                       "Historical findings and nonbinding safety guidance must not be treated as direct-case facts or binding local regulations."]
        if shortened:
            limitations.append("Long new-incident narrative truncated for BGE query token budget; full narrative remains in case data.")
        if method != "bm25" and not dense:
            limitations.append("Dense BGE was unavailable; results are BM25 only, NOT hybrid RAG.")
        if method == "rerank" and status != "crossencoder":
            limitations.append("CrossEncoder reranking NOT ACTIVE; required IIT reranker comparison remains unverified.")
        return {"results": results, "retrieval_method": (
                    "bm25" if not dense else ("hybrid+crossencoder" if status == "crossencoder" else "hybrid" if method == "rerank" else method)),
                "limitations": limitations,
                "metrics": {"eligible_chunks": len(pool), "query_truncated": shortened, "reranker_status": status,
                            "bm25_candidate_count": len(lexical), "dense_candidate_count": len(dense)}}
