from __future__ import annotations

import json
import logging
import os
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable

from .embeddings import make_embeddings
from .engine import CrossEncoderReranker, HybridIndex

logger = logging.getLogger("blackwing.rag")


def load_seed_documents() -> list[dict[str, Any]]:
    seed = files("blackwing.rag").joinpath("seed")
    rows = []
    for resource in seed.iterdir():
        if resource.name.lower().endswith(".json"):
            rows.append(json.loads(resource.read_text(encoding="utf-8")))
    return rows


def _compat_document(item: dict[str, Any]) -> dict[str, Any]:
    """Do not invent incident ID when the team's original seeded source lacks one."""
    row = dict(item)
    row.setdefault("document_id", str(row.get("source_id") or row.get("title") or "unnamed"))
    dtype = str(row.get("document_type") or "").casefold()
    status = str(row.get("source_status") or "").casefold()
    if "source_type" not in row:
        if dtype in {"regulation", "procedure"}:
            row["source_type"] = "regulation" if dtype == "regulation" else "operational_order"
        elif dtype in {"guidance", "advisory"}:
            row["source_type"] = "advisory"
        elif "final" in status:
            row["source_type"] = "official_final"
        elif status in {"user_report", "reported", "unverified_voluntary_report"}:
            row["source_type"] = "user_submitted"
        else:
            # Unknown incoming reports cannot be promoted to official factual evidence.
            row["source_type"] = "unverified_reference"
    kind = row["source_type"]
    row.setdefault("evidence_role", "historical_analogy" if kind == "official_final" else
                   "general_guidance" if kind == "advisory" else
                   "regulatory_reference" if kind in {"regulation", "operational_order"} else
                   "reported" if kind == "user_submitted" else "unverified_reference")
    row.setdefault("incident_id", row.get("case_id") or (row.get("source_id") if dtype == "incident_report" else "") or "")
    row.setdefault("original_filename", row.get("title"))
    row.setdefault("publication_date", row.get("document_date"))
    row.setdefault("evidence_status", "RETRIEVED_NOT_VERIFIED")
    row.setdefault("source_id", row["document_id"])
    return row


class RAGRetriever:
    """Use explicit source filters and local/IIT BGE; BM25 fallback is always labeled."""
    def __init__(self, index_dir: str | Path | None = None, *, strategy: str = "fixed",
                 embedder: Any = None, reranker: Any = None) -> None:
        self.index_dir = Path(index_dir) if index_dir else (
            Path(os.environ["BLACKWING_RAG_INDEX_DIR"]) if os.environ.get("BLACKWING_RAG_INDEX_DIR") else None)
        self.strategy = strategy
        self.embedder = embedder
        self.reranker = reranker
        self._index: HybridIndex | None = None
        self._signature: str | None = None
        self._dense_error: str | None = None
        self._rerank_error: str | None = None

    def _embedding(self, mode: str = "local"):
        if self.embedder is not None:
            return self.embedder
        try:
            # Local is the default to avoid institute quota usage without explicit opt in.
            cache = os.environ.get("BLACKWING_RAG_CACHE", "") or None
            return make_embeddings(mode, cache_path=cache)
        except Exception as exc:
            self._dense_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            logger.warning("RAG dense unavailable (%s); lexical-only fallback", type(exc).__name__)
            return None

    def _reranker(self):
        if self.reranker is not None:
            return self.reranker
        if os.getenv("BLACKWING_RAG_RERANKER", "none").strip().lower() != "crossencoder":
            return None
        try:
            self.reranker = CrossEncoderReranker()
            return self.reranker
        except Exception as exc:
            self._rerank_error = type(exc).__name__
            return None

    def _open_index(self, documents: list[dict[str, Any]], embedding_model: str) -> HybridIndex:
        requested_mode = os.getenv("BLACKWING_RAG_BACKEND", "local").lower()
        # Original FastAPI passes a model name; fail safely for unsupported model.
        if "bge-small-en-v1.5" not in embedding_model.casefold() and self.embedder is None:
            raise ValueError("This IIT pipeline requires BGE-small-en-v1.5; no MiniLM/BGE vector mixing")
        if self.index_dir:
            if self._index is None:
                embedder = self._embedding(requested_mode)
                try:
                    self._index = HybridIndex.load(self.index_dir, embeddings=embedder)
                except ValueError:
                    # A saved IIT/local index must NOT be searched with mismatched vectors.
                    raise
                except Exception as exc:
                    logger.warning("Dense index could not be opened (%s); lexical-only load", type(exc).__name__)
                    self._dense_error = type(exc).__name__
                    self._index = HybridIndex.load(self.index_dir)
            # In-memory case documents are additions, not replacements for the approved corpus.
            self._index.add_documents([_compat_document(row) for row in documents])
            return self._index
        # The in-memory API path is intentionally for small prototype corpora.
        # For 500-2000 real documents use a prebuilt versioned index on disk.
        if not documents and self._index is not None:
            return self._index
        prepared = [_compat_document(item) for item in documents]
        doc_signature = json.dumps([(x["document_id"], x.get("content", ""), x.get("page_number"),
                                     x.get("effective_date")) for x in prepared], sort_keys=True, default=str)
        if self._index is not None and self._signature == doc_signature:
            return self._index
        embedder = self._embedding(requested_mode)
        if embedder is not None:
            try:
                self._index = HybridIndex.from_documents(prepared, strategy=self.strategy, embeddings=embedder)
            except (ImportError, RuntimeError, ValueError) as exc:
                self._dense_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                logger.warning("Dense build unavailable (%s); using lexical-only", type(exc).__name__)
                self._index = HybridIndex.from_documents(prepared, strategy=self.strategy)
        else:
            self._index = HybridIndex.from_documents(prepared, strategy=self.strategy)
        self._signature = doc_signature
        return self._index

    def search(self, query: str, documents: Iterable[dict[str, Any]], *,
               embedding_model: str = "BAAI/bge-small-en-v1.5",
               document_types: list[str] | None = None,
               jurisdiction: str | None = None, incident_date: str | None = None,
               limit: int = 5, method: str = "rerank", scope: str = "all",
               case_id: str | None = None,
               include_retrospective_guidance: bool = False) -> dict[str, Any]:
        """Fully compatible with the existing API/search call; no JFK/Austin special cases."""
        index = self._open_index(list(documents), embedding_model)
        try:
            result = index.search(query, top_k=limit, method=method, scope=scope,
                                  case_id=case_id, jurisdiction=jurisdiction,
                                  incident_date=str(incident_date) if incident_date else None,
                                  document_types=document_types,
                                  include_retrospective_guidance=include_retrospective_guidance,
                                  reranker=self._reranker())
        except (ImportError, RuntimeError, OSError) as exc:
            # CrossEncoder weights absent or a local provider unexpectedly failed.
            self._rerank_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            logger.warning("CrossEncoder unavailable; reverting to hybrid/BM25 (%s)", type(exc).__name__)
            result = index.search(query, top_k=limit, method="hybrid", scope=scope,
                                  case_id=case_id, jurisdiction=jurisdiction,
                                  incident_date=str(incident_date) if incident_date else None,
                                  document_types=document_types,
                                  include_retrospective_guidance=include_retrospective_guidance)
            result["limitations"].append("CrossEncoder failed at runtime; reranking requirement NOT VERIFIED")
        if self._dense_error:
            result["limitations"].append("Dense backend not active: " + self._dense_error)
        if self._rerank_error:
            result["limitations"].append("Reranker not active: " + self._rerank_error)
        return result

    def retrieve_evidence(self, query: str, *, case_id: str, limit: int = 5) -> dict[str, Any]:
        if self._index is None and not self.index_dir:
            raise RuntimeError("Load or build an index before retrieve_evidence")
        return self.search(query, [], case_id=case_id, scope="direct", limit=limit)

    def retrieve_regulations(self, query: str, *, jurisdiction: str,
                             incident_date: str, limit: int = 5) -> dict[str, Any]:
        return self.search(query, [], scope="regulatory", jurisdiction=jurisdiction,
                           incident_date=incident_date, limit=limit)

    def find_similar_incidents(self, query: str, *, case_id: str,
                               incident_date: str | None = None, limit: int = 5) -> dict[str, Any]:
        return self.search(query, [], scope="historical", case_id=case_id,
                           incident_date=incident_date, limit=limit)


retriever = RAGRetriever()
