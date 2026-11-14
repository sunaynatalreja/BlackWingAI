"""Chunked hybrid retrieval with local BGE embeddings and lexical fallback."""

from __future__ import annotations

from collections import Counter
from importlib.resources import files
import json
import logging
import math
import re
from typing import Any, Iterable

logger = logging.getLogger("blackwing.rag")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.I)


def load_seed_documents() -> list[dict[str, Any]]:
    seed_dir = files("blackwing.rag").joinpath("seed")
    documents = []
    for resource in seed_dir.iterdir():
        if resource.name.endswith(".json"):
            documents.append(json.loads(resource.read_text(encoding="utf-8")))
    return documents


def _tokens(text: str) -> list[str]:
    return [t.casefold() for t in TOKEN_RE.findall(text) if len(t) > 1]


def _chunk_document(document: dict[str, Any], chunk_words: int = 150, overlap_words: int = 30) -> list[dict[str, Any]]:
    words = document.get("content", "").split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append({
            **document,
            "chunk_id": f"{document['document_id']}:{len(chunks) + 1}",
            "chunk_index": len(chunks),
            "excerpt": " ".join(words[start:end]),
        })
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


class RAGRetriever:
    """BM25 + BGE dense retrieval fused with reciprocal rank and a small reranker.

    Dense retrieval uses FastEmbed's local BAAI/bge-small-en-v1.5 model. If model
    assets are unavailable (for example, before the first model download), the
    retriever remains usable via BM25 and reports that fallback in its response.
    """

    def __init__(self) -> None:
        self._embedding_model_name: str | None = None
        self._embedder: Any = None
        self._embedding_failure: str | None = None
        self._vector_cache: dict[str, Any] = {}

    def _dense_rank(self, query: str, chunks: list[dict[str, Any]], model_name: str) -> list[tuple[float, int]]:
        if self._embedding_model_name != model_name:
            self._embedding_model_name = model_name
            self._embedder = None
            self._embedding_failure = None
            self._vector_cache.clear()
        if self._embedding_failure:
            return []
        try:
            if self._embedder is None:
                from fastembed import TextEmbedding
                self._embedder = TextEmbedding(model_name=model_name)
            query_vector = next(iter(self._embedder.embed(["Represent this sentence for searching relevant passages: " + query])))
            missing = [c for c in chunks if c["chunk_id"] not in self._vector_cache]
            if missing:
                for chunk, vector in zip(missing, self._embedder.embed([c["excerpt"] for c in missing])):
                    self._vector_cache[chunk["chunk_id"]] = vector
            q = query_vector
            ranked = []
            for index, chunk in enumerate(chunks):
                vector = self._vector_cache[chunk["chunk_id"]]
                # FastEmbed returns normalized vectors for BGE; cosine is explicit
                # here so this remains correct if provider defaults change.
                dot = sum(float(a) * float(b) for a, b in zip(q, vector))
                qn = math.sqrt(sum(float(x) ** 2 for x in q)) or 1.0
                vn = math.sqrt(sum(float(x) ** 2 for x in vector)) or 1.0
                ranked.append((dot / (qn * vn), index))
            return sorted(ranked, reverse=True)
        except Exception as exc:  # retrieval must remain available when local weights aren't installed
            self._embedding_failure = type(exc).__name__
            logger.warning("Dense retrieval unavailable (%s); using BM25 until the embedding model is available", type(exc).__name__)
            return []

    @staticmethod
    def _bm25_rank(query: str, chunks: list[dict[str, Any]]) -> list[tuple[float, int]]:
        query_terms = set(_tokens(query))
        tokenized = [_tokens(c["excerpt"] + " " + c.get("title", "")) for c in chunks]
        if not chunks or not query_terms:
            return []
        avg_len = sum(map(len, tokenized)) / len(tokenized) or 1.0
        doc_freq = Counter(term for terms in map(set, tokenized) for term in terms if term in query_terms)
        scored = []
        for index, terms in enumerate(tokenized):
            tf = Counter(terms)
            score = 0.0
            for term in query_terms:
                freq = tf[term]
                if not freq:
                    continue
                idf = math.log(1 + (len(chunks) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                score += idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * len(terms) / avg_len))
            if score > 0:
                scored.append((score, index))
        return sorted(scored, reverse=True)

    @staticmethod
    def _rerank_bonus(query: str, chunk: dict[str, Any]) -> int:
        """Small transparent boost for exact operational identifiers/entities."""
        q = query.casefold()
        text = (chunk.get("title", "") + " " + chunk.get("excerpt", "")).casefold()
        anchors = ("runway incursion", "takeoff roll", "runway crossing", "runway occupancy", "rejected takeoff", "controller", "surface detection", "takeoff clearance", "loss of separation")
        return sum(1 for anchor in anchors if anchor in q and anchor in text)

    def search(
        self,
        query: str,
        documents: Iterable[dict[str, Any]],
        *,
        embedding_model: str,
        document_types: list[str] | None = None,
        jurisdiction: str | None = None,
        incident_date: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        docs = list(documents)
        if document_types:
            wanted = set(document_types)
            docs = [d for d in docs if d.get("document_type") in wanted]
        if jurisdiction:
            docs = [d for d in docs if not d.get("jurisdiction") or d["jurisdiction"].casefold() == jurisdiction.casefold()]
        if incident_date:
            docs = [d for d in docs if not d.get("effective_date") or str(d["effective_date"]) <= incident_date]
        chunks = [chunk for doc in docs for chunk in _chunk_document(doc)]
        lexical = self._bm25_rank(query, chunks)
        dense = self._dense_rank(query, chunks, embedding_model)
        fused: dict[int, float] = {}
        for rank, (_, index) in enumerate(lexical, 1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (60 + rank)
        for rank, (_, index) in enumerate(dense, 1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (60 + rank)
        ranked = sorted(fused, key=lambda i: (-(fused[i] + self._rerank_bonus(query, chunks[i]) * 0.0002), chunks[i]["chunk_id"]))[:limit]
        results = []
        for index in ranked:
            chunk = chunks[index]
            results.append({key: chunk.get(key) for key in (
                "document_id", "chunk_id", "chunk_index", "title", "document_type",
                "authority_type", "source_status", "version", "source_id", "source_url",
                "jurisdiction", "document_date", "effective_date", "page_number", "section_reference",
                "historical_findings", "comparison_notes",
            )} | {
                "excerpt": chunk["excerpt"],
                "ranking_score": round(fused[index], 8),
                "rerank_anchor_matches": self._rerank_bonus(query, chunk),
            })
        method = "bm25+dense_bge_rrf+entity_rerank" if dense else "bm25+entity_rerank"
        limitations = ["Retrieved historical cases are analogies only. Their probable causes are not findings about the new case.", "A retrieved passage does not establish source applicability or prove a new case's facts."]
        if not dense:
            limitations.insert(0, f"BGE dense retrieval is not active yet ({self._embedding_failure or 'no dense candidates'}); this search used BM25 lexical retrieval.")
        return {"results": results, "retrieval_method": method, "limitations": limitations}


retriever = RAGRetriever()
