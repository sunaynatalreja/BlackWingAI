from __future__ import annotations

from typing import Any, Iterable

from .retriever import RAGRetriever


def retrieve_case_context(rag: RAGRetriever, narrative: str, documents: Iterable[dict[str, Any]], *,
                          case_id: str | None = None,
                          incident_date: str | None = None,
                          jurisdiction: str | None = None,
                          embedding_model: str = "BAAI/bge-small-en-v1.5",
                          limit: int = 5) -> list[dict[str, Any]]:
    """Keep selected evidence bounded to small-model context and attach explicit scopes."""
    docs = list(documents)
    query = " ".join(narrative.strip().split()[:150]) + " runway incursion"
    per_scope = (
        ("direct", 2), ("regulatory", 2), ("guidance", 2), ("historical", 2),
    )
    found: list[dict[str, Any]] = []
    for scope, n in per_scope:
        if len(found) >= limit:
            break
        if scope == "direct" and not case_id:
            continue
        if scope == "regulatory" and (not jurisdiction or not incident_date):
            continue
        response = rag.search(query, docs, embedding_model=embedding_model,
                             scope=scope, case_id=case_id,
                             jurisdiction=jurisdiction if scope == "regulatory" else None,
                             incident_date=incident_date, limit=min(n, limit - len(found)))
        for hit in response["results"]:
            if hit["chunk_id"] not in {row["chunk_id"] for row in found}:
                found.append({**hit, "retrieval_scope": scope})
    return found[:limit]
