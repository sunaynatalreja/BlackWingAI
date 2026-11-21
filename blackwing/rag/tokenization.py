from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_MAX_TOKENS = 512


class TokenBudgetError(ValueError):
    pass


class BGESubwordTokenizer:
    verified = True

    def __init__(self, model: str = BGE_MODEL) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install requirements-rag.txt to use the real BGE tokenizer") from exc
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError("BGE offset-aware tokenization requires a fast tokenizer")
        # This tokenizer scans long source documents before splitting them. It never
        # feeds those full documents to a model; enforce the real 512-token limit
        # explicitly on each chunk in assert_budget/_windows instead of emitting the
        # Transformers model-length warning while examining the unsplit source.
        self.tokenizer.model_max_length = 1_000_000_000

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(a, b) for a, b in self.tokenizer(text, add_special_tokens=False,
                  return_offsets_mapping=True)["offset_mapping"] if b > a]

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


class ApproximateTokenizer:
    """Explicit *unverified* lexical-only fallback; never validate embedding limits with it."""
    verified = False

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in re.finditer(r"\S+", text)]

    def count(self, text: str) -> int:
        return len(self.offsets(text)) + 2


@dataclass(frozen=True)
class ChunkPlan:
    name: str
    width: int
    overlap: int


PLANS = {
    "fixed": ChunkPlan("fixed", 180, 0),
    "overlap": ChunkPlan("overlap", 180, 36),
    "structure": ChunkPlan("structure", 220, 32),
}


def assert_budget(text: str, tokenizer: Any, *, query: bool = False) -> None:
    if not getattr(tokenizer, "verified", False):
        raise TokenBudgetError("A verified BGE tokenizer is required for embedding input validation")
    payload = BGE_QUERY_PREFIX + text if query else text
    n = tokenizer.count(payload)
    if n > BGE_MAX_TOKENS:
        raise TokenBudgetError(f"BGE input is {n} tokens; maximum {BGE_MAX_TOKENS}")


def _windows(text: str, tokenizer: Any, *, plan: ChunkPlan):
    offsets = tokenizer.offsets(text)
    if not offsets:
        return
    start = 0
    while start < len(offsets):
        end = min(start + plan.width, len(offsets))
        piece = text[offsets[start][0]:offsets[end - 1][1]].strip()
        # Guard against non-English tokenization peculiarities/metadata overhead.
        while piece and getattr(tokenizer, "verified", False) and tokenizer.count(piece) > BGE_MAX_TOKENS:
            if end - start <= 1:
                raise TokenBudgetError("One indivisible unit exceeds the BGE token limit")
            end -= 1
            piece = text[offsets[start][0]:offsets[end - 1][1]].strip()
        if piece:
            yield piece
        if end == len(offsets):
            break
        start = max(start + 1, end - plan.overlap)


def chunk_document(document: Mapping[str, Any], strategy: str, tokenizer: Any) -> list[dict[str, Any]]:
    """A document is one page or one record with text/content and original metadata."""
    if strategy not in PLANS:
        raise ValueError(f"Unknown strategy {strategy!r}")
    plan = PLANS[strategy]
    text = str(document.get("text") or document.get("content") or "").strip()
    if not text:
        return []
    segments = re.split(r"\n\s*\n", text) if strategy == "structure" else [text]
    records = []
    count = 0
    for segment in segments:
        for excerpt in _windows(segment, tokenizer, plan=plan):
            docid = str(document["document_id"])
            locator = document.get("page_number") or document.get("record_id") or "text"
            short_hash = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()[:10]
            chunk_id = f"{docid}:loc{locator}:{strategy}:{count:05d}:{short_hash}"
            records.append({**document,
                "chunk_id": chunk_id, "chunk_index": count,
                "excerpt": excerpt, "text": excerpt,
                "token_count": tokenizer.count(excerpt),
                "tokenizer_verified": bool(getattr(tokenizer, "verified", False)),
                "chunk_strategy": strategy,
            })
            count += 1
    return records


def clip_query(text: str, tokenizer: Any, max_query_tokens: int = 260) -> tuple[str, bool]:
    """Deterministically bound new-case narratives before BGE query embedding."""
    if not getattr(tokenizer, "verified", False):
        return (" ".join(text.split()[:150]), len(text.split()) > 150)
    offsets = tokenizer.offsets(text)
    if len(offsets) <= max_query_tokens and tokenizer.count(BGE_QUERY_PREFIX + text) <= BGE_MAX_TOKENS:
        return text, False
    n = min(len(offsets), max_query_tokens)
    if n < 1:
        return "", bool(text)
    candidate = text[offsets[0][0]:offsets[n - 1][1]]
    while n > 1 and tokenizer.count(BGE_QUERY_PREFIX + candidate) > BGE_MAX_TOKENS:
        n -= 1
        candidate = text[offsets[0][0]:offsets[n - 1][1]]
    assert_budget(candidate, tokenizer, query=True)
    return candidate, True
