from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .tokenization import BGE_MODEL, BGE_QUERY_PREFIX, BGESubwordTokenizer, assert_budget

DIMENSION = 384
BATCH_SIZE = 16


class EmbeddingFailure(RuntimeError):
    pass


def normalize(vectors: Any, expected: int) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.shape != (expected, DIMENSION) or not np.isfinite(arr).all():
        raise EmbeddingFailure(f"Expected finite {expected} x {DIMENSION} embedding matrix; got {arr.shape}")
    lengths = np.linalg.norm(arr, axis=1, keepdims=True)
    if (lengths <= 0).any():
        raise EmbeddingFailure("Zero-length embedding vector")
    return arr / lengths


class LocalBGE:
    identity = "local:" + BGE_MODEL

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self._model: Any = None

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EmbeddingFailure("Install fastembed from requirements-rag.txt") from exc
        if self._model is None:
            self._model = TextEmbedding(model_name=BGE_MODEL)
        return normalize(list(self._model.embed(list(texts), batch_size=BATCH_SIZE)), len(texts))


class IITBGE:
    """OpenAI-compatible /v1/embeddings. CA verification is mandatory."""
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.url = os.getenv("BLACKWING_EMBEDDING_URL", "").strip()
        self.key = os.getenv("BLACKWING_EMBEDDING_API_KEY", "").strip()
        self.ca = os.getenv("BLACKWING_EMBEDDING_CA_CERT", "").strip()
        self.model = os.getenv("BLACKWING_EMBEDDING_MODEL", "bge-small-en-v1.5")
        self.timeout = float(os.getenv("BLACKWING_EMBEDDING_TIMEOUT", "60"))
        self.retries = min(5, max(0, int(os.getenv("BLACKWING_EMBEDDING_RETRIES", "2"))))
        self._last_request = 0.0  # single key: serialized calls, <=60 requests/min
        if not self.url.startswith("https://") or not self.key or not self.ca or not Path(self.ca).is_file():
            raise EmbeddingFailure("Hosted embeddings require HTTPS URL, API key, and existing trusted CA certificate")
        self.identity = "iit:" + self.model + ":" + hashlib.sha256(self.url.encode()).hexdigest()[:12]

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        import httpx
        headers = {"Authorization": f"Bearer {self.key}"}
        # Never set verify=False or log the API key.
        with httpx.Client(verify=self.ca, timeout=self.timeout) as client:
            for attempt in range(self.retries + 1):
                try:
                    elapsed = time.monotonic() - self._last_request
                    if elapsed < 1.05:
                        time.sleep(1.05 - elapsed)
                    self._last_request = time.monotonic()
                    response = client.post(self.url, headers=headers,
                                           json={"model": self.model, "input": list(texts)})
                    if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                        time.sleep(min(8.0, 0.5 * (2 ** attempt)))
                        continue
                    response.raise_for_status()
                    data = response.json().get("data", [])
                    if not isinstance(data, list) or len(data) != len(texts):
                        raise EmbeddingFailure("Embedding response count mismatch")
                    ordered = sorted(data, key=lambda x: x.get("index", -1))
                    if [item.get("index") for item in ordered] != list(range(len(texts))):
                        raise EmbeddingFailure("Embedding response indices are invalid")
                    return normalize([item["embedding"] for item in ordered], len(texts))
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    if attempt == self.retries:
                        raise EmbeddingFailure(f"Hosted embedding network error: {type(exc).__name__}") from exc
                    time.sleep(min(8.0, 0.5 * (2 ** attempt)))
                except httpx.HTTPStatusError as exc:
                    raise EmbeddingFailure(f"Embedding HTTP {exc.response.status_code}") from exc
        raise EmbeddingFailure("Hosted embedding retries exhausted")


class CachedEmbeddings:
    def __init__(self, backend: LocalBGE | IITBGE, cache_path: str | Path | None = None) -> None:
        self.backend = backend
        self.identity = backend.identity
        self.tokenizer = backend.tokenizer
        self.cache_path = Path(cache_path) if cache_path else None
        if not getattr(self.tokenizer, "verified", False):
            raise EmbeddingFailure("Real BGE tokenizer is required before embedding")
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.cache_path) as con:
                con.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, model TEXT NOT NULL, value BLOB NOT NULL)")

    def _key(self, text: str) -> str:
        return hashlib.sha256((self.identity + "\0" + text).encode()).hexdigest()

    def _cached(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        if self.cache_path is None or not keys:
            return {}
        found = {}
        with sqlite3.connect(self.cache_path) as con:
            for key in keys:
                row = con.execute("SELECT value FROM vectors WHERE key=? AND model=?", (key, self.identity)).fetchone()
                if row:
                    vector = np.frombuffer(row[0], dtype=np.float32).copy()
                    if vector.shape == (DIMENSION,) and np.isfinite(vector).all():
                        found[key] = vector
        return found

    def encode(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, DIMENSION), dtype=np.float32)
        prepared = [(BGE_QUERY_PREFIX + text if query else text) for text in texts]
        for text in texts:
            assert_budget(text, self.tokenizer, query=query)
        keys = [self._key(text) for text in prepared]
        found = self._cached(keys)
        unique = {}
        for key, text in zip(keys, prepared):
            if key not in found:
                unique[key] = text
        missing = list(unique.items())
        for start in range(0, len(missing), BATCH_SIZE):
            batch = missing[start:start + BATCH_SIZE]
            vectors = self.backend.encode_batch([text for _, text in batch])
            if self.cache_path:
                with sqlite3.connect(self.cache_path) as con:
                    con.executemany("INSERT OR REPLACE INTO vectors (key, model, value) VALUES (?, ?, ?)",
                                    [(key, self.identity, vector.astype(np.float32).tobytes())
                                     for (key, _), vector in zip(batch, vectors)])
            found.update((key, vector) for (key, _), vector in zip(batch, vectors))
        return normalize([found[key] for key in keys], len(keys))


def make_embeddings(mode: str, *, cache_path: str | Path | None = None,
                    tokenizer: Any | None = None) -> CachedEmbeddings | None:
    if mode == "none":
        return None
    tokenizer = tokenizer or BGESubwordTokenizer()
    if mode == "local":
        return CachedEmbeddings(LocalBGE(tokenizer), cache_path)
    if mode == "iit":
        return CachedEmbeddings(IITBGE(tokenizer), cache_path)
    raise ValueError("Embedding mode must be none, local or iit")
