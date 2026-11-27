"""Bounded in-process cache for read-only shared-knowledge searches only."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from threading import RLock
from time import monotonic


class KnowledgeSearchCache:
    def __init__(self, max_entries: int = 256):
        self.max_entries = max_entries
        self._items: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = RLock()

    @staticmethod
    def key(request: dict, documents: list[dict], index_dir: str | None, embedding_model: str) -> str:
        manifest = ""
        if index_dir:
            from pathlib import Path
            path = Path(index_dir) / "manifest.json"
            try:
                stat = path.stat()
                manifest = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
            except OSError:
                manifest = str(path)
        corpus = hashlib.sha256(json.dumps(documents, sort_keys=True, default=str).encode()).hexdigest()
        material = json.dumps([request, corpus, manifest, embedding_model], sort_keys=True, default=str)
        return hashlib.sha256(material.encode()).hexdigest()

    def get(self, key: str, ttl_seconds: int):
        if ttl_seconds <= 0:
            return None
        with self._lock:
            row = self._items.get(key)
            if row is None:
                return None
            expires, value = row
            if expires <= monotonic():
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value: dict, ttl_seconds: int):
        if ttl_seconds <= 0:
            return
        with self._lock:
            self._items[key] = (monotonic() + ttl_seconds, value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)


knowledge_search_cache = KnowledgeSearchCache()
