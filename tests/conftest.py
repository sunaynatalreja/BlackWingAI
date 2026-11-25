"""Regression tests must never reach Ollama, hosted models, or external downloads."""
import socket
import sys
import pytest


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    def blocked(sock, address, *args, **kwargs):
        # Windows asyncio implements its internal socketpair on loopback.
        if sys._getframe(1).f_code.co_name == "_fallback_socketpair":
            return original_connect(sock, address, *args, **kwargs)
        raise AssertionError("Outbound network is forbidden in automated tests; use a mock transport")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)

    # Use lexical-only small fixture corpora; no model downloads or embeddings.
    from blackwing.rag.retriever import retriever
    monkeypatch.setattr(retriever, "_embedding", lambda mode="local": None)
