from __future__ import annotations

import csv
import json
import re

import numpy as np
import pytest

from blackwing.rag.corpus import load_manifest
from blackwing.rag.embeddings import BATCH_SIZE, CachedEmbeddings, EmbeddingFailure
from blackwing.rag.engine import HybridIndex
from blackwing.rag.evaluate import evaluate, metrics_for_ranking, validate_citation_ids
from blackwing.rag.retriever import RAGRetriever
from blackwing.rag.tokenization import ApproximateTokenizer, TokenBudgetError, assert_budget, chunk_document


class TestTokenizer:
    verified = True

    def offsets(self, text):
        return [m.span() for m in re.finditer(r"\S+", text)]

    def count(self, text):
        return len(self.offsets(text)) + 2


class FakeBGE:
    identity = "test:384-bge"

    def __init__(self):
        self.tokenizer = TestTokenizer()
        self.batches = []

    def encode_batch(self, texts):
        self.batches.append(list(texts))
        vectors = []
        for text in texts:
            arr = np.zeros(384, dtype=np.float32)
            for term in set(text.casefold().split()):
                arr[sum(term.encode()) % 384] += 1
            if not np.any(arr):
                arr[0] = 1
            vectors.append(arr)
        return np.asarray(vectors, dtype=np.float32)


def docs():
    return [
        {"document_id": "A", "source_id": "A", "text": "Runway 04L was occupied while another aircraft was departing. Crossing clearance was not issued.",
         "incident_id": "CASE-A", "source_type": "official_factual", "document_type": "incident_report", "jurisdiction": "US", "page_number": 4},
        {"document_id": "B", "source_id": "B", "text": "Runway 18L had a conflicting arriving aircraft in low visibility. The crew executed a go around.",
         "incident_id": "CASE-B", "incident_date": "2023-01-01", "source_type": "official_factual", "document_type": "incident_report", "jurisdiction": "US", "page_number": 9},
        {"document_id": "R", "source_id": "R", "text": "Airport traffic procedures require verified runway crossing clearance by the controller.",
         "source_type": "regulation", "document_type": "regulation", "jurisdiction": "US", "effective_date": "2022-01-01"},
        {"document_id": "NEW", "source_id": "NEW", "text": "New runway crossing clearance standard became effective in January 2025.",
         "source_type": "regulation", "document_type": "regulation", "jurisdiction": "US", "effective_date": "2025-01-01"},
        {"document_id": "EU", "source_id": "EU", "text": "European runway incursion recommendations and action planning.",
         "source_type": "advisory", "document_type": "guidance", "jurisdiction": "Europe", "publication_date": "2024-01-01"},
        {"document_id": "FIN", "source_id": "FIN", "text": "Final probable cause applied only to a separate incident in the historical investigation.",
         "source_type": "official_final", "document_type": "incident_report", "incident_id": "CASE-A"},
    ]


def test_chunking_three_methods_preserve_source_and_pages():
    for name in ("fixed", "overlap", "structure"):
        records = chunk_document({"document_id": "ANY-INCIDENT", "source_id": "S", "page_number": 7,
                                  "content": "Crossing clearances matter. " * 160}, name, TestTokenizer())
        assert len(records) >= 2
        assert all(row["page_number"] == 7 and row["source_id"] == "S" for row in records)
        assert len({x["chunk_id"] for x in records}) == len(records)
        assert all(row["token_count"] <= 512 for row in records)


def test_token_budget_512_including_query_prefix():
    with pytest.raises(TokenBudgetError):
        assert_budget("test " * 520, TestTokenizer())
    with pytest.raises(TokenBudgetError):
        assert_budget("some text", ApproximateTokenizer())
    assert_budget("test departure aircraft", TestTokenizer(), query=True)


def test_embeds_batch_cache_and_dimension(tmp_path):
    backend = FakeBGE()
    cache = CachedEmbeddings(backend, tmp_path / "cache.sqlite3")
    texts = [f"passage {i} runway" for i in range(35)]
    vectors = cache.encode(texts)
    assert vectors.shape == (35, 384)
    assert [len(batch) for batch in backend.batches] == [16, 16, 3]
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    cache.encode(texts)
    assert len(backend.batches) == 3
    cache.encode(["runway incursion"], query=True)
    assert backend.batches[-1][0].startswith("Represent this sentence")


def test_scoping_historical_regulation_guidance_and_direct():
    index = HybridIndex.from_documents(docs(), strategy="fixed")
    direct = index.search("runway crossing", scope="direct", case_id="CASE-A", method="bm25")
    assert {h["document_id"] for h in direct["results"]} == {"A"}
    historical = index.search("arrival visibility", scope="historical", case_id="CASE-A", method="bm25")
    assert {h["document_id"] for h in historical["results"]} == {"B"}
    assert index.search("crossing", scope="direct", case_id="UNKNOWN", method="bm25")["results"] == []
    rules = index.search("runway crossing", scope="regulatory", jurisdiction="US", incident_date="2023-01-01", method="bm25")
    assert {h["document_id"] for h in rules["results"]} == {"R"}
    assert index.search("runway", scope="regulatory", jurisdiction="Europe", incident_date="2023-01-01", method="bm25")["results"] == []
    assert index.search("runway incursion", scope="guidance", incident_date="2023-01-01", method="bm25")["results"] == []
    guidance = index.search("runway incursion", scope="guidance", incident_date="2023-01-01", method="bm25",
                            include_retrospective_guidance=True)
    assert {h["document_id"] for h in guidance["results"]} == {"EU"}
    assert "not binding" in guidance["results"][0]["caution"]


def test_manifest_approval_protection_duplicate_and_csv_provenance(tmp_path):
    (tmp_path / "incidents").mkdir()
    (tmp_path / "incidents" / "source.txt").write_text("Runway 12 was occupied by a taxiing aircraft. " * 5)
    (tmp_path / "GOLD_EVALUATION.txt").write_text("Hidden gold answer never index this text." * 2)
    (tmp_path / "asrs.csv").write_text('ACN,Narrative\n123,"Aircraft reported runway crossing conflict while taxiing to holding point"\n')
    manifest = {"sources": [
      {"source_id": "F", "path": "incidents/source.txt", "source_type": "official_factual", "approved": True, "incident_id": "CASE-ANY"},
      {"source_id": "D", "path": "incidents/source.txt", "source_type": "official_factual", "approved": True},
      {"source_id": "G", "path": "GOLD_EVALUATION.txt", "source_type": "official_factual", "approved": True},
      {"source_id": "V", "path": "asrs.csv", "source_type": "voluntary_asrs", "approved": True,
       "record_id_column": "ACN", "text_column": "Narrative", "pii_reviewed": True},
    ]}
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(manifest))
    rows, audit = load_manifest(path, tmp_path)
    assert audit.approved_document_count == 2  # CSV counts as one source file, not one approved IIT ASRS record.
    assert audit.asrs_record_count == 1
    assert {x["document_id"] for x in rows} == {"F", "V:123"}
    assert any("protected" in x["reason"] for x in audit.exclusions)
    assert not audit.meets_iit_document_count


def test_build_save_load_refuse_overwrite_tampering(tmp_path):
    backend = CachedEmbeddings(FakeBGE())
    index = HybridIndex.from_documents(docs(), embeddings=backend)
    path = tmp_path / "version-001"
    manifest = index.save(path)
    assert manifest["embedding_model"] == "test:384-bge"
    reopened = HybridIndex.load(path, embeddings=CachedEmbeddings(FakeBGE()))
    assert len(reopened.chunks) == len(index.chunks)
    with pytest.raises(FileExistsError):
        index.save(path)
    with pytest.raises(ValueError):
        HybridIndex.load(path, embeddings=object())
    (path / "chunks.jsonl").write_text((path / "chunks.jsonl").read_text() + '{"chunk_id":"tampered"}\n')
    with pytest.raises(ValueError):
        HybridIndex.load(path)


def test_metrics_not_falsely_labeled_true_recall():
    ranks = ["A", "X", "B"]
    assert metrics_for_ranking(ranks, {"A", "B"}, complete=True, k=5)["recall"] == 1.0
    assert metrics_for_ranking(ranks, {"A", "B"}, complete=False, k=5)["recall"] is None
    assert validate_citation_ids(["A", "fake"], {"A"})["invalid_citations"] == ["fake"]
    index = HybridIndex.from_documents(docs())
    gold = [{"question_id": "Q1", "query": "runway crossing", "complete_relevance": "false",
             "relevant_chunk_ids": index.chunks[0]["chunk_id"]}]
    result = evaluate(index, gold, methods=["bm25"])
    assert result["methods"]["bm25"]["recall@5"] is None


def test_team_api_search_signature_compatible_without_model():
    # Mocked embeddings prevent network/model download and prove public API compatibility.
    rag = RAGRetriever(embedder=CachedEmbeddings(FakeBGE()))
    data = [{"document_id": "reg-1", "title": "Runway guidance", "content": "Runway entry control requires positive clearance.",
             "document_type": "procedure", "jurisdiction": "US", "source_id": "SRC1", "effective_date": "2022-01-01"}]
    output = rag.search("runway entry clearance", data, embedding_model="BAAI/bge-small-en-v1.5",
                        document_types=["procedure"], jurisdiction="US", incident_date="2023-01-01", limit=5)
    assert output["results"][0]["source_id"] == "SRC1"
    assert output["results"][0]["citation"]["chunk_id"]
    assert output["results"][0]["citation_label"]


def test_chunking_comparison_only_development():
    from blackwing.rag.chunking_experiments import compare
    question = [{"question_id": "Q1", "query": "runway crossing",
                 "expected_source_pages": "A#4", "split": "development"}]
    result = compare(docs(), question, TestTokenizer())
    assert len(result["methods"]) == 3
    assert "not verified" in result["methods"]["fixed"]["retrieval_recall_at_k"].lower()
    question[0]["split"] = "final"
    with pytest.raises(ValueError):
        compare(docs(), question, TestTokenizer())


def test_long_narrative_is_bounded_for_bge_query():
    backend = CachedEmbeddings(FakeBGE())
    idx = HybridIndex.from_documents(docs(), embeddings=backend)
    result = idx.search("runway " * 700, method="dense", top_k=3)
    assert result["metrics"]["query_truncated"] is True
    assert result["results"]


def test_case_context_keeps_analogy_not_direct():
    from blackwing.rag.context import retrieve_case_context
    rag = RAGRetriever(embedder=CachedEmbeddings(FakeBGE()))
    content = docs()
    for row in content:
        row["content"] = row.pop("text")
    result = retrieve_case_context(rag, "An aircraft unexpectedly crossed an active runway.", content,
                                   case_id="SOME-NEW-CASE", jurisdiction="US", incident_date="2023-01-01")
    assert all(row["retrieval_scope"] != "direct" for row in result)
    assert any(row["retrieval_scope"] == "regulatory" for row in result)
    assert len(result) <= 5


def test_manifest_draft_does_not_auto_approve_files(tmp_path):
    from blackwing.rag.manifest_draft import draft_manifest
    (tmp_path / "runway_report.txt").write_text("An occurrence was reported while taxiing.")
    (tmp_path / "heldout_gold.txt").write_text("Gold answer not for RAG index.")
    draft = draft_manifest(tmp_path)
    assert len(draft["sources"]) == 1
    assert draft["sources"][0]["approved"] is False
    assert draft["audit"]["unique_approved_documents"] == 0


def test_hybrid_rrf_and_crossencoder_are_separate_methods():
    class FakeCrossEncoder:
        def score(self, question, passages):
            return [10.0 if "go around" in passage.lower() else 0.0 for passage in passages]
    index = HybridIndex.from_documents(docs(), embeddings=CachedEmbeddings(FakeBGE()))
    hybrid = index.search("aircraft runway", method="hybrid", top_k=5)
    assert hybrid["retrieval_method"] == "hybrid"
    reranked = index.search("aircraft runway", method="rerank", reranker=FakeCrossEncoder(), top_k=5)
    assert reranked["metrics"]["reranker_status"] == "crossencoder"
    assert reranked["results"][0]["document_id"] == "B"
    without = index.search("aircraft runway", method="rerank", top_k=5)
    assert any("NOT ACTIVE" in note for note in without["limitations"])


def test_historical_date_filter_rejects_future_and_unknown_occurrences():
    idx = HybridIndex.from_documents(docs())
    before = idx.search("runway", scope="historical", case_id="NEW", incident_date="2022-01-01", method="bm25")
    assert before["results"] == []
    after = idx.search("runway", scope="historical", case_id="NEW", incident_date="2024-01-01", method="bm25")
    assert {r["source_id"] for r in after["results"]} == {"B"}
