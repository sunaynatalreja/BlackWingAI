"""Read-only smoke against existing corpus. No embeddings, models or index writes."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from blackwing.rag.engine import HybridIndex
from blackwing.tools import tool_audit, retrieve_evidence, find_similar_incidents, assess_reference_risk

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
integrity = []
for name in ("expanded_bge_asrs_20261121", "expanded_bge_asrs_metadata_v2"):
    directory = ROOT / "rag_indexes" / name
    manifest = json.loads((directory / "manifest.json").read_text())
    vectors = np.load(directory / "vectors.npy", allow_pickle=False)
    vector_hash = hashlib.sha256(vectors.astype(np.float32).tobytes()).hexdigest()
    assert vector_hash == manifest["vector_sha256"]
    index = HybridIndex.load(directory)  # validates chunk signature, lexical only
    assert len(index.chunks) == 27255
    integrity.append({"index": name, "document_count": manifest["document_count_indexed"],
                      "chunk_count": len(index.chunks), "vector_sha256": vector_hash,
                      "chunk_signature_valid": True, "vector_hash_valid": True,
                      "mode": "read-only; no embedding generation"})

selected = next(row for row in index.chunks if row.get("source_type") == "official_factual" and row.get("incident_id"))
incident_id = selected["incident_id"]
events = []
query = "runway taxi clearance aircraft crossing"
with tool_audit("LOCAL-READONLY-SMOKE", events.append):
    direct = retrieve_evidence(index.search, input_identifiers=[incident_id], query=query,
                               scope="direct", case_id=incident_id, method="bm25", top_k=3)
    historical = index.search(query, scope="historical", case_id=incident_id, method="bm25", top_k=3)
    # Match build_agent_packets: label records returned by historical scope as
    # analogies, without altering stored metadata or promoting them to facts.
    analogies = [{**row, "evidence_role": "historical_analogy"} for row in historical["results"]]
    comparison = find_similar_incidents(narrative=query, evidence=analogies, current_incident_id=incident_id)
    # No event-specific severity/likelihood inference has been performed. The
    # real calculator must return unassessed, not create unsupported ratings.
    assessment = assess_reference_risk(evidence_ids=[r["chunk_id"] for r in direct["results"]])
assert direct["results"]
assert assessment["risk_index"] is None
assert len(events) == 3 and all(e["status"] == "complete" for e in events)
(out / "corpus_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
(out / "actual_local_tool_executions.json").write_text(json.dumps({
    "test_type": "read-only real local index and deterministic tools; no model calls",
    "lookup_incident_id": incident_id, "tool_events": events,
    "historical_matches": len(comparison["matches"]), "risk_status": assessment["status"],
}, indent=2), encoding="utf-8")
print(json.dumps({"corpora": integrity, "tools": [{"name": e["tool"], "status": e["status"], "duration_ms": e["duration_ms"]} for e in events]}))
