# Austin RAG — Local Retrieval Evaluation

## Test Configuration

- Incident: Austin runway incursion, 4 February 2023
- Evaluation questions: 15
- Indexed documents: 12
- Indexed chunks: 2,363
- Embeddings: Local BGE-small-en-v1.5
- Retrieval: BM25, Dense, Hybrid RRF, CrossEncoder

## Measured Results

| Method | Expected-page Hit@5 | Hit@10 | MRR@10 |
|---|---:|---:|---:|
| BM25 | 100.0% | 100.0% | 0.8389 |
| Dense | 93.3% | 93.3% | 0.7633 |
| Hybrid RRF | 100.0% | 100.0% | 0.7833 |
| Hybrid + CrossEncoder | 100.0% | 100.0% | 0.9667 |

## Integration

- Risk and Compliance integration tests: 24 passed
- Seven-agent evidence routing: Offline tests passed
- PDF extraction fallback: Tested successfully

## Evaluation Scope

Results are based on expected evidence-page matching.
True passage-level Recall@k has not yet been established.
The evaluation is limited to the Austin development workflow.

## Remaining Validation

- Three chunking accuracy comparisons
- Complete Recall@5/10 relevance judgments
- Live Risk and Compliance report-quality evaluation
- Full IIT corpus evaluation
