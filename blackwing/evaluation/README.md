# BLACKWING evaluation

## Retrieval

The existing JFK development split contains 10 questions with draft source-page labels. Both runs used the 424-chunk pilot index.

| Retrieval | Source document Hit@5 | Primary page Hit@5 | Primary page MRR@5 |
|---|---:|---:|---:|
| BM25 | 9/10 | 5/10 | 0.367 |
| BM25 + MiniLM + cross-encoder | 10/10 | 8/10 | 0.533 |

Hybrid retrieval missed the designated primary pages for BW-G003 and BW-G010 even though it found the expected document for both. A match establishes that a page was retrieved. Exact evidence support requires review, particularly for the figure-dependent BW-G005. Full traces and retrieval CSVs are in the local execution outputs.

## Failure taxonomy

| Category | Decision rule | What to record |
|---|---|---|
| Retrieval miss | A required source passage is absent from the retrieved context | Expected and retrieved source IDs and page locators |
| Reasoning error | The draft misreads or contradicts a relevant passage that was supplied | The claim, cited passage and corrected interpretation |
| Tool-call error | Retrieval, endpoint, parsing, schema or token-budget handling prevents completion | Stage, status, latency and attempt |
| Hallucination | A citation is invented or a claim lacks support in the reviewed evidence | Unsupported claim or identifier and evidence scope |

The initial live JFK run retrieved evidence successfully and timed out during CPU model inference. The client timeout was increased and the model prompt shortened. In controlled tests with simulated responses, the critic rejected invented citation IDs and accepted a corrected response on its bounded retry. The same test confirmed submitted text remained in the new incident scope and an email address was masked before persistence.

## Reproduce

- Run `python evaluate_retrieval.py` and `python evaluate_retrieval.py --hybrid --rerank` from the BLACKWING directory.
- Run `python review_workflow.py` to generate the three live model traces, then `python demo_results.py` for completion, latency and usage.
- Review the brief against the cited original pages before awarding claim accuracy. The full 80-question draft, comparative live prompt baselines and audited 500–2,000-document corpus remain outside the measured pilot.

## Case review notes

- **JFK.** The first run timed out after retrieval. The next run returned JSON, but its citations were rejected. Scoped citation decoding resolved that structural failure. The broad-retrieval brief still missed the intervention, so a case-independent topic retrieval pass was added. The revised brief cited takeoff clearance, the ASDE-X alert, the controller's cancellation and the rejected takeoff. It left the American 106 crossing out of the timeline even though the Airports factual passage documenting the crossing was retrieved. The reviewer should record that as a generation coverage error.
- **Austin.** The first structurally valid brief confused an investigation visit with an operational timeline event. The same topic retrieval pass was tested on Austin to surface the operational clearance and go-around passages.
- **Mumbai.** The third case uses a fictional narrative in a separate case scope. The PDF and text upload tests exercise extraction, provenance, scope, redaction and rejection of an invented citation.
