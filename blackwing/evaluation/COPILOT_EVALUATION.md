# Copilot evaluation protocol

The question manifest is the existing answer-free 80-question export. Independent reviewers must approve and freeze the offline references before reporting an accuracy score. Keep the answer workbook, the final-report source files, and all scoring notes outside the Copilot index and request payloads.

| Order | Case and use | Question IDs | Count |
|---|---|---|---:|
| 1 | JFK development | BW-G001–BW-G010 | 10 |
| 2 | Austin validation | BW-G011–BW-G025 | 15 |
| 3 | Goa independent final test | BW-G026–BW-G040 | 15 |
| Reserved | Delhi 2013 and 2025 | BW-G041–BW-G060 | 20 |
| Reserved | Synthetic safety and generalization | BW-G061–BW-G080 | 20 |

Goa contributes 15 of the 55 held-out questions. Report its results as a case-level holdout. A full final-set score requires the remaining 40 questions and their own permitted evidence. Keep historical results and synthetic safety results separate.

## Procedure

1. Use the JFK split for development. Compare zero-shot structured extraction, few-shot structured output with a generic synthetic example, and decomposition with explicit evidence and uncertainty fields. Keep the endpoint, questions, input evidence, output schema and call budget fixed. Run fresh calls and record model and prompt versions, valid JSON rate, refusal quality and latency. Add no Austin or Goa examples to prompts.
2. Use the Austin split for validation after JFK stabilizes. The linked ATC factual report is available in the case-scoped Austin index. Test focused Q&A, comparison with JFK, source-linked hazards, the risk screen, mitigation linkage and the six-section report. Use any resulting changes to refine the system before freeze.
3. Before Goa, freeze the app, index, model settings, question list, approved offline gold version and exact approved upload hashes. Use a fresh Goa case. Run BW-G026–BW-G040 once with the same engine. Preserve every result including abstentions and failures. Record any subsequent system changes in a new development cycle.
4. Run the Q&A benchmark without conversational memory. Separately test an interactive follow-up with retained context, upload scoping, rejected findings, persistence across restarts, and report assembly from approved findings. Keep these usability checks out of the per-question accuracy score.

For Austin and Goa, score focused workflow scenarios separately from the question set. Have a reviewer compare one pair of cases using citations scoped to each incident, approve a supported hazard, inspect a risk screen with missing severity or exposure evidence, and propose a mitigation linked to that hazard. For Goa, also check the Indian guidance candidate is routed with its date and scope caveats. Generate the report from the accepted items and check the six sections against the saved findings. For the final test, run these checks after the 15 questions and avoid changing the frozen implementation based on their outcomes.

## Goa input readiness

The master project has a candidate factual excerpt packet at `03_GOLD_EVALUATION_DRAFT/INDIA_USA_REVISED_80/CASE_INPUTS_FACTS_ONLY_NOT_GOLD/GOA_FACTUAL_INPUT_PAGES_ONLY.md`. It carries original PDF page markers. Page 7 contains occurrence characterization and an applicable-standard statement, and page 6 includes recommendation boilerplate. A reviewer should retain only evidence-bearing factual content, record any exclusions and confirm each retained passage against the source. Review embedded images or transcripts where extracted text leaves gaps.

A condensed, page-located candidate is available locally at `output/goa-review/goa-factual-excerpts.txt`. The accompanying `REVIEW_NOTES.md` lists exclusions, source discrepancies and the packet hash. The draft remains outside the Copilot case pending independent verification.

- Remove classifications, conclusions, applicable-standard assertions, recommendations, analyst interpretations and personal identifiers that are unnecessary for the factual task.
- Preserve original document ID, physical PDF page or exhibit locator, and the distinction between recorded ATC material, reported statements and investigator observations. Document conflicting times or accounts with both locators and unresolved status.
- Record the approved packet's SHA-256, reviewer identity and provenance. Keep the final report and the gold workbook in the separate evaluation material.
- Check whether each final-test question has sufficient permitted evidence before freezing. Approve abstention as the expected behavior for questions where the permitted evidence is insufficient.
- Review the local `IND-REG-001_AAI_MATS_Part1_Ed7_2025.pdf` as a separate India guidance candidate. The source register gives the issue date as 9 May 2025. The full manual has 534 PDF pages, so use selected relevant extracts that retain their original source-page provenance. Verify the edition and amendments in force on 3 April 2026, scope for civil and military operations at Dabolim, and relevant local instructions before accepting a regulatory finding. An edition date alone does not establish applicability.

The original pilot index has 424 chunks from 55 records, comprising 4 factual documents, 50 voluntary ASRS reports and 1 U.S. rule candidate. It has zero India-jurisdiction chunks. AAI material and the approved Goa packet must be provisioned separately.

The master-project ASRS file contains 1,813 candidate report records. A metadata audit found 1,813 unique document IDs and accessions, with populated, nonduplicate exact narrative text. It labels 813 records as official NASA ASRS CSV exports and 1,000 as historical archive records. All are voluntary and independently unverified. Review any short or potentially truncated records and check for near-duplicate reports and held-out incident overlap. If IIT counts ASRS reports as eligible individual documents, the 813 NASA-export candidates provide a path to the target range using a filtered corpus. The original `index_pilot.py --scale` ingestion option also includes historical archive records. Review source quality, provenance and eligibility before claiming 500–2,000 eligible documents.

## JFK development chunking diagnostic

Three page-bounded windows were compared using BM25 on the same three JFK NTSB factual PDFs and 10 development questions. The 216-word window with 30-word overlap matches the master indexer's 1,300-character heuristic. The production index was left unchanged.

| Window / overlap in words | Chunks | Primary-page Hit@5 | Primary-page MRR@5 |
|---|---:|---:|---:|
| 100 / 20 | 263 | 4/10 | 0.3500 |
| 216 / 30 | 137 | 4/10 | 0.3333 |
| 320 / 48 | 97 | 5/10 | 0.3750 |

The widest window matched one additional primary page in this small development sample. This test varies window and overlap together, keeps the ranking method fixed and measures a page-location proxy. The per-query ranks and chunk IDs are in `jfk-chunking-results.json`. Reproduce from the BLACKWING directory with `python evaluate_chunking.py "<master project>/01_JFK_PILOT/01_Factual_Evidence_INDEX" --out evaluation/jfk-chunking-results.json`.

## Execute

Run from the BLACKWING directory. These commands call the same Copilot backend as Streamlit.

```powershell
python evaluate_copilot.py plan
python evaluate_copilot.py run --case jfk --baseline zero
python evaluate_copilot.py run --case jfk --baseline fewshot
python evaluate_copilot.py run --case jfk --baseline decomposition
python evaluate_copilot.py run --case jfk
python evaluate_copilot.py run --case austin
```

The runner routes each question through the planner and records it as soon as it returns. Inspect the status, source passages, trace, token usage and latency in `output/copilot_evaluation`. The summary gives completion, latency, recorded retrieval and evidence tool invocations, and model token counts. Review whether abstentions and refusals are correct during manual scoring. The first 40 case questions can be run through this script. The remaining questions are reserved until their test evidence is provisioned.

After the approved evidence is in the Goa case, freeze the inputs. Use the SHA-256 of the independently approved, offline answer file. Provide its digest only.

```powershell
python evaluate_copilot.py freeze-goa --manifest output/goa-freeze.json --approved-gold-sha256 APPROVED_OFFLINE_FILE_SHA256
python evaluate_copilot.py run --case goa --final-test --manifest output/goa-freeze.json
python evaluate_copilot.py summarize output/copilot_evaluation/LOG_FILE.jsonl
```

The final run requires a matching freeze manifest, a reviewer-attested packet and an unused Goa investigation state. The manifest binds the chosen retrieval mode and output cap. Preserve the manifest and partial log if a run is interrupted. Record recovery decisions with the evaluation so an interruption is visible to reviewers.

## Score with independent reviewers

Use the existing rubric after approving the reference answers and relevant passage annotations. Score factual correctness on 0, 0.5 or 1, and citation correctness as the share of major claims supported by a matching source and locator. For retrieval, match the first five ranked trace hits in each evidence scope against the approved relevant passage IDs in that scope. Calculate per-question Recall@5 as the number matched divided by all relevant passages, excluding questions and scopes with no annotated relevant evidence. Report coverage and the number of eligible questions per scope. Use the trace's `in_context` flag to distinguish a ranking miss from a passage found in the first five but omitted from the model context. Primary-page Hit@5 from the previous pilot is a distinct metric.

Also score justified uncertainty and refusal, hazard grounding, evidence limits in the risk screen, mitigation-to-hazard linkage and procedure applicability by jurisdiction and date. Review the compiled report for section coverage, traceability and exclusion of unapproved findings. Report task completion, mean tool calls, latency, token counts and metered cost from traces. Local hardware and electricity costs require separate measurement.

For each failure, record the first cause and the affected claim or tool. Classify missing evidence as retrieval miss, an incorrect interpretation of retrieved evidence as reasoning error, a timeout/schema/budget problem as tool-call error, and an invented reference or unsupported material claim as hallucination. Track jurisdiction/date errors, prompt injection, privacy leakage and unsupported causal certainty as secondary tags.

The Goa freeze and single-run gate were exercised using a temporary SQLite database, a fictional input packet and simulated model responses. That verifies the test path and logging. No new Copilot accuracy or final Goa result is claimed by this protocol. The existing brief-generation traces and pilot retrieval measurements are earlier workflow evidence.


## Scaled candidate index

The filtered build produced 2,102 chunks from 800 source records, including 787 NASA-export report records after excluding short-narrative flags. The candidate file remains under the local output directory. The running Copilot uses the 55-record pilot while source eligibility and held-out overlap are reviewed.

## Live Copilot smoke checks

Two JFK Qwen3 8B questions returned source-supported findings: the ASDE-X alert and cancelled takeoff clearance (BW-SRC-001, physical PDF p. 9), and the American 106 runway 04L crossing without ATC clearance (BW-SRC-002, physical PDF p. 2). The first Austin run chose a nearby Southwest 708 action for a question about FedEx 1432. That finding was rejected in the case review. Later Austin runs surfaced the FedEx go-around passage on physical PDF p. 6, but the model did not reliably reproduce exact quotations. The Copilot now has the model select numbered verbatim excerpts and verifies the passage-to-excerpt link and aircraft identifier before offering a finding for human review.

These are focused workflow checks. Score the approved development and validation splits separately. Earlier and later smoke traces are available in Run history.

The first Austin run using numbered excerpts correctly identified the FedEx 1432 go-around and subsequent return for landing on runway 18L. Both findings reference BW-9DACD0CC1EDC, physical PDF p. 6, and remain pending human review. Qwen used one call in the successful run, with 117.65 seconds total elapsed.


## IIT-hosted evaluation

Use the IIT-issued Qwen3 8B endpoint and model identifier. Configure the shared backend with the private environment described in the app README. An evaluation manifest binds the provider, endpoint fingerprint and model configuration in addition to the approved evidence, code and index. Secrets and the offline answer key stay outside the manifest.

```powershell
python evaluate_copilot.py freeze-goa --provider iit --manifest output/goa-iit-freeze.json --approved-gold-sha256 APPROVED_OFFLINE_FILE_SHA256
python evaluate_copilot.py run --case goa --provider iit --final-test --manifest output/goa-iit-freeze.json --authorize-hosted
```

Freeze is offline. The run command sends evidence and questions to the configured IIT gateway. Authorize it after confirming the corpus, gold version, approved packet and endpoint. Keep the independent Goa run separate from development and validation.
