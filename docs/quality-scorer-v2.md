# BLACKWING completeness scorer v2 integration

Integrated on `feature/arun-rag-risk-compliance`, 21 November 2026, from `Downloads/BLACKWING_QUALITY_SCORER_V2_PROPOSAL.zip`.

## Independent results

**Prompt Completeness** measures only the information supplied at intake. It never includes generated report content. The supplied rubric weights are:

| Check | Points |
| --- | ---: |
| Incident narrative | 25 |
| Airport/location | 20 |
| Incident date | 20 |
| Jurisdiction | 15 |
| Source identification | 20 |

**Report Completeness** measures only the generated investigation documentation. It does not add the prompt score or reward incident severity/risk magnitude.

| Check | Points |
| --- | ---: |
| Seven specialist outputs | 8 |
| Executive summary and report sections | 12 |
| Current-case factual evidence | 12 |
| Factual reference IDs present | 13 |
| Generated-report factual points referenced | 10 |
| Human verification required | 8 |
| Unknowns/completeness disclosed | 7 |
| Regulatory applicability review recorded | 8 |
| Risk rationale or explicit abstention documented | 8 |
| Mitigation traceability | 9 |
| Critic review and issue handling | 5 |

These weights and the proposal's partial-credit rules are retained. Explicit mitigation deferral with documented unknowns can earn four points. Disclosed unresolved Critic issues with required human review can earn two points. Safe unassessed risk can earn documentation credit without inventing a rating.

The aviation risk assessment remains entirely separate: its existing FAA-reference categorical matrix, backend-calculated index/classification and required human verification are unchanged. The report rubric checks whether risk rationale or abstention was documented, not whether risk is high or low. Tests verify that changing an evidenced C3/medium assessment to A1/high without changing its documentation leaves completeness unchanged.

## Integrity and presentation

- No minimum of 70 is forced, and no score is an automatic acceptance decision.
- The proposed ceiling of 69 applies when no current-case documented facts are supplied.
- The proposed ceiling of 59 applies when documented facts lack reference IDs.
- Empty generated-report factual-point lists earn no citation-coverage credit.
- `raw_score` exposes the weighted total before any disclosed integrity ceiling; flags explain applied ceilings.
- Null/blank IDs are not counted as references. Both `evidence_ids` and `source_ids` are checked.
- Scores measure structural completeness and reference presence, not source entailment, factual accuracy, regulatory legality or operational safety.

Sunayana's nine-tab workstation, branding, finding review, reporting/PDF controls and original `build_evidence_checklist()` remain intact. Only the existing completeness panel was changed: it now presents **Prompt Completeness** and **Report Completeness** in two native Streamlit columns, with weights, earned points, missing-information actions and flags. When no completed report exists, the report metric is **Not available**, not a fabricated zero.

## Integration adaptations

The downloaded proposal used some field names that differ from the actual backend. The integrated function recognizes `compliance.overall_applicability`, `compliance.missing_context` and `risk.assessment_status`, in addition to the proposal's aliases. This avoids incorrectly withholding documentation credit from explicit backend applicability/abstention records. It does not claim those records prove legal applicability or sound risk inputs.

The original function signature and existing result fields are retained. Added result metadata is `raw_score` and `rubric_version`. Public FastAPI schemas, backend investigation behavior and persistence contracts were not changed.

## Tests against actual saved reports

The offline evaluator read existing API JSON exports; it did not generate, rewrite or re-investigate them. The SQLite lookup used URI `mode=ro`, not the repository constructor, so it cannot run interrupted-case recovery or modify run history.

| Saved report | Report Completeness | Prompt Completeness |
| --- | ---: | --- |
| `work/JFK_Investigation_Report_da0e8da7-7aea-4ade-a227-70f4900673ce.json` | **84/100** | Not available |
| `work/austin_report_local.json` | **58/100** | Not available |

`JFK_Agent_Diagnostics.json` duplicates the saved JFK export and was deduplicated rather than counted as another independent report.

The saved exports do not contain their intake case record, and the read-only lookup in `work/blackwing.sqlite3` found no matching cases. Their Prompt Completeness results were therefore left unavailable. Airport/date/narrative data was not reconstructed from generated report text. The UI calculates prompt completeness normally from its actual canonical case record.

Austin's export predates the generated-report artifact: it loses summary/section and generated-point citation credit, plus risk-documentation and mitigation-linkage credit; unresolved Critic issues receive only the disclosed partial credit. JFK's per-check breakdown is saved alongside Austin's in the evaluation artifact. Neither score is a factual-correctness or safety finding.

## Validation

**158 passed, zero failures, zero skips; two existing dependency deprecation warnings; 13.74 seconds.** All previous 141 cases remain included, plus 17 scorer cases.

- The eight tests supplied in the downloaded ZIP run against the integrated production function.
- Added tests cover prompt/report independence, backend field aliases, null/blank citation IDs, disclosed caps, unchanged scores across risk magnitudes, input immutability, read-only database evaluation, missing-intake handling and the two-result UI.
- Existing UI, report/PDF, risk, three-tool, Planner → Executor → Critic, diagnostic, abrupt-restart and persistence tests remain green.
- A real localhost Streamlit/FastAPI smoke with mocked model outputs displayed **Prompt Completeness 80%**, **Report Completeness 87%**, and the separate backend aviation-risk view. These are synthetic smoke values, not the saved JFK/Austin results.
- The running UI remained `ready`, retained its report/review/export controls, and produced no new Streamlit server warning or traceback. Temporary validation servers were stopped.
- The source evidence checklist was compared by AST with Sunayana's branch and is identical. Saved reports and all 12 RAG-index files passed unchanged SHA-256 checks. Risk, tools, planning, diagnostics and backend implementation files were not edited in this task.

Artifacts: `docs/validation/quality-v2/actual-report-results.json`, `regression.xml`, `running-app-summary.json`, `preservation.json`, `streamlit.log` and `validation-summary.txt`.

## Files changed

Modified:

- `blackwing/report_quality.py` — downloaded v2 rubric, documented aliases and citation-ID hygiene; original checklist preserved.
- `blackwing/copilot_ui.py` — independent result panels with weights, earned points and explicit unavailable state.
- `tests/test_streamlit_integration.py` — update the existing scorer criterion/metric assertions to the v2 names and partial credit.
- `README.md` — current scorer guide link.

Added:

- `tests/test_report_quality_v2.py` — proposal's eight tests plus nine integration/independence tests.
- `scripts/evaluate_report_completeness.py` — offline evaluator for saved exports, with optional read-only case lookup and duplicate detection.
- This guide and the six validation artifacts listed above.

## Reproduce

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\evaluate_report_completeness.py --report work\austin_report_local.json work\JFK_Investigation_Report_da0e8da7-7aea-4ade-a227-70f4900673ce.json --database work\blackwing.sqlite3 --output work\quality-v2-evaluation.json
```

If the original case database is elsewhere, supply that path to score the corresponding intake. Omitting `--database` is supported; prompt completeness then remains unavailable unless the export itself includes genuine incident metadata.

No Ollama/IIT calls, embedding generation, index rebuilds, protected-gold modifications, commits, merges or pushes were performed. The proposed rubric remains provisional and is not independently validated as a factual-quality metric.
