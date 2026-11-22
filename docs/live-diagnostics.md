# Live-investigation diagnostics and recovery visibility

Implemented on `feature/arun-rag-risk-compliance`, 21 November 2026. No model calls, index rebuilds, protected-gold changes, branch switches, merges, commits or pushes were performed.

## Findings from the existing execution path

FastAPI uses Starlette `BackgroundTasks` to invoke `_execute_run` after returning HTTP 202. Retrieval preflight occurs during run creation and agent-specific retrieval occurs in the background runner. The default runner initializes the HTTP client and LangGraph, plans the investigation, then executes the eight existing roles with at most one targeted correction.

Previously, the legacy agent trace row was appended **after** an agent returned. The incremental callback saved Planner/agent snapshots after those tasks completed, while the unfinished-run `/trace` endpoint exposed only tool events. No durable agent-start or model-wait marker existed. Therefore, two successful retrieval events with no agent rows **do not prove that the first agent never started**. The supplied `logs/jfk_interrupted_trace.json` indeed contains exactly two successful `retrieve_evidence` events and no agent events. It was read without modification.

The supplied trace and absence of a matching Windows Application event are insufficient to identify why the old process stopped. No crash cause is claimed. Existing JFK history is retained; historical starts/completions are not fabricated or backfilled. The new instrumentation applies to subsequent executions.

## Incremental lifecycle events

Every actual graph role execution, including deterministic Similar Case, unassessed Compliance and deterministic Report assembly, has a unique `execution_id`. A targeted correction gets a different execution ID for that repeated role. Lifecycle observations also have unique event IDs.

| Status | Meaning |
| --- | --- |
| `agent_started` | The Executor entered the role; event committed before executing the agent function |
| `waiting_for_model_response` | One physical provider request is about to be awaited, after prompt/token preflight; event committed before that await |
| `model_response_received` | A successful HTTP response arrived; schema/grounding checks may still fail |
| `model_request_failed` | Timeout, transport error, unsuccessful HTTP status or unexpected exception during the request |
| `agent_completed` | The complete role function, including output validation/grounding or disclosed deterministic fallback, returned successfully |
| `agent_failed` | The role raised an error; the failure is committed before propagation to the background boundary |
| `execution_interrupted` | Cooperative cancellation observed, or a previously active run discovered after restart; no success is inferred |

Workflow-level events distinguish background startup, workflow/client initialization, graph readiness and final completion/failure. A stop before the first agent can now be localized to the last recorded initialization phase.

Agent completion/failure and model response/failure record measured `duration_ms`. Each event includes UTC timestamp, case ID, role, configured provider/model, and execution identity. Provider attempts have request IDs and attempt numbers. The agent's execution mode is recorded so deterministic processing is distinguishable from model processing. Deterministic/unassessed roles do **not** emit model-wait events.

Model-wait is an observation of a request attempt, not a claim that the provider successfully received/completed it. `model_response_received` means HTTP succeeded, not that the model result passed validation. The existing bounded schema-repair and hosted transient retries remain unchanged; each physical hosted retry has its own request observation.

## Durability and API visibility

The default FastAPI runner passes an execution-event sink into `analyze_incident`. The sink synchronously calls the existing SQLite repository's `append_event`; its WAL/full-synchronous transaction commits before the model await. Events do not wait for the agent, investigation, report generation or HTTP response to finish.

The existing repository method signatures and records interface remain intact. SQLite tables and serialization are reused. Appending an event updates existing run progress and the stored current execution state in the same write-through operation. Partial output checkpoints are retained. Callers using the service as a standalone library must supply a durable sink if they need persistence outside FastAPI.

- Existing `GET /api/v1/runs/{run_id}` returns progress containing lifecycle states while the run remains `running`.
- Existing `/trace` exposes both tool and lifecycle events during running, failed, interrupted and completed runs. Events are deduplicated and presented by recorded timestamp, with regenerated sequence numbers.
- Existing `/audit` retains the full event stream and now includes `execution_state`, alongside the plan, partial outputs, step count and recovery guidance.
- Trace/report schemas add optional diagnostic fields (`event_id`, `execution_id`, `agent_key`, `provider`, `request_id`, `attempt`). Existing role summary rows and tool records are retained.

## Restart and controlled recovery

On reopen, SQLite still marks queued/running records `failed` with `error_code=interrupted`, preserving their outputs and history. It now appends a durable `execution_interrupted` observation correlated with the last unfinished agent execution when one was actually recorded. If no agent was observed, it records a workflow interruption without inventing an agent identity.

The interruption timestamp is the **recovery detection time**, not a claimed crash time. There is no fabricated completion or failure event for the stopped agent. Reopening the already-marked record does not append another interruption. A new case run is the controlled recovery action; automatic checkpoint resumption is not claimed.

Cancellation is also persisted as interruption before re-raising `CancelledError`, allowing the hosting server's cooperative shutdown to proceed normally.

## Error sanitation

`safe_error` allow-lists diagnostic codes, exception class, HTTP status where available, and up to five source-frame descriptors (basename, function, line number). Public messages come from a fixed catalog.

New lifecycle events and failure logs never copy exception text/arguments, request URLs, headers, response bodies, prompts or model outputs. Tests put credential sentinels into exceptions/provider responses and verify that neither trace nor failure logs contain them. Provider timeouts, unreachable providers, HTTP failures, output validation, token budgets, unexpected processing errors and process-exit attempts have distinct diagnostic codes.

Existing optional local model-validation diagnostics retain their prior behavior; this change does not enable raw response logging or add secrets to configuration. It removes raw exception detail from the service's Critic/Verification fallback warnings.

## Background failures and process termination

The entire background body is now inside `_execute_run`'s containment boundary, including the initial repository update, context preparation, graph/client initialization and final bookkeeping. Ordinary exceptions and `SystemExit` attempts become failed-run diagnostics. Failures while persisting the error are separately logged by operation/error type and do not escape as another unhandled background exception. If the database cannot be opened or written, durability cannot be guaranteed; the server log explicitly reports that limitation.

The graph also converts a `SystemExit` raised directly by an agent into `AgentExecutionError(process_exit_attempt)` **inside the graph task**, before asyncio can treat it as a process-exiting BaseException. The normal targeted-correction policy still handles correctable agent errors without repeating the full workflow.

Static inspection found no `sys.exit`, `os._exit`, process-kill or explicit abort in the active API → service → runtime path. `SystemExit` statements found in benchmark/prompt-baseline CLI entrypoints are not invoked by the investigation API.

The following can still stop a process and cannot be promised recoverable by Python task handlers:

- External process termination, closing the host/terminal, machine shutdown or forced Uvicorn/service shutdown.
- `KeyboardInterrupt`/normal host shutdown, which is intentionally not swallowed.
- Native-extension/interpreter faults, severe resource exhaustion, or `os._exit`/termination in code outside the controlled execution path.
- A process-exiting BaseException raised in an unrelated/uncontrolled asyncio task.

An ordinary investigation exception is not evidence of a whole-server crash. The real localhost smoke test confirms that the Uvicorn PID stayed alive after a mocked provider timeout, `/health` returned **200**, and a subsequent independent case creation returned **201**. It does not prove the cause of the historical JFK process stop.

## Streamlit HTTP 409 handling

The report endpoint keeps HTTP 409 and the existing `detail` string shape, but explains whether the investigation is still running, failed, interrupted, or has not completed.

The frontend continues to treat this expected response as an unavailable report, not a generic UI error. It derives a clear message from canonical run state:

- Running/queued: refresh status and inspect Run history; do not submit a duplicate run.
- Failed: inspect the saved failure trace, resolve the error and start a new run.
- Interrupted: saved traces/partial outputs remain; start a new controlled recovery run, with no checkpoint-resumption claim.
- No run: start an investigation to generate a preliminary report.

The workspace displays human-readable agent-start, model-wait, completion, failure and interruption labels from current run progress. Existing older-completed-report warnings are retained. No second case database, new workflow, inference request or automatic retry was added to the UI.

## Automated and running validation

**141 tests passed, zero failures, zero skips, two existing dependency deprecation warnings; 20.22 seconds.** This includes all previous 128 cases plus 13 focused cases.

Focused coverage:

1. Reads SQLite from a separate connection inside the MockTransport handler: agent-start and model-wait are already committed before the model returns.
2. Confirms all eight roles produce measured completion events and deterministic roles do not claim model requests.
3. Timeout, HTTP 503 and unexpected request exceptions persist sanitized model/agent/run failures; the API continues to accept requests.
4. A graph-agent `SystemExit` is contained without exposing its arguments.
5. Setup failures, failure-bookkeeping failures and lazy repository initialization errors do not escape the background boundary.
6. Cooperative cancellation records interruption and propagates cancellation.
7. Starts a separate Python process blocked in a mocked model request, forcibly terminates it without Python cleanup, and reopens SQLite. The same execution is marked interrupted, its committed start/wait events remain, no false completion/failure appears, reopening is idempotent and a new recovery run is accepted.
8. API and Streamlit checks cover running, queued, failed and interrupted report-409 states and the live model-wait label.

The forced-process test waits 100 ms after process termination before reopening SQLite mapped files in this Windows environment. Immediate reopening in the test harness exposed `SQLITE_IOERR_TRUNCATE`; the short process-cleanup interval resolved that test synchronization issue. Production SQLite configuration was not changed to hide or retry storage errors, and every durability assertion remains in place.

A running Streamlit agent-API smoke used a real localhost FastAPI/Uvicorn process, the actual graph/provider adapter, temporary SQLite and `httpx.MockTransport`. A simulated timeout produced the full agent-start → model-wait → request-failure → agent-failure → run-failure sequence. The process stayed alive, the report returned 409, Streamlit remained `ready` with the specific failed-report message, and another case could be created. Both temporary servers were stopped.

Raw evidence: `docs/validation/live-diagnostics/` contains the JUnit XML, running smoke summary, failure trace, restart-recovery example, preservation hashes, server logs and validation summary. No real model or production case database was used in validation.

## Preservation and modified files

All 12 saved RAG-index files were compared by SHA-256 before/after and are unchanged. Risk matrix/grounding/schema, executable tools, planning policy and RAG modules are unchanged. Existing regression coverage for three-tool execution, risk/mitigation, bounded Planner → Executor → Critic correction, persistence, UI report quality and exports remains green. Protected datasets and the supplied JFK trace were not modified.

Modified for this task:

- `blackwing/service.py` — scoped role lifecycle, execution sink, workflow initialization phases, contained agent exit attempts and sanitized fallback warnings.
- `blackwing/agents/runtime.py` — physical provider-request observations, including existing hosted retry attempts.
- `blackwing/api.py` — guarded background body/error persistence, execution callback, live chronological traces, state-specific report 409 and audit state.
- `blackwing/repository.py` — incremental progress/current state and correlated restart-interruption observations.
- `blackwing/schemas.py` — optional diagnostic trace fields.
- `blackwing/copilot.py`, `blackwing/copilot_state.py`, `blackwing/copilot_ui.py` — report availability context and human-readable execution labels.
- `tests/test_streamlit_integration.py` — retain the eight-role contract assertion while allowing additive diagnostic rows.
- `README.md` — link to this operational guide.

Added:

- `blackwing/diagnostics.py`.
- `tests/test_live_diagnostics.py`.
- This guide and `docs/validation/live-diagnostics/` evidence artifacts.

Re-run without model calls:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

For a future authorized live run, use the existing single-worker FastAPI configuration. Poll `/runs/{id}` and inspect `/trace` or `/audit`: an agent-start means the role entered; model-wait means a physical request is pending; completion/failure/interruption remain separate states. A process killed outside Python may only be classified as interrupted when the repository is next opened. No live JFK replay was performed for this task.
