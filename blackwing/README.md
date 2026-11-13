# BLACKWING AI Copilot

Investigators can open a case, ask focused questions, review the cited passages, and accept or reject proposed findings. The Hazard Register and Mitigation Action Plan preserve the relationship between a hazard and its proposed controls. The preliminary PDF assembles the same reviewed findings into six sections without another model call.

## Start

In this directory, with Python 3.12 or 3.13 and Ollama available:

```powershell
python -m pip install -r requirements.txt
ollama list
python -m streamlit run ui.py --global.developmentMode=false --server.address=127.0.0.1 --server.port=8501
```

The default model is `qwen3:8b`. Hybrid retrieval uses BM25, MiniLM, rank fusion and a cross-encoder reranker. First use downloads the retrieval weights if they are absent. BM25 works independently of those weights. CPU-only inference can take several minutes. Keep the app and its SQLite state on a trusted workstation.

## Investigate and review

1. Choose **Start New Investigation**. Enter the airport, country or jurisdiction, date, aircraft details and narrative. Attach supporting PDFs or text files with their provenance. **Explore Historical Incidents** opens the indexed library and saved investigations.
2. Ask a question or follow-up in **Q&A**. The planner selects up to two relevant specialties and the executor chooses the matching retrieval scopes and tools. Use the workspace tabs for chronology, comparisons, guidance, hazards, risk, and evidence gaps.
3. In **Evidence and verification**, inspect the source excerpts and accept or reject findings. Accept a hazard before requesting a linked mitigation. The register connects its evidence, potential consequence, existing controls, risk screen and mitigations.
4. Open **Report** to export the six sections from accepted findings. Check chronology times and time zones before sharing. **Run history** shows the tools, model requests, usage, latency and critic outcomes.

Case evidence, reported observations, candidate regulations and historical analogs have separate retrieval scopes. Verified findings and recent turns inform follow-ups. The critic resolves source quotes, checks scope and aircraft identifiers, and flags unsupported numeric claims. A human reviewer decides whether the cited passage supports the finding.

## JFK, Austin, and Goa

JFK is the 10-question development case. Austin is the 15-question validation case. Goa is the 15-question independent final test. Goa remains reserved in the interactive UI. The test requires independently reviewed factual evidence, any approved candidate Indian guidance, an approved offline gold digest and a frozen evaluation manifest. Keep AAIB conclusions and answer keys outside retrieval. The broader 80-question export includes Delhi and synthetic cases.

See [the evaluation protocol](evaluation/COPILOT_EVALUATION.md) and run:

```powershell
python evaluate_copilot.py plan
python evaluate_copilot.py run --case jfk
python evaluate_copilot.py run --case austin
```

Running a split records output and trace data. An independent scorer must compare the answers and citations with approved gold offline. The pilot contains 424 chunks from 55 records. Use `python index_pilot.py --source-root <BLACKWING_AI_REORG> --out <candidate.jsonl> --scale --official-asrs-only` to build the larger candidate index for eligibility and leakage review. The app keeps using the pilot index until a reviewed replacement is installed. The original report-first CLI and retrieval experiments remain available.

## Inference providers and limits

Local Ollama is the default. Both providers use the same workflow, citations and structured response schema. Set configuration through environment variables or a private `.env` beside `blackwing_app.py`. [The example file](.env.example) lists the variables.

To use the IIT-approved Qwen3 8B deployment, configure `BLACKWING_API_BASE_URL`, `BLACKWING_MODEL`, `BLACKWING_API_KEY`, `BLACKWING_ALLOW_REMOTE=YES`, and `BLACKWING_TEAM_KEY_COORDINATED=YES`. Use the institute's HTTPS endpoint and its trusted CA bundle if supplied. The team should route all requests using its key through a single backend with a shared output directory. The process lock coordinates instances on that backend. IIT controls overall shared-VM capacity.

The Streamlit **Inference status** panel reads local accounting and uses the last observed request result. Selecting a provider makes no network call. Each hosted turn requires a separate confirmation to send the question and relevant case excerpts. For a scheduled evaluation, choose `--provider iit` and explicitly authorize that run using `--authorize-hosted`. The quota field appears only when a configured IIT response header supplies a numeric value.

Calls use an 8,192-token combined budget and reserve 512 output tokens by default, up to 1,024. The request lock allows one in-flight call and spaces request starts by at least 1.1 seconds. The workflow caps the trace at 10 steps and allows at most two model attempts, including retry or repair. Schema-validated responses can be cached. Baselines and evaluation runs use fresh requests. Actual token usage is shown when reported by the provider. Client token preflight is conservative and the provider enforces its own context limit.

Uploads and narratives are treated as untrusted data. Emails and common formatted phone numbers are masked. Review uploads for additional personal details before submission. The local state, caches and traces are stored under `output/`. Set `BLACKWING_OTLP_ENDPOINT` to an approved OTLP HTTP collector for metadata-only spans.

## Streamlit Community Cloud

Use the [Community Cloud deployment guide](cloud/DEPLOYMENT.md) for the hosted profile and setup.

## Container

```powershell
docker build -t blackwing-review .
docker run --rm -p 127.0.0.1:8501:8501 -v blackwing-runs:/app/output blackwing-review
```

Docker Desktop uses `host.docker.internal` to reach the host's Ollama service. On Linux configure the host gateway alias and Ollama bind address for container access.



See the [ten-minute demo](DEMO.md) for the presentation sequence.
