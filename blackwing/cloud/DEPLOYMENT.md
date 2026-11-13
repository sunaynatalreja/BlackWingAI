# BLACKWING on Streamlit Community Cloud

## Deploy the team demonstration

Use the repository owner's Streamlit workspace and a GitHub account with deployment access. Once the reviewed changes are on GitHub, create an app with these settings.

| Setting | Value |
|---|---|
| Repository | `sunaynatalreja/BlackWingAI` |
| Branch | `feature/blackwing-agentic-copilot` |
| Main file path | `blackwing/cloud/streamlit_app.py` |
| Python | 3.13 |
| App access | Private, with approved team viewers |

Community Cloud starts the app at the repository root and installs `blackwing/cloud/requirements.txt`, which sits beside the entry point. The app imports the existing Copilot and loads its example cases, source registry, JSONL indexes and glossary from `blackwing/`. The Cloud profile uses BM25 and omits PyTorch, the embedding models and Ollama. The workstation entry point remains `blackwing/ui.py` and supports hybrid retrieval and local inference. [Streamlit deployment setup](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) · [Dependency file locations](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) · [Private app access](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)

Start in browse-only mode while access and the endpoint are reviewed. Set the app to private and verify viewer access before enabling inference with a team key. The inference panel shows whether IIT is configured and displays the last request result. Risk screening, evidence browsing and report assembly do not call a model.

## Configure IIT securely

Only after IIT authorizes this deployment and the team designates it as the single backend for the team key, enter these **top-level strings** in Community Cloud **Advanced settings → Secrets**, or **App settings → Secrets** after creation. Fill in the institute-issued values in that console. Top-level secrets are also available as environment variables. [Streamlit Secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) · [Top-level environment variables](https://docs.streamlit.io/develop/concepts/connections/secrets-management)

```toml
BLACKWING_API_BASE_URL = "<approved HTTPS base URL including the API version path>"
BLACKWING_MODEL = "<exact IIT Qwen3 8B model ID>"
BLACKWING_API_KEY = "<team key>"
BLACKWING_ALLOW_REMOTE = "YES"
BLACKWING_TEAM_KEY_COORDINATED = "YES"
BLACKWING_IIT_RESPONSE_FORMAT = "json_schema"
```

Use the base URL for which the gateway exposes `/chat/completions` under the version path. Use `json_object` only if IIT confirms that is the supported response format. The app validates both formats against its response schema. Every hosted question still requires the user to authorize sending the question and relevant excerpts. The gateway's file lock and pacing serialize requests within this deployment. A single instance or an IIT-approved proxy must coordinate team-key traffic if the team uses multiple runtimes.

If IIT requires a VPN, an institute-only network, an IP allowlist or a private trust chain, confirm whether the Community Cloud runtime can reach and verify the endpoint. Follow IIT's approved networking and certificate process or use an IIT-approved reachable gateway. Leave TLS verification enabled and keep credential and certificate files out of Git. Community Cloud publishes its outbound addresses and notes that they can change. It hosts apps in the United States, which IIT must consider when approving evidence transfers. [Community Cloud networking and hosting](https://docs.streamlit.io/deploy/streamlit-community-cloud/status)

## Data and evaluation boundaries

The Cloud profile is for approved public-source or synthetic demonstration material. New case state, uploads, findings and traces use a temporary database scoped to the current browser session. Download reviewed reports before leaving. Use a durable, access-controlled store for persistent investigations. Community Cloud does not guarantee local file persistence. [Streamlit storage limitations](https://docs.streamlit.io/develop/concepts/connections/connecting-to-data)

The independent Goa intake is reserved for the evaluation workflow. Keep the protected packet, answer keys and final reports offline. The bundled text indexes are the existing pilot corpus. Hybrid retrieval and a scored IIT evaluation belong to the separately controlled evaluation setup.

## Run the same entry point locally

From the repository root in a clean Python 3.13 environment:

```powershell
python -m pip install -r blackwing/cloud/requirements.txt
python -m pip check
python -m unittest discover -s blackwing -p "test_*.py"
python -m streamlit run blackwing/cloud/streamlit_app.py --server.headless=true
```

Without IIT secrets, open a historical case and verify its evidence view and deterministic risk screen. The automated suite exercises the hosted request and preliminary PDF with a simulated IIT response. A real connection check requires approval and should use one public-source question after the app is private and the endpoint is configured.
