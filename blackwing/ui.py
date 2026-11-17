from __future__ import annotations

import time

import httpx
import streamlit as st
from blackwing.config import Settings

st.set_page_config(page_title="BlackWing AI", page_icon="✈️", layout="wide")
st.title("BlackWing AI")
st.caption("Runway incursion investigation workspace · Preliminary decision support")
st.warning("Model-generated content is preliminary. Verify evidence and obtain qualified human review before relying on findings or mitigation options.")

with st.form("incident_form"):
    narrative = st.text_area("Incident narrative", height=220, placeholder="Describe the event and include any known date, airport, aircraft, and source references.")
    submitted = st.form_submit_button("Analyze narrative", type="primary")

if submitted:
    if len(narrative.strip()) < 20:
        st.error("Enter at least 20 characters of incident narrative.")
    else:
        try:
            settings = Settings.load()
            with httpx.Client(timeout=settings.ui_request_timeout_seconds) as client:
                case_response = client.post(f"{settings.ui_api_url}/api/v1/cases", json={"category": "runway_incursion", "narrative": narrative})
                case_response.raise_for_status()
                case = case_response.json()
                with st.spinner("Running the runway incursion workflow through the local model…"):
                    run_response = client.post(f"{settings.ui_api_url}/api/v1/cases/{case['case_id']}/runs", json={})
                    run_response.raise_for_status()
                    run = run_response.json()
                    deadline = time.monotonic() + settings.ui_request_timeout_seconds
                    poll_errors = 0
                    while run["status"] in {"queued", "running"} and time.monotonic() < deadline:
                        time.sleep(2)
                        try:
                            status_response = client.get(f"{settings.ui_api_url}/api/v1/runs/{run['run_id']}")
                            status_response.raise_for_status()
                            run = status_response.json()
                            poll_errors = 0
                        except httpx.RequestError as exc:
                            # A dropped local connection should not hide a run
                            # that the API continues processing in the background.
                            poll_errors += 1
                            st.warning(
                                "The connection to the API dropped while checking the run "
                                f"(attempt {poll_errors}); reconnecting…"
                            )
                            time.sleep(min(2 * poll_errors, 10))
            st.subheader("Analysis run")
            st.caption(f"Case ID: {case['case_id']} · Run ID: {run['run_id']} · Status: {run['status']}")
            if run["status"] == "completed":
                report_response = httpx.get(f"{settings.ui_api_url}/api/v1/cases/{case['case_id']}/report", timeout=settings.ui_request_timeout_seconds)
                report_response.raise_for_status()
                report = report_response.json()
                critic_output = report["report"]["agent_outputs"].get("critic", {})
                critic_issues = critic_output.get("issues", [])
                if any(issue.get("issue_type") in {
                    "critic_model_output_invalid", "critic_model_context_budget_exceeded"
                } for issue in critic_issues):
                    if any(issue.get("issue_type") == "critic_model_context_budget_exceeded"
                           for issue in critic_issues):
                        st.warning("The Critic input exceeded the safe model context budget, so no Critic response was produced. A deterministic audit was used; check the API service log for details.")
                    else:
                        st.warning("The Critic model response did not match its schema. A deterministic audit was used; check the API service log for validation details.")
                for agent_name, agent_output in report["report"]["agent_outputs"].items():
                    with st.expander(
                        agent_name.replace("_", " ").title(),
                        expanded=True,
                    ):
                        if agent_name == "similar_case" and agent_output.get("case_status", "").startswith("unseen_case"):
                            st.info("Incoming case is unseen in the currently indexed reports. The retrieved cases below are historical analogies, not findings about this case.")
                        st.json(agent_output)
                trace_response = httpx.get(f"{settings.ui_api_url}/api/v1/runs/{run['run_id']}/trace", timeout=settings.ui_request_timeout_seconds)
                trace_response.raise_for_status()
                with st.expander("Execution trace and limitations", expanded=True):
                    st.json({"trace": trace_response.json(), "limitations": report["report"]["limitations"]})
            elif run["status"] == "failed":
                st.error(run.get("error") or "The analysis run failed.")
            else:
                st.info(f"Run is still {run['status']}. Poll GET /api/v1/runs/{run['run_id']} for its status.")
        except httpx.ConnectError as exc:
            st.error(f"Could not connect to the BlackWing API at {settings.ui_api_url}. Check that its server is running and that the UI API URL matches its host and port. ({exc})")
        except httpx.TimeoutException:
            st.error("The BlackWing API request timed out. Check the API run status and Ollama model response time.")
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", "")
            except ValueError:
                detail = ""
            st.error(f"The BlackWing API returned HTTP {exc.response.status_code}. {detail}")
        except httpx.HTTPError as exc:
            st.error(f"The BlackWing API request failed: {exc}")
        except (ValueError, OSError) as exc:
            # Configuration exceptions identify property keys/environment variable names only;
            # they never include resolved values, so showing the message does not expose secrets.
            st.error(f"Could not load application settings: {exc}")
