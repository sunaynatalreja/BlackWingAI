"""Shared Copilot inference with opt-in hosted use and metadata-only accounting."""
import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing
from urllib.parse import urlparse

from filelock import FileLock
from jsonschema import Draft202012Validator

from blackwing_app import OUTPUT, SharedGateway
from token_budget import BudgetConfig, assert_preflight

PROVIDERS = {"local": "Local Ollama", "iit": "IIT Hosted"}
TRANSIENT_STATUS = {429, 500, 502, 503, 504}


class InferenceError(RuntimeError):
    """A provider failure whose message is safe for investigators and traces."""

    def __init__(self, message, *, retryable=False):
        super().__init__(message)
        self.retryable = retryable


def provider_name(provider=None):
    name = (provider or os.getenv("BLACKWING_INFERENCE_PROVIDER", "local")).strip().lower()
    if name not in PROVIDERS:
        raise ValueError("Select local or iit inference.")
    return name


def _configuration(provider):
    if provider == "local":
        return (
            os.getenv("BLACKWING_OLLAMA_MODEL", "qwen3:8b").strip(),
            os.getenv("BLACKWING_OLLAMA_URL", "http://127.0.0.1:11434").strip(),
        )
    return os.getenv("BLACKWING_MODEL", "").strip(), os.getenv("BLACKWING_API_BASE_URL", "").strip()


def _fingerprint(provider):
    model, endpoint = _configuration(provider)
    return hashlib.sha256(json.dumps([provider, model, endpoint.rstrip("/")]).encode("utf-8")).hexdigest()


def _stats_connection():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(OUTPUT / "inference_status.sqlite3", timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY, config TEXT NOT NULL, received INTEGER NOT NULL,
            completed INTEGER NOT NULL, total_tokens INTEGER, quota INTEGER
        )
    """)
    connection.execute("CREATE INDEX IF NOT EXISTS inference_requests_config ON requests(config, id)")
    return connection


def _record(entry):
    if not entry["request_sent"]:
        return
    try:
        with closing(_stats_connection()) as connection, connection:
            status = entry.get("http_status")
            connection.execute(
                "INSERT INTO requests(config,received,completed,total_tokens,quota) VALUES (?,?,?,?,?)",
                (_fingerprint(entry["provider"]), int(entry["response_received"]),
                 int(entry["response_received"] and (status is None or status < 400)),
                 entry.get("total_tokens"), entry.get("remaining_quota")),
            )
    except (OSError, sqlite3.Error):
        entry["status_write_failed"] = True


def inference_status(provider=None):
    """Return accounting for the selected configuration without contacting its API."""
    name = provider_name(provider)
    model, endpoint = _configuration(name)
    result = {
        "provider": name, "provider_label": PROVIDERS[name], "model": model or "Not configured",
        "api_status": "Not checked" if model and endpoint and (
            name == "local" or os.getenv("BLACKWING_API_KEY", "").strip()
        ) else "Not configured",
        "tokens_used": None, "requests_completed": 0, "requests_attempted": 0,
    }
    try:
        with closing(_stats_connection()) as connection:
            signature = _fingerprint(name)
            counts = connection.execute("""
                SELECT COUNT(*) AS attempts, SUM(completed) AS completions, SUM(total_tokens) AS tokens
                FROM requests WHERE config = ?
            """, (signature,)).fetchone()
            latest = connection.execute("""
                SELECT received, completed, quota FROM requests WHERE config = ? ORDER BY id DESC LIMIT 1
            """, (signature,)).fetchone()
    except (OSError, sqlite3.Error):
        return result
    result.update(requests_attempted=counts["attempts"], requests_completed=counts["completions"] or 0,
                  tokens_used=counts["tokens"])
    if result["api_status"] == "Not configured":
        return result
    if name == "iit" and (
        os.getenv("BLACKWING_ALLOW_REMOTE", "") != "YES"
        or os.getenv("BLACKWING_TEAM_KEY_COORDINATED", "") != "YES"
    ):
        result["api_status"] = "Disabled"
    elif latest:
        result["api_status"] = ("Connected" if latest["completed"] else
                                "Request rejected" if latest["received"] else "Disconnected")
        if latest["quota"] is not None:
            result["remaining_quota"] = latest["quota"]
    return result


def _trace(trace, provider, **details):
    entry = {"step": "Model", "provider": provider, "model": _configuration(provider)[0],
             "request_sent": False, "response_received": False, "prompt_tokens": None,
             "completion_tokens": None, "total_tokens": None, "latency_seconds": 0.0, **details}
    _record(entry)
    if trace is not None:
        trace.append(entry)
    return entry


def _authorized_iit(authorized):
    if not authorized or os.getenv("BLACKWING_ALLOW_REMOTE", "").upper() != "YES":
        raise InferenceError("IIT hosted inference requires explicit authorization for this operation.")
    if os.getenv("BLACKWING_TEAM_KEY_COORDINATED", "").upper() != "YES":
        raise InferenceError("IIT hosted inference requires the single shared team backend.")
    model, endpoint = _configuration("iit")
    parsed = urlparse(endpoint)
    key = os.getenv("BLACKWING_API_KEY", "").strip()
    if not model or not key or any("REPLACE" in value.upper() for value in (model, key, endpoint)):
        raise InferenceError("Configure the IIT endpoint, model identifier and API key.")
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise InferenceError("Configure a verified HTTPS IIT API base URL without embedded credentials.")
    format_name = os.getenv("BLACKWING_IIT_RESPONSE_FORMAT", "json_schema").strip().lower()
    if format_name not in {"json_schema", "json_object"}:
        raise InferenceError("Select json_schema or json_object for IIT response format.")
    return format_name


def _schema_validator(schema):
    validator = Draft202012Validator(schema)

    def valid(raw):
        try:
            return validator.is_valid(json.loads(raw))
        except (TypeError, ValueError):
            return False

    return valid


def _local_attempt(system, prompt, cap, schema, cache, valid_cache, trace, transport):
    if transport is None:
        from review_workflow import local_call
        transport = local_call
    if not cache:
        return _local_request(system, prompt, cap, schema, trace, transport)

    cache_key = hashlib.sha256(json.dumps(
        [_fingerprint("local"), system, prompt, cap, schema], ensure_ascii=False, sort_keys=True
    ).encode("utf-8")).hexdigest()
    directory = OUTPUT / "copilot_local_cache"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{cache_key}.json"
    with FileLock(str(path.with_suffix(".lock")), timeout=620):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if valid_cache(raw):
                _trace(trace, "local", status="CACHE_HIT_NO_API_CALL", prompt_tokens=0,
                       completion_tokens=0, total_tokens=0)
                return raw
        except (OSError, ValueError, TypeError):
            pass
        raw = _local_request(system, prompt, cap, schema, trace, transport)
        if valid_cache(raw):
            try:
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
                tmp.replace(path)
            except OSError:
                if trace is not None:
                    trace[-1]["cache_write_failed"] = True
        return raw


def _local_request(system, prompt, cap, schema, trace, transport):
    started = time.monotonic()
    try:
        raw, log = transport(system, prompt, cap, schema)
    except Exception as exc:
        code = getattr(exc, "code", None)
        _trace(trace, "local", status="error", error_type=type(exc).__name__,
               request_sent=getattr(exc, "blackwing_request_attempted", False), response_received=code is not None,
               http_status=code, latency_seconds=round(time.monotonic() - started, 2))
        raise InferenceError("Local Ollama request failed.", retryable=code in TRANSIENT_STATUS) from None
    _trace(trace, "local", **{"request_sent": True, "response_received": True,
                            "latency_seconds": round(time.monotonic() - started, 2), **log})
    if log["status"] != "ok":
        raise InferenceError("Local Ollama returned a truncated or over-budget response.")
    return raw


def model_call(system, prompt, cap, schema, *, provider=None, authorized=False, cache=True,
               trace=None, request_limit=1, local_transport=None):
    """Send one bounded, schema-constrained model request and append sanitized diagnostics."""
    name = provider_name(provider)
    if request_limit != 1:
        raise ValueError("Each inference call uses one request. The investigation planner owns retries.")
    try:
        format_name = _authorized_iit(authorized) if name == "iit" else None
        assert_preflight("copilot_question", system, prompt, cap)
    except (ValueError, InferenceError) as exc:
        _trace(trace, name, status="blocked", error_type=type(exc).__name__)
        raise InferenceError(str(exc) if isinstance(exc, InferenceError) else "The model input exceeds the token budget.") from None
    cache_enabled = cache and os.getenv("BLACKWING_USE_RESPONSE_CACHE", "YES") == "YES"
    valid_cache = _schema_validator(schema)
    if name == "local":
        try:
            return _local_attempt(system, prompt, cap, schema, cache_enabled, valid_cache, trace, local_transport)
        except InferenceError:
            raise
        except Exception as exc:
            _trace(trace, name, status="error", error_type=type(exc).__name__)
            raise InferenceError("Local inference setup or cache access failed.") from None
    gateway = None
    try:
        gateway = SharedGateway(authorize_hosted=authorized)
        return gateway.call("copilot_question", system, prompt, cap, schema=schema,
                            response_format=format_name, cache=cache_enabled, retry_limit=0,
                            cache_validator=valid_cache)
    except Exception as exc:
        if gateway is None or not gateway.logs:
            _trace(trace, name, status="error", error_type=type(exc).__name__)
        code = getattr(exc, "status_code", None)
        raise InferenceError("IIT hosted inference could not complete the request.",
                             retryable=code in TRANSIENT_STATUS) from None
    finally:
        if gateway is not None:
            for entry in gateway.logs:
                _trace(trace, name, **entry)
            try:
                gateway.close()
            except Exception:
                # A close error cannot invalidate an already recorded request result.
                pass
