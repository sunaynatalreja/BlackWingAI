"""BLACKWING AI - conservative, offline preflight for IIT Qwen3 shared gateway.

Assumptions based on institute/user-provided constraints (not tokenized with the
remote model's own tokenizer): one request/key; 60 requests/min/key; combined
prompt+completion <=8192; completion <=1024. The 5800 estimated-input target
leaves >=1368 tokens of headroom with a 1024 completion cap.

The estimate is deliberately pessimistic. It is NOT a precise Qwen token count.
The remote server's actual tokenization and chat-template overhead are authoritative.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

CONTEXT_LIMIT = 8192
MAX_OUTPUT = 1024
DEFAULT_INPUT_TARGET = 5800
DEFAULT_MAX_PROMPT_CHARS = 8200

# Eight possible subcalls (risk and mitigation have compact, expanded schemas), at most one at a time. The six report sections are
# assembled from these structured outputs, never by one gigantic report prompt.
STAGE_OUTPUT_TOKENS = {
    "chronology_early": 700,
    "chronology_late": 700,
    "rca": 1024,
    "comparison": 750,
    "compliance": 700,
    "risk": 950,
    "mitigation": 900,
    "verifier": 900,
    "smoke": 80,
}

@dataclass(frozen=True)
class BudgetConfig:
    context_limit: int = CONTEXT_LIMIT
    max_output: int = MAX_OUTPUT
    input_target: int = DEFAULT_INPUT_TARGET
    prompt_char_limit: int = DEFAULT_MAX_PROMPT_CHARS
    overhead_estimate: int = 256

    def __post_init__(self):
        if self.context_limit > CONTEXT_LIMIT or self.max_output > MAX_OUTPUT:
            raise ValueError("IIT caps must not be increased without verified permission")
        if self.input_target + self.max_output > self.context_limit - 512:
            raise ValueError("Keep at least 512 tokens of extra combined-context margin")
        if self.input_target < 1200 or self.max_output < 256:
            raise ValueError("Budget too small for meaningful analysis")

    @classmethod
    def from_env(cls):
        # Environment may TIGHTEN limits, never raise above institute constraints.
        return cls(
            context_limit=min(CONTEXT_LIMIT, int(os.getenv("BLACKWING_CONTEXT_LIMIT", CONTEXT_LIMIT))),
            max_output=min(MAX_OUTPUT, int(os.getenv("BLACKWING_MAX_OUTPUT_TOKENS", MAX_OUTPUT))),
            input_target=min(DEFAULT_INPUT_TARGET, int(os.getenv("BLACKWING_INPUT_TOKEN_TARGET", DEFAULT_INPUT_TARGET))),
            prompt_char_limit=min(DEFAULT_MAX_PROMPT_CHARS, int(os.getenv("BLACKWING_MAX_PROMPT_CHARS", DEFAULT_MAX_PROMPT_CHARS))),
        )


def conservative_estimate(system: str, user: str, overhead: int = 256) -> int:
    """Deliberate overestimate for typical text; no exact safety guarantee.

    UTF-8 byte-count / 1.5 is intentionally pessimistic for aviation English
    and guards against heavy symbols; code must never label this 'true tokens'.
    """
    data = (str(system) + "\n" + str(user))
    return int(max(math.ceil(len(data) / 2.0), math.ceil(len(data.encode('utf-8')) / 1.5)) + overhead)


def preflight(stage: str, system: str, user: str, max_tokens: int, config: BudgetConfig | None = None) -> dict:
    config = config or BudgetConfig.from_env()
    estimate = conservative_estimate(system, user, config.overhead_estimate)
    max_tokens = int(max_tokens)
    if max_tokens < 1 or max_tokens > config.max_output:
        raise ValueError(f"{stage}: requested {max_tokens} output tokens; permitted 1-{config.max_output}")
    if stage != "smoke" and max_tokens > STAGE_OUTPUT_TOKENS.get(stage, config.max_output):
        raise ValueError(f"{stage}: output above configured per-stage cap")
    guard = min(config.input_target, config.context_limit - max_tokens - 512)
    n_chars = len(system) + len(user)
    byte_reserve = len((system + "\n" + user).encode("utf-8")) + config.overhead_estimate + max_tokens
    allowed = (estimate <= guard and n_chars <= config.prompt_char_limit
               and byte_reserve <= config.context_limit)
    return {
        "stage": stage,
        "est_input_tokens_conservative_not_exact": estimate,
        "requested_max_output_tokens": max_tokens,
        "estimated_upper_total": estimate + max_tokens,
        "utf8_bytes_plus_output_and_template_reserve": byte_reserve,
        "remote_combined_limit": config.context_limit,
        "safe_input_target": guard,
        "prompt_characters": n_chars,
        "max_prompt_characters": config.prompt_char_limit,
        "status": "PREFLIGHT_PASS" if allowed else "PREFLIGHT_BLOCKED",
        "token_count_kind": "CONSERVATIVE_ESTIMATE_NOT_MODEL_TOKENIZER",
    }


def assert_preflight(stage: str, system: str, user: str, max_tokens: int,
                     config: BudgetConfig | None = None) -> dict:
    result = preflight(stage, system, user, max_tokens, config)
    if result["status"] != "PREFLIGHT_PASS":
        raise ValueError(f"Context preflight blocked {stage}: {result}")
    return result
