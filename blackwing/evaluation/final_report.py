"""Generic section-level comparison against an evaluator-only gold directory.

Gold reports are discovered by case metadata and never enter generation, case
evidence, or the shared RAG index.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from pypdf import PdfReader

from blackwing.agents.json_output import parse_agent_json
from blackwing.agents.runtime import _hosted_post
from blackwing.rag.embeddings import IITBGE, normalize
from blackwing.rag.tokenization import BGE_QUERY_PREFIX, BGESubwordTokenizer, chunk_document

logger = logging.getLogger("blackwing.final_report_evaluation")

# These aliases describe BlackWing's report fields, not any particular case or
# gold document. A heading match narrows the reference text; if a report uses
# different headings, BGE retrieves the most relevant passages from that PDF.
SECTION_ALIASES = {
    "overview": ("executive summary", "overview", "abstract", "incident summary", "history of flight", "incident description"),
    "findings_and_causes": ("analysis", "findings", "conclusions", "probable cause", "causal factors", "cause"),
    "comparisons_and_compliance": ("comparisons", "similar incidents", "compliance", "regulations", "regulatory framework", "procedures", "airport information", "organizational information", "safety issues"),
    "risk_and_follow_up": ("recommendations", "safety recommendations", "safety actions", "postincident actions", "post-incident actions", "mitigations", "corrective actions", "follow-up"),
}

_NUMBERED_HEADING = re.compile(r"^\s*((?:\d+\.)*\d+)\.?\s+(.{2,120}?)\s*$")
_APPENDIX_HEADING = re.compile(r"^\s*(Appendix\s+[A-Z](?:\s*[:.-]\s*.*)?)\s*$", re.I)
_REFERENCE_BOUNDARIES = {"contents", "table of contents", "references", "board member statements", "abbreviations", "figures", "tables"}


class GoldReferenceUnavailable(ValueError):
    """No unambiguous report in the configured evaluator-only directory."""


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


def _case_source_id(case: dict[str, Any]) -> str:
    value = str(case.get("source_incident_id") or "").strip()
    if value:
        return value
    narrative = str(case.get("narrative") or "")
    # Generic source-ID fallback; this does not assume a particular authority.
    match = re.search(r"\b[A-Z]{2,}[A-Z0-9-]*\d[A-Z0-9-]*\b", narrative, re.I)
    return match.group(0) if match else ""


def _date_strings(value: Any) -> list[str]:
    if not value:
        return []
    try:
        parsed = value if isinstance(value, date) else date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return [str(value).casefold()]
    return [
        parsed.isoformat().casefold(),
        f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}".casefold(),
        parsed.strftime("%B %d, %Y").casefold(),
        parsed.strftime("%d %B %Y").casefold(),
    ]


def _preview_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception:
            unlocked = 0
        if not unlocked:
            return ""
    return "\n".join((page.extract_text() or "") for page in reader.pages[:5]).casefold()


def find_gold_pdf(gold_directory: str, case: dict[str, Any]) -> Path:
    root = Path(gold_directory).expanduser().resolve()
    if not root.is_dir():
        raise GoldReferenceUnavailable("Configured gold-report directory is unavailable")
    candidates = sorted(path for path in root.rglob("*.pdf") if path.is_file())
    if not candidates:
        raise GoldReferenceUnavailable("No PDF gold reports were found under the configured directory")

    source_id = _case_source_id(case)
    source_key = _normalized(source_id)
    if source_key:
        direct = [path for path in candidates if source_key in _normalized(path)]
        if len(direct) == 1:
            return direct[0]
        if len(direct) > 1:
            raise GoldReferenceUnavailable("More than one gold PDF matches this source incident ID")
        by_content = [path for path in candidates if source_key in _normalized(_preview_text(path))]
        if len(by_content) == 1:
            return by_content[0]
        if len(by_content) > 1:
            raise GoldReferenceUnavailable("More than one gold PDF contains this source incident ID")

    # A metadata fallback supports gold files whose folders omit source IDs.
    dates = _date_strings(case.get("incident_date"))
    location_tokens = [token for token in re.findall(r"[a-z0-9]+", str(case.get("location") or "").casefold())
                       if len(token) > 3 and token not in {"airport", "international", "field", "united", "states"}]
    if dates and location_tokens:
        ranked: list[tuple[int, Path]] = []
        for path in candidates:
            preview = _preview_text(path)
            date_hit = any(value in preview for value in dates)
            location_hits = sum(token in preview for token in location_tokens)
            if date_hit and location_hits:
                ranked.append((location_hits, path))
        ranked.sort(key=lambda row: row[0], reverse=True)
        if ranked and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]):
            return ranked[0][1]
    raise GoldReferenceUnavailable("No unique gold PDF matched the case ID or its date and location")


def _heading_records(text: str) -> list[tuple[int, str, str | None]]:
    records = []
    unnumbered = {alias for aliases in SECTION_ALIASES.values() for alias in aliases} | _REFERENCE_BOUNDARIES
    offset = 0
    for line in text.splitlines(keepends=True):
        clean = line.strip()
        match = _NUMBERED_HEADING.match(clean)
        appendix = _APPENDIX_HEADING.match(clean)
        normalized = " ".join(re.sub(r"[^a-z0-9 ]", " ", clean.casefold()).split())
        if match:
            records.append((offset, match.group(2).strip(), match.group(1).rstrip(".")))
        elif appendix:
            records.append((offset, appendix.group(1).strip(), None))
        elif normalized in unnumbered:
            records.append((offset, clean, None))
        offset += len(line)
    return records


def _body_text(pages: list[str]) -> str:
    """Drop cover and contents pages when the report has a recognizable body start."""
    body_start = None
    body_start_pattern = re.compile(
        r"^\s*(?:Executive Summary|Abstract|Summary|Factual Information|Incident Description|"
        r"Occurrence Description|Background|Introduction|History of Flight)\s*$", re.I | re.M)
    numbered_start_pattern = re.compile(r"^\s*1\.?\s+(?:Factual|Incident|Background|Introduction|History)\b", re.I | re.M)
    for index, page in enumerate(pages[2:], start=2):
        if body_start_pattern.search(page) or numbered_start_pattern.search(page):
            body_start = index
            break
    if body_start is None:
        contents_pages = [i for i, page in enumerate(pages) if re.search(r"^\s*(?:Table of )?Contents\s*$", page, re.I | re.M)]
        body_start = min(max(contents_pages) + 1, len(pages) - 1) if contents_pages else 0
    return "\n\n".join(page for page in pages[body_start:] if page.strip())


def _matches_alias(title: str, aliases: tuple[str, ...]) -> bool:
    normalized_title = re.sub(r"[^a-z0-9 ]", " ", title.casefold())
    normalized_title = " ".join(normalized_title.split())
    return any(alias in normalized_title for alias in aliases)


def _matched_heading_sections(text: str, report_sections: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    records = _heading_records(text)
    extracted: dict[str, str] = {}
    mappings: dict[str, str] = {}
    for section in report_sections:
        section_id, title = section["task_id"], section["title"]
        aliases = SECTION_ALIASES.get(section_id, ())
        matches = []
        for index, (start, heading, number) in enumerate(records):
            if _matches_alias(heading, aliases):
                end = len(text)
                for next_start, _, next_number in records[index + 1:]:
                    if number is None or next_number is None:
                        end = next_start
                        break
                    if len(next_number.split(".")) <= len(number.split(".")):
                        end = next_start
                        break
                block = text[start:end].strip()
                if block:
                    matches.append((heading, block))
        if matches:
            extracted[section_id] = "\n\n".join(block for _, block in matches)
            mappings[section_id] = "; ".join(dict.fromkeys(heading for heading, _ in matches))
        else:
            mappings[section_id] = "Semantic selection from the matched gold report"
    return extracted, mappings


def _report_sections(report: dict[str, Any]) -> list[dict[str, str]]:
    generated = report.get("generated_report") or {}
    sections = []
    for section in generated.get("sections") or []:
        section_id = str(section.get("task_id") or "").strip()
        title = str(section.get("title") or section_id or "Report section").strip()
        parts = [str(section.get("summary") or "").strip()]
        parts.extend(str(point.get("text") or "").strip()
                     for point in section.get("key_points") or [] if isinstance(point, dict))
        value = "\n".join(part for part in parts if part)
        if value:
            sections.append({"task_id": section_id, "title": title, "text": value})
    return sections


def load_reference_for_case(gold_directory: str, case: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    pdf_path = find_gold_pdf(gold_directory, case)
    reader = PdfReader(str(pdf_path))
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception:
            unlocked = 0
        if not unlocked:
            raise GoldReferenceUnavailable("Matching gold PDF requires a password")
    pages = [page.extract_text() or "" for page in reader.pages]
    full_text = _body_text(pages)
    source_id = _case_source_id(case)
    report_sections = _report_sections(report)
    extracted, mappings = _matched_heading_sections(full_text, report_sections)
    return {
        "reference_id": source_id or pdf_path.parent.name or pdf_path.stem,
        "reference_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "sections": extracted,
        "section_mappings": mappings,
        "full_text": full_text,
        "page_count": len(pages),
        "source_name": pdf_path.name,
    }


def _chunks(text: str, tokenizer: Any) -> list[str]:
    return [row["excerpt"] for row in chunk_document(
        {"document_id": "report-evaluation", "content": text}, "fixed", tokenizer)]


def _encode(backend: IITBGE, texts: list[str], *, query: bool) -> np.ndarray:
    prepared = [BGE_QUERY_PREFIX + text if query else text for text in texts]
    batches = []
    for start in range(0, len(prepared), 16):
        batches.extend(backend.encode_batch(prepared[start:start + 16]))
    return normalize(batches, len(prepared))


def _bertscore_for_pairs(pairs: list[tuple[str, str]], model: str) -> tuple[float, float, float]:
    from bert_score import score as bert_score

    candidates = [candidate for candidate, _ in pairs]
    references = [reference for _, reference in pairs]
    precision, recall, f1 = bert_score(candidates, references, lang="en", model_type=model,
                                      device="cpu", verbose=False, rescale_with_baseline=False)
    return float(precision.mean()), float(recall.mean()), float(f1.mean())


_SCORE_FIELDS = ("score", "coverage", "consistency")


def _salvage_scores(content: str) -> dict[str, int] | None:
    """Read the integer score fields the model actually wrote, even from malformed or
    truncated JSON. Nothing is invented: a field is returned only if it is present."""
    found: dict[str, int] = {}
    for key in _SCORE_FIELDS:
        match = re.search(r'["\']?%s["\']?\s*[:=]\s*(\d{1,3})(?!\d)' % key, content, re.I)
        if match and 0 <= int(match.group(1)) <= 100:
            found[key] = int(match.group(1))
    return found if "score" in found else None


def _judge_result(parsed: dict[str, Any], attempt: int, *, salvaged: bool = False) -> dict[str, Any]:
    result = {"score": parsed["score"]}
    for key in ("coverage", "consistency"):
        value = parsed.get(key)
        result[key] = value if type(value) is int and 0 <= value <= 100 else None
    for key in ("material_omissions", "unsupported_or_conflicting_claims"):
        value = parsed.get(key)
        result[key] = [str(item)[:500] for item in value[:8]] if isinstance(value, list) else []
    rationale = str(parsed.get("rationale") or "")[:1200]
    result["rationale"] = rationale or ("Judge output was cut off or malformed; the integer scores it wrote "
                                        "were read directly." if salvaged else "")
    result["attempts"] = attempt + 1
    if salvaged:
        result["scores_salvaged_from_partial_output"] = True
    return result


async def _judge_section(section_id: str, section_title: str, generated: str, reference: str,
                         reference_scope: str, settings: Any,
                         client: httpx.AsyncClient) -> dict[str, Any]:
    system = (
        "You are a strict evaluator of a preliminary aviation incident report against the supplied "
        "matching gold report passages. Evaluate only information the published report section actually "
        "contains. Assess factual coverage, consistency with the reference, and unsupported additions. "
        "Do not reward verbosity or expose private reasoning. Return only one compact JSON object whose "
        "FIRST three fields are integer score, coverage and consistency (each 0-100), followed by "
        "material_omissions and unsupported_or_conflicting_claims (at most 3 short strings each) and a "
        "rationale of at most 40 words."
    )
    minimal_system = (
        "You are a strict evaluator of an aviation incident report section against matching gold report "
        "passages. Reply with ONLY this JSON and nothing else: "
        '{"score": <int 0-100>, "coverage": <int 0-100>, "consistency": <int 0-100>}'
    )
    payload_data = {
        "report_section": section_title,
        "section_id": section_id,
        "reference_scope": reference_scope,
        "generated_report_section": generated,
        "relevant_gold_report_passages": reference,
        "scoring_guide": "100 means accurate and materially complete for this section; 0 means largely absent or materially incorrect. Missing reference coverage is not a contradiction. Do not count unsupported detail as coverage.",
    }
    user_content = json.dumps(payload_data, ensure_ascii=False)
    # Qwen3 models may spend the whole output budget on hidden reasoning; turn it off.
    if "qwen3" in str(settings.api_model).casefold():
        user_content += " /no_think"
    output_cap = max(1, settings.output_token_quota("report_evaluation"))
    last_error = None
    use_json_mode = bool(settings.api_json_response_format)
    # Small hosted models intermittently emit malformed or over-long JSON. Retry with
    # progressively simpler requests; the last attempts only ask for the three scores.
    max_attempts = 6
    for attempt in range(max_attempts):
        minimal = attempt >= 3
        payload = {
            "model": settings.api_model, "stream": False,
            "messages": [{"role": "system", "content": minimal_system if minimal else system},
                         {"role": "user", "content": user_content}],
            "temperature": 0,
            "max_tokens": min(output_cap, 64) if minimal else min(output_cap, 320 if attempt == 0 else 384),
        }
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        content = ""
        try:
            response = await _hosted_post(client, url=f"{settings.api_base_url}/chat/completions",
                                          headers={"Authorization": f"Bearer {settings.api_key}"}, payload=payload,
                                          min_interval_seconds=settings.api_min_interval_seconds)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            content = choice["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            content = re.sub(r"<think>.*?</think>", "", str(content), flags=re.S | re.I).strip()
            if choice.get("finish_reason") in {"length", "max_tokens"}:
                salvaged = _salvage_scores(content)
                if salvaged:
                    return _judge_result(salvaged, attempt, salvaged=True)
                raise ValueError("Judge response was truncated")
            try:
                parsed, _ = parse_agent_json(content)
            except ValueError:
                salvaged = _salvage_scores(content)
                if salvaged:
                    return _judge_result(salvaged, attempt, salvaged=True)
                raise
            if not isinstance(parsed, dict):
                raise ValueError("Judge returned a non-object response")
            score = parsed.get("score")
            if type(score) is not int or not 0 <= score <= 100:
                salvaged = _salvage_scores(content)
                if salvaged:
                    return _judge_result(salvaged, attempt, salvaged=True)
                raise ValueError("Judge score did not validate as an integer from 0 to 100")
            return _judge_result(parsed, attempt)
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            last_error = exc
            if isinstance(exc, httpx.HTTPStatusError):
                status_code = exc.response.status_code
                if status_code in {400, 422} and use_json_mode:
                    use_json_mode = False  # some hosts reject response_format; retry without it
                elif status_code not in {408, 425, 429, 500, 502, 503, 504}:
                    break
            if attempt + 1 >= max_attempts:
                break
            logger.warning("Retrying final report judge; section=%s attempt=%s/%s error_type=%s",
                           section_id, attempt + 2, max_attempts, type(exc).__name__)
            await asyncio.sleep(min(4.0, 0.5 * (2 ** attempt)))
    raise ValueError(f"Section judge failed validation after {attempt + 1} attempts") from last_error


async def evaluate_final_report(report: dict[str, Any], reference: dict[str, Any], settings: Any,
                                *, save, run_id: str, prior: dict[str, Any] | None = None) -> None:
    generated = _report_sections(report)
    available_ids = {section["task_id"] for section in generated}
    prior_sections = deepcopy((prior or {}).get("sections") or {})
    result: dict[str, Any] = {
        **deepcopy(prior or {}),
        "status": "running", "label": "Final report section evaluation", "run_id": run_id,
        "reference_id": reference["reference_id"], "reference_sha256": reference["reference_sha256"],
        "reference_source": reference["source_name"], "reference_page_count": reference["page_count"],
        "section_mappings": {**reference["section_mappings"], **(prior or {}).get("section_mappings", {})},
        "sections_total": len(generated), "sections_completed": len(prior_sections),
        "sections": prior_sections, "human_review_required": True,
        "notice": "Gold reference is evaluator-only and is not used during generation or RAG.",
        "metric_warnings": list((prior or {}).get("metric_warnings") or []),
    }
    save(result)
    if not generated:
        result.update(status="not_assessable", reason="The saved report has no published sections to compare.")
        save(result)
        return

    cosine_rows: dict[str, dict[str, Any]] = {}
    bert_pairs: dict[str, list[tuple[str, str]]] = {}
    judge_reference_sections: dict[str, str] = {}
    section_by_id = {section["task_id"]: section for section in generated}
    try:
        tokenizer = await asyncio.to_thread(BGESubwordTokenizer)
        backend = IITBGE(tokenizer, base_url=settings.rag_embedding_url,
                         api_key=settings.rag_embedding_api_key,
                         ca_cert=settings.rag_embedding_ca_cert,
                         model=settings.rag_embedding_model,
                         timeout=settings.rag_embedding_timeout_seconds,
                         retries=settings.rag_embedding_retries)
        for section_id, section in section_by_id.items():
            previous = prior_sections.get(section_id) or {}
            gold_text = reference["sections"].get(section_id) or reference["full_text"]
            gen_chunks = await asyncio.to_thread(_chunks, section["text"], tokenizer)
            ref_chunks = await asyncio.to_thread(_chunks, gold_text, tokenizer)
            if not gen_chunks or not ref_chunks:
                continue
            if previous.get("cosine"):
                cosine_rows[section_id] = previous["cosine"]
            if previous.get("bertscore"):
                # Retain the validated score and skip local BERT inference.
                pass
            gen_vectors = await asyncio.to_thread(_encode, backend, gen_chunks, query=True)
            ref_vectors = await asyncio.to_thread(_encode, backend, ref_chunks, query=False)
            matrix = gen_vectors @ ref_vectors.T
            gen_best, ref_best = matrix.argmax(axis=1), matrix.argmax(axis=0)
            forward, reverse = float(matrix.max(axis=1).mean()), float(matrix.max(axis=0).mean())
            similarity = max(-1.0, min(1.0, (forward + reverse) / 2))
            if not cosine_rows.get(section_id):
                cosine_rows[section_id] = {
                    "cosine_similarity": round(similarity, 4), "cosine_distance": round(1.0 - similarity, 4),
                    "generated_chunks": len(gen_chunks), "reference_chunks": len(ref_chunks),
                    "method": "mean of bidirectional best-match cosine over BGE chunks",
                }
            pairs = [(gen_chunks[i], ref_chunks[int(j)]) for i, j in enumerate(gen_best)]
            pairs += [(ref_chunks[j], gen_chunks[int(i)]) for j, i in enumerate(ref_best)]
            if not previous.get("bertscore"):
                bert_pairs[section_id] = pairs
            relevance = matrix.max(axis=0)
            selected = sorted(np.argsort(relevance)[-min(30, len(ref_chunks)):].tolist())
            judge_reference_sections[section_id] = "\n\n".join(ref_chunks[index] for index in selected)
            result["section_mappings"][section_id] = reference["section_mappings"].get(
                section_id, "Semantic selection from matched gold report")
    except Exception as exc:
        logger.warning("Final report embedding evaluation unavailable; error_type=%s", type(exc).__name__)
        result["metric_warnings"].append({"metric": "cosine_similarity", "error_type": type(exc).__name__})
        result["metric_warnings"].append({"metric": "BERTScore", "error_type": "embedding_chunk_alignment_unavailable"})

    for section_id, section in section_by_id.items():
        if section_id not in judge_reference_sections:
            gold_text = reference["sections"].get(section_id) or reference["full_text"]
            judge_reference_sections[section_id] = gold_text[:10000] + (
                "\n[Reference passage selection bounded.]" if len(gold_text) > 10000 else "")

    bert_rows: dict[str, dict[str, Any]] = {}
    for section_id, previous in prior_sections.items():
        if previous.get("bertscore"):
            bert_rows[section_id] = previous["bertscore"]
    try:
        for section_id, pairs in bert_pairs.items():
            precision, recall, f1 = await asyncio.to_thread(
                _bertscore_for_pairs, pairs, settings.final_report_bertscore_model)
            bert_rows[section_id] = {"precision": round(precision, 4), "recall": round(recall, 4),
                                     "f1": round(f1, 4), "model": settings.final_report_bertscore_model}
    except Exception as exc:
        logger.warning("Final report BERTScore unavailable; error_type=%s", type(exc).__name__)
        result["metric_warnings"].append({"metric": "BERTScore", "error_type": type(exc).__name__})

    async with httpx.AsyncClient(verify=settings.api_ca_cert, timeout=settings.api_timeout_seconds) as client:
        for section in generated:
            section_id, title = section["task_id"], section["title"]
            previous = prior_sections.get(section_id) or {}
            if previous.get("llm_judge") and previous.get("status") == "completed":
                continue
            row: dict[str, Any] = {"title": title,
                                   "reference_scope": result["section_mappings"].get(section_id),
                                   "llm_judge": None,
                                   "cosine": cosine_rows.get(section_id) or previous.get("cosine"),
                                   "bertscore": bert_rows.get(section_id) or previous.get("bertscore")}
            # Two full rounds per section: a transient host or model failure in one round
            # should not leave the section without a judge score.
            for round_number in range(2):
                try:
                    row["llm_judge"] = await _judge_section(
                        section_id, title, section["text"], judge_reference_sections[section_id],
                        str(row["reference_scope"] or "Matched gold report passages"), settings, client)
                    row["status"] = "completed"
                    row.pop("judge_error", None)
                    break
                except Exception as exc:
                    row["status"] = "partial" if row["cosine"] or row["bertscore"] else "failed"
                    cause = exc.__cause__ or exc
                    row["judge_error"] = type(cause).__name__
                    if isinstance(cause, httpx.HTTPStatusError):
                        row["judge_http_status"] = cause.response.status_code
                    logger.warning("Final report judge unavailable; section=%s round=%s error_type=%s",
                                   section_id, round_number + 1, type(cause).__name__)
                    await asyncio.sleep(2.0)
            result["sections"][section_id] = row
            result["sections_completed"] = len(result["sections"])
            save(result)

    rows = list(result["sections"].values())
    judge_scores = [row["llm_judge"]["score"] for row in rows if row.get("llm_judge")]
    cosine_scores = [row["cosine"]["cosine_similarity"] * 100 for row in rows if row.get("cosine")]
    bert_scores = [row["bertscore"]["f1"] * 100 for row in rows if row.get("bertscore")]
    result["overall"] = {
        "llm_judge_score": round(sum(judge_scores) / len(judge_scores), 1) if judge_scores else None,
        "cosine_similarity_percent": round(sum(cosine_scores) / len(cosine_scores), 1) if cosine_scores else None,
        "cosine_distance": round(1 - (sum(cosine_scores) / len(cosine_scores) / 100), 4) if cosine_scores else None,
        "bertscore_f1_percent": round(sum(bert_scores) / len(bert_scores), 1) if bert_scores else None,
        "sections_scored": len(rows),
        "aggregation": "unweighted mean of available section values; metric families remain separate",
    }
    result["status"] = "completed" if rows and all(
        row.get("status") == "completed" and row.get("llm_judge") for row in rows) else "incomplete"
    save(result)
