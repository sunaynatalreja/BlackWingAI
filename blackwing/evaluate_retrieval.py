"""BLACKWING AI | reproducible, no-API retrieval evaluation (JFK DEVELOPMENT_10).

Run in the existing 04_CODE_STARTER folder:
    python evaluate_retrieval.py

Reuses blackwing_app.py's exact BM25 (and optional hybrid/rerank) implementation.
Never sends gold answers to a model, never modifies your original gold questions,
never indexes final reports, and never changes index_pilot.py or blackwing_app.py.

Output automatically recorded under 04_CODE_STARTER/output/retrieval_evaluation/.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import re
import sqlite3
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QUESTIONS_DEFAULT = HERE / "evaluation" / "jfk-development-retrieval.csv"
OUTPUT_DEFAULT = HERE / "output" / "retrieval_evaluation"
SCRIPT_VERSION = "1.0.0"

# Local import ensures identical search ranking to the running BLACKWING app.
from blackwing_app import load_index, search


def file_sha256(file: Path) -> str:
    h = hashlib.sha256()
    with file.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_development_questions(file: Path) -> list[dict]:
    if not file.is_file():
        raise FileNotFoundError(f"Development CSV not found: {file}")
    with file.open("r", encoding="utf-8-sig", newline="") as f:
        questions = list(csv.DictReader(f))
    if not questions:
        raise ValueError("Development question CSV has no data rows.")
    required = {"question_id", "query", "gold_supporting_source_ids", "exact_page_or_locator_to_verify", "evaluation_split"}
    missing = required - set(questions[0])
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
    if len(questions) != 10:
        raise ValueError(f"Safety check: expected exactly 10 JFK development questions, received {len(questions)}")
    if any(q.get("evaluation_split", "") != "DEVELOPMENT_10" or q.get("case_label", "").upper() != "JFK" for q in questions):
        raise ValueError("Only JFK DEVELOPMENT_10 questions are allowed. Never run held-out final questions during tuning.")
    ids = [q["question_id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question IDs in development CSV.")
    if not all((q.get("query") or "").strip() for q in questions):
        raise ValueError("One or more questions have an empty query.")
    return questions


def expected_source_ids(row: dict) -> list[str]:
    # Example: 'BW-SRC-001 (JFK NTSB ATC Group Chair Factual)'
    ids = re.findall(r"BW-(?:SRC|REG)-\d{3}", row["gold_supporting_source_ids"], flags=re.I)
    if not ids:
        raise ValueError(f"{row['question_id']}: cannot parse expected document ID from 'gold_supporting_source_ids'")
    return sorted(set(x.upper() for x in ids))


def expected_pages(row: dict) -> tuple[list[int], list[int]]:
    """Return PRIMARY pages and SUPPORTING page span for diagnostic use.

    'PDF pp.6–8, especially Fig.2 p.7' -> primary p7, supporting pp6-8.
    One-page cases -> primary and supporting are the same.
    A page locator is ONLY a proxy for answer-bearing evidence, not semantic proof.
    """
    loc = (row.get("exact_page_or_locator_to_verify") or "").strip()
    normalized = loc.lower().replace("–", "-").replace("—", "-")
    ranges = re.findall(r"\bpp?\.?\s*(\d+)\s*-\s*(\d+)\b", normalized)
    supporting: set[int] = set()
    for left, right in ranges:
        l, r = int(left), int(right)
        if 0 < l <= r <= 10000 and r - l <= 20:
            supporting.update(range(l, r + 1))
    singles = [int(x) for x in re.findall(r"\bp\.?\s*(\d+)\b", normalized)]
    supporting.update(singles)
    if not supporting:
        raise ValueError(f"{row['question_id']}: cannot parse PDF pages from locator '{loc}'.")
    if "especially" in normalized and singles:
        primary = {singles[-1]}
    elif ranges:
        # If a range is supplied without a priority, use all pages as acceptable.
        primary = supporting.copy()
    else:
        primary = set(singles)
    return sorted(primary), sorted(supporting)


def ensure_safe_index(recs: list[dict]) -> None:
    if not recs:
        raise ValueError("The index is empty; run index_pilot.py first.")
    for chunk in recs:
        name = str(chunk.get("filename", "")).upper()
        typ = str(chunk.get("source_type", "")).upper()
        role = str(chunk.get("source_role", "")).upper()
        if ("GOLD" in name or "FINAL" in name or "GOLD" in role or
                typ in {"NTSB_FINAL", "AAIB_FINAL", "GOLD_ANSWER"}):
            raise ValueError(f"Potential data leakage found: chunk {chunk.get('chunk_id')} / {name}. Stop; review index policy.")


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def append_csv(path: Path, fields: list[str], row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)


def to_float(v):
    return 0.0 if v is None else float(v)


def metrics_for(results: list[dict]) -> dict:
    n = len(results)
    if not n:
        return {}
    def pct(field):
        return round(100.0 * sum(int(r[field]) for r in results) / n, 1)
    return {
        "questions": n,
        "document_hit_at_5_pct": pct("document_hit_at_5"),
        "primary_page_hit_at_1_pct": pct("primary_page_hit_at_1"),
        "primary_page_hit_at_3_pct": pct("primary_page_hit_at_3"),
        "primary_page_hit_at_5_pct": pct("primary_page_hit_at_5"),
        "mrr_at_5_primary_page": round(sum(1.0 / int(r["first_primary_page_rank"]) if r["first_primary_page_rank"] else 0 for r in results) / n, 4),
        "supporting_page_coverage_at_5_pct": round(sum(to_float(r["supporting_page_coverage_at_5"]) for r in results) / n * 100.0, 1),
    }


def make_report_html(run_id: str, method: str, rows: list[dict], stats: dict, index_sha: str, question_sha: str) -> str:
    esc = lambda s: html.escape(str(s), quote=True)
    cards = "".join(
        f'<div class="metric"><div class="big">{esc(value)}</div><div>{esc(label)}</div></div>'
        for label, value in [
            ("Questions tested", stats["questions"]),
            ("Page Hit@5", str(stats["primary_page_hit_at_5_pct"]) + "%"),
            ("Page Hit@1", str(stats["primary_page_hit_at_1_pct"]) + "%"),
            ("MRR@5 (page)", stats["mrr_at_5_primary_page"]),
            ("Document Hit@5", str(stats["document_hit_at_5_pct"]) + "%"),
        ]
    )
    tr = []
    for r in rows:
        status = 'hit' if r["primary_page_hit_at_5"] else 'miss'
        tr.append("<tr>" +
                  f'<td><b>{esc(r["question_id"])}</b></td>' +
                  f'<td>{esc(r["query"])}</td>' +
                  f'<td>{esc(r["expected_document_ids"])} p.{esc(r["expected_primary_pages"])}</td>' +
                  f'<td class="{status}">{"HIT" if r["primary_page_hit_at_5"] else "MISS"}</td>' +
                  f'<td>{esc(r["first_primary_page_rank"] or "—")}</td>' +
                  f'<td>{esc(r["top5_chunk_ids"])}</td>' + "</tr>")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>BLACKWING AI | JFK retrieval evaluation | {esc(run_id)}</title>
<style>
:root{{--blue:#15345a;--mint:#d7f7e6;--rose:#fce4e4;--pale:#f3f7fb}}
body{{font-family:Segoe UI,Arial,sans-serif;color:#172b42;max-width:1280px;margin:0 auto;padding:36px;background:var(--pale)}}
header{{background:var(--blue);color:white;padding:30px;border-radius:15px}}
h1{{margin:0 0 8px;font-size:29px}}.muted{{color:#a7cee9}}.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:25px 0}}
.metric{{background:#fff;border-radius:12px;border:1px solid #dbe4ec;min-width:155px;flex:1;padding:18px}}
.big{{font-size:27px;font-weight:700;color:var(--blue)}}.notice{{background:#fff6da;padding:16px;border-left:4px solid #dd9a25;margin:20px 0}}
table{{border-collapse:collapse;background:white;width:100%;font-size:13px}}th{{background:#e6eef6;text-align:left}}th,td{{padding:11px;border:1px solid #dce4ec;vertical-align:top}}
.hit{{background:var(--mint);font-weight:bold}}.miss{{background:var(--rose);font-weight:bold}}
code{{font-family:Consolas,monospace;overflow-wrap:anywhere}}footer{{font-size:12px;color:#65798d;margin-top:25px}}
</style></head><body>
<header><h1>BLACKWING AI</h1><div>JFK Pilot · Automated retrieval evaluation</div><div class="muted">Run {esc(run_id)} | Method: {esc(method)} | Development questions only</div></header>
<div class="cards">{cards}</div>
<div class="notice"><b>What this measures:</b> whether the expected NTSB source <b>page</b> appears in the top 1 / 3 / 5 retrieved chunks. It does <b>not</b> score a Qwen3 answer, prove that a passage entails the reference answer, or approve draft gold labels. Figure-based BW-G005 needs manual visual verification. The source document can appear multiple times as different chunks.</div>
<h2>Question-by-question evidence retrieval</h2><table><thead><tr><th>ID</th><th>Development question</th><th>Expected page(s)</th><th>Page Hit@5</th><th>First page rank</th><th>Top five chunk IDs</th></tr></thead><tbody>{''.join(tr)}</tbody></table>
<h2>Audit</h2><p><b>Index SHA-256:</b> <code>{esc(index_sha)}</code><br><b>Question CSV SHA-256:</b> <code>{esc(question_sha)}</code></p>
<p>Detailed results and excerpts are available as <code>results.csv</code> and <code>retrieved_passages.csv</code>. No API requests were made.</p>
<footer>Prototype evaluation • Human verification required • Keep protected final-test questions and answer files outside retrieval.</footer>
</body></html>'''


def log_sqlite(db: Path, run_row: dict, results: list[dict]):
    # Only metadata/results saved. Never write API keys or raw protected gold answers.
    with sqlite3.connect(db) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS runs (
          run_id TEXT PRIMARY KEY, timestamp_utc TEXT, method TEXT,
          question_count INTEGER, page_hit_at_5_pct REAL, mrr_at_5 REAL,
          index_sha256 TEXT, questions_sha256 TEXT, output_folder TEXT)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS per_question (
          run_id TEXT, question_id TEXT, query TEXT, page_hit_at_5 INTEGER,
          first_page_rank INTEGER, expected_pages TEXT, top5_chunk_ids TEXT,
          PRIMARY KEY (run_id, question_id))''')
        conn.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)", (
            run_row["run_id"], run_row["timestamp_utc"], run_row["method"], run_row["questions"],
            run_row["primary_page_hit_at_5_pct"], run_row["mrr_at_5_primary_page"],
            run_row["index_sha256"], run_row["questions_sha256"], run_row["output_folder"]
        ))
        conn.executemany('INSERT INTO per_question VALUES (?,?,?,?,?,?,?)', [(
            run_row["run_id"], r["question_id"], r["query"], r["primary_page_hit_at_5"],
            int(r["first_primary_page_rank"]) if r["first_primary_page_rank"] else None,
            r["expected_primary_pages"], r["top5_chunk_ids"]
        ) for r in results])


def main() -> int:
    p = argparse.ArgumentParser(description="BLACKWING AI: JFK development questions → automatic retrieval scorecard; no LLM API")
    p.add_argument("--questions", type=Path, default=QUESTIONS_DEFAULT, help="JFK DEVELOPMENT_10 CSV only")
    p.add_argument("--out", type=Path, default=OUTPUT_DEFAULT, help="Where to save the automatically logged results")
    p.add_argument("--hybrid", action="store_true", help="Use local dense embeddings + BM25, if installed")
    p.add_argument("--rerank", action="store_true", help="Use cross-encoder reranker (requires local model)")
    a = p.parse_args()
    if a.rerank and not a.hybrid:
        p.error("For this baseline, rerank is enabled only with --hybrid --rerank")
    method = "BM25" if not a.hybrid else ("HYBRID_RRF_RERANK" if a.rerank else "HYBRID_RRF")
    started = dt.datetime.now(dt.timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_" + method.lower()
    folder = a.out / "runs" / run_id
    if folder.exists():
        run_id += "_" + started.strftime("%f")
        folder = a.out / "runs" / run_id
    folder.mkdir(parents=True)
    log_file = folder / "run.log"
    def log(message):
        line = f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log("START | BLACKWING AI retrieval evaluation | method=" + method)
    rows, retrieved = [], []
    fields = [
        "question_id", "evaluation_split", "query", "expected_document_ids", "expected_primary_pages",
        "expected_supporting_pages", "gold_validation_status", "document_hit_at_5", "primary_page_hit_at_1",
        "primary_page_hit_at_3", "primary_page_hit_at_5", "first_primary_page_rank", "first_matching_chunk_id",
        "supporting_page_coverage_at_5", "top5_chunk_ids", "review_note"
    ]
    detail_fields = ["question_id", "rank", "chunk_id", "document_id", "page", "source_status", "score_rrf", "excerpt"]
    try:
        questions = load_development_questions(a.questions)
        recs = load_index()
        ensure_safe_index(recs)
        from blackwing_app import INDEX
        index_sha, question_sha = file_sha256(INDEX), file_sha256(a.questions)
        log(f"Input: {len(questions)} questions, {len(recs)} chunks | index_sha={index_sha[:12]} | csv_sha={question_sha[:12]}")
        for q in questions:
            tick = time.perf_counter()
            expected_docs = expected_source_ids(q)
            primary, supporting = expected_pages(q)
            hits, actual_method = search(recs, q["query"], topk=5, hybrid=a.hybrid, rerank=a.rerank)
            if actual_method.startswith("BM25_ONLY") and a.hybrid:
                raise RuntimeError("Hybrid retrieval requested but search returned BM25 mode")
            def match(h):
                return str(h.get("document_id", "")).upper() in expected_docs
            primary_ranks = [rank for rank, h in enumerate(hits, 1) if match(h) and int(h.get("page", 0)) in primary]
            anydoc = any(match(h) for h in hits)
            available_supporting = {int(h.get("page", 0)) for h in hits if match(h) and int(h.get("page", 0)) in supporting}
            first = primary_ranks[0] if primary_ranks else None
            found = hits[first-1]["chunk_id"] if first else ""
            note = ("Primary source page retrieved; inspect excerpt for answer support." if first else
                    "Expected page absent in top 5; inspect alternative top chunks and query/indexing.")
            if q["question_id"] == "BW-G005":
                note += " Figure 2 is visual; page match alone does not prove figure understanding."
            rec = {
                "question_id": q["question_id"], "evaluation_split": q["evaluation_split"],
                "query": q["query"], "expected_document_ids": ";".join(expected_docs),
                "expected_primary_pages": ";".join(map(str, primary)),
                "expected_supporting_pages": ";".join(map(str, supporting)),
                "gold_validation_status": q.get("human_validation_status", "DRAFT"),
                "document_hit_at_5": int(anydoc),
                "primary_page_hit_at_1": int(bool(first and first <= 1)),
                "primary_page_hit_at_3": int(bool(first and first <= 3)),
                "primary_page_hit_at_5": int(bool(first and first <= 5)),
                "first_primary_page_rank": first or "", "first_matching_chunk_id": found,
                "supporting_page_coverage_at_5": round(len(available_supporting) / len(supporting), 4),
                "top5_chunk_ids": " | ".join(h["chunk_id"] for h in hits),
                "review_note": note,
            }
            rows.append(rec)
            for rank, h in enumerate(hits, 1):
                retrieved.append({
                    "question_id": q["question_id"], "rank": rank, "chunk_id": h["chunk_id"],
                    "document_id": h.get("document_id", ""), "page": h.get("page", ""),
                    "source_status": h.get("source_status", ""), "score_rrf": h.get("score_rrf", ""),
                    "excerpt": str(h.get("text", ""))[:800]
                })
            # Continuous file write means evidence/results survive an interrupted run.
            write_csv(folder / "results.csv", fields, rows)
            write_csv(folder / "retrieved_passages.csv", detail_fields, retrieved)
            log(f"{q['question_id']}: PAGE_HIT@5={rec['primary_page_hit_at_5']} first_rank={first or '-'} "
                f"doc_hit={int(anydoc)} elapsed={time.perf_counter()-tick:.2f}s")

        stats = metrics_for(rows)
        summary = {
            "run_id": run_id, "started_at_utc": started.isoformat(timespec="seconds"),
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "method": method, "evaluation_split": "DEVELOPMENT_10", "questions_csv": str(a.questions),
            "index": str(INDEX), "index_chunk_count": len(recs), "index_sha256": index_sha,
            "questions_sha256": question_sha, "metrics": stats,
            "metric_definitions": {
                "document_hit_at_5": "Expected document ID occurs in top 5 chunks; insufficient for evidence quality.",
                "primary_page_hit_at_k": "At least one expected primary document page appears among top k chunks (page-level proxy).",
                "mrr_at_5_primary_page": "Mean reciprocal rank of first expected source page within top five; zero for misses.",
                "supporting_page_coverage_at_5": "Fraction of specified supporting pages seen among top five, averaged over questions.",
            },
            "caveats": [
                "This is retrieval only; NO Qwen API call, NO generated answer, NO answer accuracy score.",
                "Page/source presence does not establish that chunk text actually entails a proposed gold answer.",
                "The 10 answers remain draft pending independent human approval.",
                "BW-G005 needs human inspection of the NTSB Figure 2 image.",
                "Do not tune on validation or protected final-test questions.",
            ]
        }
        (folder / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        html_content = make_report_html(run_id, method, rows, stats, index_sha, question_sha)
        (folder / "report.html").write_text(html_content, encoding="utf-8")
        lines = [
            "# BLACKWING AI — JFK retrieval evaluation", "", f"**Run:** `{run_id}`", f"**Method:** {method}",
            "**Dataset:** JFK development 10 only (draft gold labels)", "",
            f"- Expected-source document Hit@5: **{stats['document_hit_at_5_pct']}%**",
            f"- Expected-primary-page Hit@1: **{stats['primary_page_hit_at_1_pct']}%**",
            f"- Expected-primary-page Hit@3: **{stats['primary_page_hit_at_3_pct']}%**",
            f"- Expected-primary-page Hit@5: **{stats['primary_page_hit_at_5_pct']}%**",
            f"- MRR@5 (expected page): **{stats['mrr_at_5_primary_page']}**", "",
            "| Question | Primary page | First correct-page rank | Hit@5 |", "|---|---|---:|---|",
        ]
        lines += [f"| {r['question_id']} | {r['expected_primary_pages']} | {r['first_primary_page_rank'] or '-'} | {'Yes' if r['primary_page_hit_at_5'] else 'No'} |" for r in rows]
        lines += ["", "**Interpretation:** Page-match retrieval proxy, not semantic relevance or answer correctness. No IIT API used. Human gold approval remains pending."]
        (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        run_row = {
            "run_id": run_id, "timestamp_utc": started.isoformat(timespec="seconds"), "method": method,
            "questions": stats["questions"], "primary_page_hit_at_5_pct": stats["primary_page_hit_at_5_pct"],
            "mrr_at_5_primary_page": stats["mrr_at_5_primary_page"], "index_sha256": index_sha,
            "questions_sha256": question_sha, "output_folder": str(folder),
        }
        hist_fields = list(run_row)
        append_csv(a.out / "run_history.csv", hist_fields, run_row)
        log_sqlite(a.out / "evaluation_history.sqlite", run_row, rows)
        log("DONE | primary-page Hit@5 = " + str(stats["primary_page_hit_at_5_pct"]) + "% | MRR@5=" + str(stats["mrr_at_5_primary_page"]))
        print("\nSaved files:")
        for n in ("report.html", "report.md", "results.csv", "retrieved_passages.csv", "summary.json", "run.log"):
            print("  " + str(folder / n))
        print("\nHistory: " + str(a.out / "run_history.csv"))
        print("SQLite: " + str(a.out / "evaluation_history.sqlite"))
        print("IMPORTANT: No generated-answer accuracy was measured; gold answers remain unapproved.")
        return 0
    except Exception as e:
        log("ERROR | " + type(e).__name__ + ": " + str(e))
        with (folder / "error_traceback.txt").open("w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        return 1

if __name__ == "__main__":
    sys.exit(main())
