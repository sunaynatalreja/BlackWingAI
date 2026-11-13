"""Compare page-bounded chunk sizes on the JFK development questions using BM25."""
import argparse
import json
from pathlib import Path
from statistics import mean

import pymupdf

from blackwing_app import search
from evaluate_retrieval import expected_pages, expected_source_ids, load_development_questions

SOURCES = ("BW-SRC-001", "BW-SRC-002", "BW-SRC-003")
SETTINGS = ((100, 20), (216, 30), (320, 48))


def factual_pages(source_directory):
    for source_id in SOURCES:
        paths = list(source_directory.glob(f"{source_id}_*_Factual_*.pdf"))
        if len(paths) != 1:
            raise ValueError(f"Expected one JFK factual PDF for {source_id}.")
        with pymupdf.open(paths[0]) as document:
            for page in document:
                yield source_id, page.number + 1, page.get_text("text")


def windows(pages, size, overlap):
    records = []
    for source_id, page, text in pages:
        words = text.split()
        for offset in range(0, len(words), size - overlap):
            passage = " ".join(words[offset:offset + size])
            if len(passage) > 60:
                records.append({"chunk_id": f"{source_id}-p{page:03}-w{offset:04}",
                                "document_id": source_id, "page": page, "text": passage})
            if offset + size >= len(words):
                break
    return records


def trial(pages, questions, size, overlap):
    records = windows(pages, size, overlap)
    results = []
    for question in questions:
        sources = expected_source_ids(question)
        primary, _ = expected_pages(question)
        hits, _ = search(records, question["query"], topk=5)
        rank = next((rank for rank, hit in enumerate(hits, 1)
                     if hit["document_id"] in sources and hit["page"] in primary), None)
        results.append({"question_id": question["question_id"], "first_primary_page_rank": rank,
                        "top5_chunk_ids": [hit["chunk_id"] for hit in hits]})
    ranks = [result["first_primary_page_rank"] for result in results]
    return {"words": size, "overlap": overlap, "chunks": len(records),
            "primary_page_hit_at_5_count": sum(rank is not None for rank in ranks),
            "primary_page_mrr_at_5": round(mean(1 / rank if rank else 0 for rank in ranks), 4),
            "questions": len(questions), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path, help="JFK factual PDF directory in the master project")
    parser.add_argument("--out", type=Path, help="Write JSON results to this path")
    args = parser.parse_args()
    questions_path = Path(__file__).parent / "evaluation" / "jfk-development-retrieval.csv"
    questions = load_development_questions(questions_path)
    pages = list(factual_pages(args.source_directory))
    results = {"method": "BM25", "source_ids": SOURCES, "pages": len(pages),
               "results": [trial(pages, questions, size, overlap) for size, overlap in SETTINGS]}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for row in results["results"]:
        print(f"{row['words']}/{row['overlap']} words | {row['chunks']} chunks | "
              f"Hit@5 {row['primary_page_hit_at_5_count']}/{row['questions']} | "
              f"MRR@5 {row['primary_page_mrr_at_5']}")


if __name__ == "__main__":
    main()
