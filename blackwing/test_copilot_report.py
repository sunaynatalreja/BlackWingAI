"""Regression checks for accepted-finding report assembly and evidence references."""

from io import BytesIO
from unittest import TestCase, main

from pypdf import PdfReader

from copilot_report import REPORT_SECTIONS, compile_report, render_report_pdf


class CopilotReportTests(TestCase):
    def test_report_contains_only_traceable_accepted_case_findings(self):
        incident = {"case_id": "DEMO", "airport": "Example", "narrative": "Raw intake text."}
        finding = {
            "case_id": "DEMO", "status": "accepted", "finding_id": "H1", "section": "hazards",
            "kind": "potential_hazard", "claim": "Potential taxi conflict.",
            "uncertainty": "Separation requires verification.",
            "reviewed_by": "Test reviewer", "reviewed_at": "2026-11-13T10:00:00+05:30",
            "citations": [{
                "source_id": "SRC-1", "title": "Example case notes", "page": 2,
                "chunk_id": "SRC-1-p002-c00", "scope": "submitted",
                "excerpt": "Two aircraft were reported taxiing near the same intersection.",
            }],
        }
        findings = [
            finding,
            {**finding, "finding_id": "M1", "section": "mitigation", "claim": "Review the taxi sequence.",
             "related_finding_ids": ["H1", "P1"]},
            {**finding, "finding_id": "P1", "status": "proposed", "claim": "Exclude this draft claim."},
            {**finding, "finding_id": "U1", "citations": [], "claim": "Exclude this uncited claim."},
            {**finding, "finding_id": "X1", "case_id": "OTHER", "claim": "Exclude this different case."},
            {**finding, "finding_id": "A1", "reviewed_by": "", "claim": "Exclude this unsigned finding."},
        ]
        report = compile_report(incident, findings)
        self.assertEqual(report["accepted_count"], 2)
        self.assertEqual(list(REPORT_SECTIONS), [section["key"] for section in report["sections"]])
        self.assertEqual(len(report["sources"]), 1)
        self.assertEqual(report["sections"][-1]["findings"][0]["related_finding_ids"], ["H1"])

        pdf = PdfReader(BytesIO(render_report_pdf(report)))
        text = " ".join(page.extract_text() for page in pdf.pages)
        self.assertIn("Test reviewer", text)
        self.assertIn("SRC-1-p002-c00", text)
        self.assertIn("No accepted findings recorded.", text)
        self.assertIn("Separation requires verification.", text)
        self.assertNotIn("Exclude this", text)
        self.assertNotIn("Raw intake text.", text)
        self.assertTrue(any(
            annotation.get_object().get("/Subtype") == "/Link"
            for page in pdf.pages for annotation in page.get("/Annots", [])
        ))


if __name__ == "__main__":
    main()
