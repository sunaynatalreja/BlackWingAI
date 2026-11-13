from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

import copilot_state as state
from safety_register import build_hazard_register


class InvestigationStateTest(TestCase):
    def setUp(self):
        workspace = TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.enterContext(patch.object(state, "DATABASE", Path(workspace.name) / "case.sqlite"))

    def test_new_case_initial_upload_is_atomic_and_redacted(self):
        document = ("radar.txt", b"Observed vehicle crossed the hold line. Contact: test@example.com")
        bad_document = ("final investigation report.txt", b"An excluded source.")
        with self.assertRaisesRegex(ValueError, "cleared for investigation"):
            state.create_case("EGLL", "2025-11-06", "United Kingdom", "Pilot reported a vehicle crossing.",
                              files=[document, bad_document], provenance="Operations log")
        self.assertFalse(any(item["airport"] == "EGLL" for item in state.list_cases()))

        case = state.create_case("EGLL", "2025-11-06", "United Kingdom", "Pilot reported a vehicle crossing.",
                                 aircraft_details="Cargo 91, Boeing 777", files=[document],
                                 provenance="Operations log")
        workspace = state.get_case(case["case_id"])
        self.assertEqual(workspace["incident"]["aircraft_details"], "Cargo 91, Boeing 777")
        self.assertEqual(len(workspace["evidence"]), 1)
        self.assertEqual(workspace["evidence"][0]["source_role"], "submitted")
        self.assertNotIn("test@example.com", state.case_chunks(case["case_id"])[0]["text"])

    def test_renamed_protected_sources_are_blocked(self):
        case = state.create_case("KSEA", "2025-11-06", "United States", "Potential runway conflict.")
        with self.assertRaisesRegex(ValueError, "final report or evaluation"):
            state.add_case_evidence(case["case_id"], [("notes.txt", b"Aviation Investigation Final Report")],
                                    provenance="Renamed document")
        self.assertEqual(state.get_case(case["case_id"])["evidence"], [])

    def test_historical_inventory_uses_trusted_index_and_saved_cases(self):
        real_case = state.create_case("EGLL", date.today(), "United Kingdom", "A reported runway conflict.")
        future_case = state.create_case("EGLL", date.today() + timedelta(days=1), "United Kingdom",
                                        "Example of a future-dated case.")
        reference = {"source_role": "evidence", "airport": "KSEA", "jurisdiction": "US",
                     "event_date": "2024-10-01"}
        records = [
            {**reference, "case_id": "SEA-2024"},
            {**reference, "case_id": "SEA-2024-PILOT"},
            {**reference, "case_id": "HELD-OUT-2024", "held_out": True},
            {**reference, "case_id": "IND-AAIB-001-VT-BXL"},
            {**reference, "case_id": "DABOLIM-2026", "document_id": "IND-AAIB-001"},
            {**reference, "case_id": "VOLUNTARY", "source_status": "UNVERIFIED_VOLUNTARY_REPORT"},
        ]
        with patch.object(state, "library", return_value=records):
            cases = {case["case_id"] for case in state.list_historical_cases()}
            self.assertTrue({"SEA-2024", real_case["case_id"]} <= cases)
            self.assertFalse(cases.intersection({"SEA-2024-PILOT", "HELD-OUT-2024", "VOLUNTARY",
                                                "IND-AAIB-001-VT-BXL", "DABOLIM-2026",
                                                state.GOA, "NEW-MUMBAI-SYNTHETIC", future_case["case_id"]}))
            self.assertEqual(state.get_case("SEA-2024")["incident"]["airport"], "KSEA")

    def test_hazard_register_preserves_evidence_and_linked_findings(self):
        hazard = {"finding_id": "H1", "kind": "hazard", "claim": "Potential runway conflict",
                  "status": "pending", "verification": {"eligible": False}, "citations": [{"source_id": "ATC"}],
                  "potential_consequence": "Aircraft collision", "existing_controls": ["Controller warning"]}
        risk = {"finding_id": "R1", "kind": "risk", "related_finding_ids": ["H1"],
                "claim": "Official severity and probability remain undetermined."}
        action = {"finding_id": "A1", "section": "mitigation", "related_finding_ids": ["H1"]}
        unrelated = {"finding_id": "A2", "section": "mitigation", "related_finding_ids": ["H2"]}
        row, legacy = build_hazard_register([hazard, risk, action, unrelated,
                                            {"finding_id": "H2", "kind": "hazard", "status": "accepted"}])
        self.assertEqual(row["supporting_evidence"], [{"source_id": "ATC"}])
        self.assertEqual(row["existing_controls"], ["Controller warning"])
        self.assertEqual(row["potential_consequence"], "Aircraft collision")
        self.assertEqual(row["risk_assessment"], [risk])
        self.assertEqual(row["proposed_mitigations"], [action])
        self.assertEqual(row["verification_status"], "Evidence check flagged")
        self.assertEqual(legacy["existing_controls"], [])
        self.assertEqual(legacy["potential_consequence"], "")
        self.assertEqual(legacy["verification_status"], "Accepted by reviewer")


if __name__ == "__main__":
    main()
