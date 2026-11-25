from blackwing.agents.incident_identity import conflicts_with_case


NARRATIVE = "On 2 October 2013 at Delhi IGI, flight GOW336 conflicted with an AN-32."


def test_explicit_other_incident_is_rejected_even_with_matching_case_metadata():
    source = {"incident_id": "current-case", "title": "Delhi Ariana AFG311 Preliminary 2025-11-23"}
    assert conflicts_with_case(source, NARRATIVE)


def test_matching_and_missing_identifiers_are_not_treated_as_conflicts():
    assert not conflicts_with_case({"title": "Delhi GOW336", "incident_date": "2013-10-02"}, NARRATIVE)
    assert not conflicts_with_case({"title": "Untitled report"}, NARRATIVE)


def test_excerpt_dates_are_not_assumed_to_be_incident_dates():
    assert not conflicts_with_case({"title": "GOW336 report", "excerpt": "Procedure revised 2025-11-23."}, NARRATIVE)
