"""PDF fallback preserves page citations when pypdf has a parsing limit."""
from types import SimpleNamespace

import pymupdf
from blackwing.rag.corpus import _text_units


def test_pdf_page_fallback_retains_original_page_number(tmp_path, monkeypatch):
    source = tmp_path / "two_pages.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50, 50), "First page factual record")
        pdf.new_page().insert_text((50, 50), "Second page safety evidence")
        pdf.save(source)

    class WorkingPage:
        def extract_text(self):
            return "First page factual record"

    class FailingPage:
        def extract_text(self):
            raise RuntimeError("Reached maximum number of tokens per stream")

    monkeypatch.setattr(
        "pypdf.PdfReader",
        lambda _: SimpleNamespace(pages=[WorkingPage(), FailingPage()]),
    )
    units = _text_units(source, {})
    assert len(units) == 2
    assert units[0] == (1, "", "First page factual record")
    assert units[1][0] == 2
    assert "Second page safety evidence" in units[1][2]
