"""
Builds docs/Local_Voice_Chatbot_Technical_Documentation.pdf from the
_part*.html fragments.

Two-pass build:
  Pass 1 renders with the table-of-contents page placeholders blanked,
  then reads back which PDF page each numbered section actually landed
  on. Pass 2 substitutes those real page numbers and re-renders. The
  substituted strings are 1-2 characters inside a fixed-size list, so
  pagination is unaffected between passes.

Rendering uses Playwright driving the system Chrome installation
(channel="chrome"), which avoids downloading a second browser. Page
numbering in the footer uses Chrome's own print templates, since CSS
`counter(page)` in @page margin boxes is not supported by Chromium.

Usage:
    venv\\Scripts\\python.exe docs/build_pdf.py
"""

from __future__ import annotations

import base64
import pathlib
import re
import sys

DOCS = pathlib.Path(__file__).resolve().parent
ROOT = DOCS.parent
PARTS = [DOCS / f"_part{i}.html" for i in range(1, 7)]
ASSEMBLED = DOCS / "technical_documentation.html"
PDF_OUT = DOCS / "Local_Voice_Chatbot_Technical_Documentation.pdf"
CHART = ROOT / "data" / "results" / "accuracy_comparison.png"

# (section number, heading text) -> the TOC placeholder(s) carrying its page.
# The number is part of the match because several of these titles also occur
# as ordinary prose or table headings elsewhere in the document (e.g. the word
# "Verification" is a column header in the requirements tables).
SECTION_PLACEHOLDERS = {
    (1, "Executive Summary"): ["__P1__"],
    (2, "Introduction"): ["__P2__", "__P2B__"],
    (3, "Product Overview"): ["__P3__", "__P3B__", "__P3C__"],
    (4, "Functional Requirements"): ["__P4__"],
    (5, "Non-Functional Requirements"): ["__P5__"],
    (6, "Use Cases"): ["__P6__"],
    (7, "System Architecture"): ["__P7__"],
    (8, "Detailed Design"): ["__P8__", "__P8B__"],
    (9, "Data Design"): ["__P9__"],
    (10, "Interface Specification"): ["__P10__"],
    (11, "Verification & Validation"): ["__P11__"],
    (12, "Experimental Evaluation"): ["__P12__", "__P12B__"],
    (13, "Performance Characteristics"): ["__P13__"],
    (14, "Security Considerations"): ["__P14__"],
    (15, "Ethical Considerations"): ["__P15__"],
    (16, "Operations & Deployment"): ["__P16__"],
    (17, "Known Limitations"): ["__P17__"],
    (18, "Future Work"): ["__P18__"],
    (19, "Appendices"): ["__P19__"],
}

FOOTER = """
<div style="width:100%;font-family:'Segoe UI',Arial,sans-serif;font-size:7pt;
            color:#7c8899;padding:0 17mm;display:flex;justify-content:space-between;
            border-top:0.5pt solid #d7dee6;padding-top:2mm;margin-top:4mm;">
  <span>LVC-FTS-001 &middot; v1.0 &middot; Local Voice Chatbot</span>
  <span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>
</div>
"""
HEADER = '<div style="display:none"></div>'


def assemble(chart_uri: str) -> str:
    html = "\n".join(p.read_text(encoding="utf-8") for p in PARTS)
    return html.replace("__CHART_DATA_URI__", chart_uri)


def chart_data_uri() -> str:
    if not CHART.exists():
        print(f"[build] WARNING: chart not found at {CHART}; figure will be blank.")
        return ""
    encoded = base64.b64encode(CHART.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render(html_path: pathlib.Path, pdf_path: pathlib.Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page()
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.emulate_media(media="print")
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template=HEADER,
            footer_template=FOOTER,
            margin={"top": "20mm", "bottom": "20mm", "left": "0mm", "right": "0mm"},
        )
        browser.close()


def section_pages(pdf_path: pathlib.Path) -> dict[tuple[int, str], int]:
    """Map each (number, heading) to the 1-based PDF page it starts on.

    Scanning begins after the table of contents, because the contents page
    itself lists every heading and would otherwise match all of them.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    flats = [re.sub(r"\s+", " ", (p.extract_text() or "")) for p in reader.pages]

    start = 0
    for index, flat in enumerate(flats):
        if "Table of Contents" in flat:
            start = index + 1  # keep advancing if the TOC spans pages

    found: dict[tuple[int, str], int] = {}
    for offset, flat in enumerate(flats[start:]):
        page_no = start + offset + 1  # 1-based
        for key in SECTION_PLACEHOLDERS:
            if key in found:
                continue
            number, title = key
            # The h1 renders as the number immediately followed by the title;
            # whitespace between them varies with text extraction.
            if re.search(rf"\b{number}\s*{re.escape(title)}", flat):
                found[key] = page_no
    return found


def main() -> int:
    uri = chart_data_uri()

    # ---- Pass 1: blank placeholders, discover pagination ----
    html = assemble(uri)
    pass1 = html
    for placeholders in SECTION_PLACEHOLDERS.values():
        for token in placeholders:
            pass1 = pass1.replace(token, "")
    ASSEMBLED.write_text(pass1, encoding="utf-8")
    print("[build] pass 1: rendering to discover pagination...")
    render(ASSEMBLED, PDF_OUT)

    pages = section_pages(PDF_OUT)
    print(f"[build] located {len(pages)}/{len(SECTION_PLACEHOLDERS)} sections")

    missing = [k for k in SECTION_PLACEHOLDERS if k not in pages]
    if missing:
        print(f"[build] WARNING: page not resolved for {missing}")

    # ---- Pass 2: substitute real page numbers ----
    final = html
    for key, placeholders in SECTION_PLACEHOLDERS.items():
        page_no = pages.get(key)
        for token in placeholders:
            final = final.replace(token, str(page_no) if page_no else "")
    ASSEMBLED.write_text(final, encoding="utf-8")
    print("[build] pass 2: rendering final document...")
    render(ASSEMBLED, PDF_OUT)

    from pypdf import PdfReader

    total = len(PdfReader(str(PDF_OUT)).pages)
    size_kb = PDF_OUT.stat().st_size / 1024
    print(f"[build] done: {PDF_OUT.name} — {total} pages, {size_kb:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
