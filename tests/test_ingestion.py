import http.server
import threading

import pytest

from carag.ingestion import IngestionError, html_to_document, load_bytes, load_file, load_url, validate_url

HTML = """<html><head><title>Mars facts</title></head><body>
<nav><a href="/">Home</a> Jump to content. Main menu navigation links here.</nav>
<header>Site header with a login button and search box</header>
<main>
  <h1>Mars</h1>
  <p>Mars is the fourth planet from the Sun.[1] It has two small moons called Phobos and Deimos.</p>
  <h2>Atmosphere</h2>
  <ul><li>The atmosphere of Mars is mostly carbon dioxide.</li></ul>
  <div class="reflist"><p>Smith, J. (2020). A reference entry. doi:10.1000/xyz.</p></div>
</main>
<footer>Copyright notice and privacy policy links.</footer>
</body></html>"""


def test_pdf_ingestion_tracks_pages_and_sections(test_pdf):
    doc = load_file(test_pdf)
    assert doc.pages == 3 and doc.source_type == "pdf"
    by_text = {u.text: u for u in doc.units}
    unit = next(u for t, u in by_text.items() if "118 milliwatt-hours" in t)
    assert unit.page == 2
    assert unit.section == "3. Results"
    assert unit.text == "Trial Two consumed an average of 118 milliwatt-hours per day."
    limitation = next(u for t, u in by_text.items() if "only one site" in t)
    assert limitation.page == 3 and limitation.section == "5. Limitations"
    # Identifiers are unique and encode page + position.
    assert len({u.evidence_id for u in doc.units}) == len(doc.units)
    assert unit.evidence_id.startswith("claim_aware_rag_test_document.pdf#p2.s")
    # Headings are not evidence sentences.
    assert all(u.text != "3. Results" for u in doc.units)


def test_txt_ingestion_and_dedup():
    data = b"Results\nThe battery lasted 71 days in total.\nThe battery lasted 71 days in total.\n\nOk.\n"
    doc = load_bytes(data, "notes.txt")
    assert [u.text for u in doc.units] == ["The battery lasted 71 days in total."]
    assert doc.units[0].section == "Results"
    assert doc.dropped_units == 2   # one duplicate, one fragment


def test_docx_ingestion(tmp_path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Findings", level=1)
    document.add_paragraph("The sensor recorded 12 pollution spikes during the trial.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Trial Two"
    table.rows[0].cells[1].text = "10 Hz sampling"
    path = tmp_path / "report.docx"
    document.save(path)
    doc = load_file(path)
    assert doc.units[0].text == "The sensor recorded 12 pollution spikes during the trial."
    assert doc.units[0].section == "Findings"
    assert any("Trial Two; 10 Hz sampling" in u.text for u in doc.units)


def test_html_extraction_skips_navigation_and_references():
    doc = html_to_document(HTML, "https://example.org/mars")
    texts = [u.text for u in doc.units]
    assert "Mars is the fourth planet from the Sun." in texts
    assert "The atmosphere of Mars is mostly carbon dioxide." in texts
    assert not any("menu" in t.lower() or "doi" in t.lower() or "privacy" in t.lower() for t in texts)
    atmosphere = next(u for u in doc.units if "carbon dioxide" in u.text)
    assert atmosphere.section == "Atmosphere" and atmosphere.url == "https://example.org/mars"
    assert atmosphere.citation() == "https://example.org/mars"
    assert doc.title == "Mars facts"


def test_url_ingestion_over_http():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://localhost:{server.server_port}/mars"
        doc = load_url(url)
        assert doc.source_type == "url" and len(doc.units) == 3
    finally:
        server.shutdown()


@pytest.mark.parametrize("bad", ["not a url", "ftp://example.org/x", "http://", "https://nodot"])
def test_invalid_urls_rejected(bad):
    with pytest.raises(IngestionError):
        validate_url(bad)


def test_unreachable_url_raises_ingestion_error():
    with pytest.raises(IngestionError):
        load_url("http://localhost:9/nothing-here")


def test_empty_and_unsupported_files(tmp_path):
    with pytest.raises(IngestionError, match="empty"):
        load_bytes(b"", "empty.txt")
    with pytest.raises(IngestionError, match="Unsupported"):
        load_bytes(b"data", "image.png")
    with pytest.raises(IngestionError, match="No extractable text"):
        load_bytes(b"   \n  ", "blank.txt")
    with pytest.raises(IngestionError, match="not a readable PDF"):
        load_bytes(b"this is not a pdf", "fake.pdf")
    with pytest.raises(IngestionError, match="not found"):
        load_file(tmp_path / "missing.pdf")
