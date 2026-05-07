"""Tests for Layer 1: Document Intelligence.

Tests cover:
- PDF parsing utilities (pdf_utils.py)
- Image pre-processing pipeline stub (image_processing.py)
- OCR utilities (ocr_utils.py)
- L1 document processing layer (l1_document.py)
"""

import os
import sqlite3
import tempfile
import uuid

import pytest
from PyPDF2 import PdfWriter

from backend.database.connection import get_db
from backend.database.schema import create_tables
from backend.layers.l1_document import process_document
from backend.utils.image_processing import preprocess_page_image
from backend.utils.ocr_utils import (
    compute_page_confidence,
    extract_text_from_image,
    is_degraded_page,
)
from backend.utils.pdf_utils import extract_page_images, parse_pdf


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_pdf():
    """Create a temporary blank PDF file."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    path = tempfile.mktemp(suffix=".pdf")
    with open(path, "wb") as f:
        writer.write(f)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def tmp_multi_page_pdf():
    """Create a temporary multi-page blank PDF file."""
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=612, height=792)
    path = tempfile.mktemp(suffix=".pdf")
    with open(path, "wb") as f:
        writer.write(f)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def db_conn():
    """Create a temporary database with schema for testing."""
    db_path = tempfile.mktemp(suffix=".db")
    conn = get_db(db_path)
    create_tables(conn)
    # Create a test tender
    tender_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO tenders (id, title, department, category, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (tender_id, "Test Tender", "Public Works", "Construction",
         "DOCUMENTS_UPLOADED", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    )
    conn.commit()
    yield conn, tender_id, db_path
    conn.close()
    if os.path.exists(db_path):
        os.unlink(db_path)


# ─── PDF Utils Tests ───────────────────────────────────────────────────────────


class TestParsePdf:
    def test_parses_valid_pdf(self, tmp_pdf):
        result = parse_pdf(tmp_pdf)
        assert result["page_count"] == 1
        assert len(result["pages"]) == 1
        assert result["pages"][0]["page_number"] == 1
        assert "error" not in result

    def test_detects_scanned_pdf(self, tmp_pdf):
        """A blank PDF with no text should be detected as scanned."""
        result = parse_pdf(tmp_pdf)
        assert result["is_scanned"] is True

    def test_multi_page_pdf(self, tmp_multi_page_pdf):
        result = parse_pdf(tmp_multi_page_pdf)
        assert result["page_count"] == 3
        assert len(result["pages"]) == 3
        for i, page in enumerate(result["pages"]):
            assert page["page_number"] == i + 1

    def test_missing_file_returns_error(self):
        result = parse_pdf("/tmp/nonexistent_file_12345.pdf")
        assert result["error"] is True
        assert "not found" in result["message"].lower()
        assert result["page_count"] == 0

    def test_corrupted_file_returns_error(self):
        """A non-PDF file should return an error."""
        path = tempfile.mktemp(suffix=".pdf")
        with open(path, "w") as f:
            f.write("This is not a PDF file")
        try:
            result = parse_pdf(path)
            assert result["error"] is True
            assert result["page_count"] == 0
        finally:
            os.unlink(path)


class TestExtractPageImages:
    def test_extracts_images_for_each_page(self, tmp_multi_page_pdf):
        output_dir = tempfile.mkdtemp()
        images = extract_page_images(tmp_multi_page_pdf, output_dir)
        assert len(images) == 3
        for img_path in images:
            assert os.path.exists(img_path)
            assert img_path.endswith(".png")
        # Clean up
        for img in images:
            os.unlink(img)
        os.rmdir(output_dir)

    def test_missing_file_returns_empty_list(self):
        result = extract_page_images("/tmp/nonexistent.pdf", "/tmp/output")
        assert result == []

    def test_creates_output_directory(self, tmp_pdf):
        output_dir = tempfile.mktemp()  # Does not exist yet
        images = extract_page_images(tmp_pdf, output_dir)
        assert len(images) == 1
        assert os.path.isdir(output_dir)
        # Clean up
        for img in images:
            os.unlink(img)
        os.rmdir(output_dir)


# ─── Image Processing Tests ───────────────────────────────────────────────────


class TestPreprocessPageImage:
    """Tests for the real OpenCV image preprocessing pipeline.

    Uses a real PNG rasterised from a demo PDF so the full five-step
    pipeline executes end-to-end (no stub behaviour).
    """

    @pytest.fixture
    def real_page_image(self, tmp_path):
        """Rasterise page 1 of a demo PDF to a real PNG."""
        from backend.utils.pdf_utils import extract_page_images
        demo_pdf = "backend/demo_data/sample_nit.pdf"
        images = extract_page_images(demo_pdf, str(tmp_path), dpi=200)
        assert images, "Demo PDF should produce at least one page image"
        return images[0]

    def test_returns_processed_image_path(self, real_page_image):
        result = preprocess_page_image(real_page_image)
        # Real pipeline writes a *_processed.png next to the input
        assert result["processed_image_path"].endswith("_processed.png")
        assert os.path.exists(result["processed_image_path"])

    def test_returns_300_dpi(self, real_page_image):
        result = preprocess_page_image(real_page_image)
        assert result["dpi"] == 300

    def test_returns_five_steps(self, real_page_image):
        result = preprocess_page_image(real_page_image)
        assert len(result["steps_applied"]) == 5
        assert "dpi_normalisation" in result["steps_applied"]
        assert "deskew" in result["steps_applied"]
        assert "binarisation" in result["steps_applied"]
        assert "stamp_separation" in result["steps_applied"]
        assert "denoising" in result["steps_applied"]

    def test_returns_processing_notes(self, real_page_image):
        result = preprocess_page_image(real_page_image)
        # Real notes describe what actually happened
        assert "skew_corrected=" in result["processing_notes"]
        assert "stamps_detected=" in result["processing_notes"]
        assert "steps=" in result["processing_notes"]

    def test_not_a_stub(self, real_page_image):
        result = preprocess_page_image(real_page_image)
        assert result["is_stub"] is False

    def test_missing_file_returns_error(self):
        result = preprocess_page_image("/some/path/nonexistent.png")
        assert result.get("error") is True
        assert any("file_not_found" in w for w in result["warnings"])


# ─── OCR Utils Tests ──────────────────────────────────────────────────────────


class TestExtractTextFromImage:
    """Tests for real Tesseract OCR via pytesseract."""

    @pytest.fixture
    def real_page_image(self, tmp_path):
        """Rasterise page 1 of a demo PDF to a real PNG."""
        from backend.utils.pdf_utils import extract_page_images
        demo_pdf = "backend/demo_data/sample_nit.pdf"
        images = extract_page_images(demo_pdf, str(tmp_path), dpi=300)
        assert images, "Demo PDF should produce at least one page image"
        return images[0]

    def test_returns_words_list(self, real_page_image):
        result = extract_text_from_image(real_page_image)
        assert "words" in result
        # Real NIT page should have plenty of words
        assert len(result["words"]) > 20

    def test_words_have_required_fields(self, real_page_image):
        result = extract_text_from_image(real_page_image)
        for word in result["words"]:
            assert "id" in word
            assert "text_content" in word
            assert "x_min" in word
            assert "y_min" in word
            assert "x_max" in word
            assert "y_max" in word
            assert "confidence" in word
            assert "source_engine" in word

    def test_confidence_in_valid_range(self, real_page_image):
        result = extract_text_from_image(real_page_image)
        for word in result["words"]:
            assert 0.0 <= word["confidence"] <= 1.0

    def test_bounding_boxes_valid(self, real_page_image):
        result = extract_text_from_image(real_page_image)
        for word in result["words"]:
            assert word["x_max"] > word["x_min"]
            assert word["y_max"] > word["y_min"]

    def test_raw_text_is_concatenation(self, real_page_image):
        result = extract_text_from_image(real_page_image)
        expected = " ".join(w["text_content"] for w in result["words"])
        assert result["raw_text"] == expected

    def test_not_a_stub(self, real_page_image):
        result = extract_text_from_image(real_page_image)
        assert result["is_stub"] is False
        assert "tesseract" in result["engine"].lower()

    def test_missing_file_returns_error(self):
        result = extract_text_from_image("/tmp/nonexistent_image.png")
        assert result["words"] == []
        assert result["raw_text"] == ""
        assert "error" in result

    def test_extracts_real_tender_keywords(self, real_page_image):
        """Real NIT should contain recognisable keywords."""
        result = extract_text_from_image(real_page_image)
        text_lower = result["raw_text"].lower()
        # At least one of these tender-specific words should appear
        keywords = ["tender", "bidder", "crore", "eligibility", "gst", "turnover"]
        matches = sum(1 for kw in keywords if kw in text_lower)
        assert matches >= 2, (
            f"Expected at least 2 tender keywords in OCR output, found {matches}. "
            f"Text sample: {result['raw_text'][:200]}"
        )


class TestComputePageConfidence:
    def test_equal_length_words(self):
        words = [
            {"text_content": "hello", "confidence": 0.9},
            {"text_content": "world", "confidence": 0.8},
        ]
        conf = compute_page_confidence(words)
        assert abs(conf - 0.85) < 0.001

    def test_length_weighted(self):
        words = [
            {"text_content": "hi", "confidence": 0.5},       # weight 2
            {"text_content": "longword", "confidence": 0.9},  # weight 8
        ]
        # Expected: (2*0.5 + 8*0.9) / (2+8) = (1.0 + 7.2) / 10 = 0.82
        conf = compute_page_confidence(words)
        assert abs(conf - 0.82) < 0.001

    def test_empty_list_returns_zero(self):
        assert compute_page_confidence([]) == 0.0

    def test_single_word(self):
        words = [{"text_content": "test", "confidence": 0.75}]
        assert abs(compute_page_confidence(words) - 0.75) < 0.001

    def test_result_in_valid_range(self):
        words = [
            {"text_content": "a", "confidence": 0.0},
            {"text_content": "b", "confidence": 1.0},
        ]
        conf = compute_page_confidence(words)
        assert 0.0 <= conf <= 1.0


class TestIsDegradedPage:
    def test_below_threshold(self):
        assert is_degraded_page(0.49) is True
        assert is_degraded_page(0.0) is True
        assert is_degraded_page(0.30) is True

    def test_at_threshold(self):
        assert is_degraded_page(0.50) is False

    def test_above_threshold(self):
        assert is_degraded_page(0.51) is False
        assert is_degraded_page(0.95) is False
        assert is_degraded_page(1.0) is False


# ─── L1 Document Layer Tests ──────────────────────────────────────────────────


class TestProcessDocument:
    """End-to-end L1 pipeline tests against real demo PDFs.

    The `tmp_pdf` fixture creates a blank PDF which correctly yields
    zero words from real OCR; these tests therefore use the real
    `backend/demo_data/sample_nit.pdf` fixture instead.
    """

    @pytest.fixture
    def real_nit_pdf(self, tmp_path):
        """Copy the real NIT into a tmp directory so uploads can be written."""
        import shutil
        src = "backend/demo_data/sample_nit.pdf"
        dst = tmp_path / "sample_nit.pdf"
        shutil.copy(src, dst)
        return str(dst)

    def test_processes_valid_pdf(self, db_conn, real_nit_pdf):
        conn, tender_id, _ = db_conn
        result = process_document(conn, tender_id, real_nit_pdf, "nit")
        conn.commit()

        assert result["processing_status"] == "complete"
        assert result["page_count"] >= 1
        assert result["avg_ocr_confidence"] > 0
        assert result["sha256_hash"] is not None
        assert len(result["sha256_hash"]) == 64

    def test_stores_document_in_database(self, db_conn, real_nit_pdf):
        conn, tender_id, _ = db_conn
        result = process_document(conn, tender_id, real_nit_pdf, "nit")
        conn.commit()

        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (result["id"],)
        ).fetchone()
        assert row is not None
        assert row["processing_status"] == "complete"
        assert row["page_count"] >= 1

    def test_stores_pages_in_database(self, db_conn, tmp_multi_page_pdf):
        conn, tender_id, _ = db_conn
        result = process_document(conn, tender_id, tmp_multi_page_pdf, "nit")
        conn.commit()

        pages = conn.execute(
            "SELECT * FROM pages WHERE document_id = ? ORDER BY page_number",
            (result["id"],),
        ).fetchall()
        assert len(pages) == 3
        for i, page in enumerate(pages):
            assert page["page_number"] == i + 1

    def test_stores_word_objects(self, db_conn, real_nit_pdf):
        conn, tender_id, _ = db_conn
        result = process_document(conn, tender_id, real_nit_pdf, "nit")
        conn.commit()

        word_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM word_objects WHERE page_id IN "
            "(SELECT id FROM pages WHERE document_id = ?)",
            (result["id"],),
        ).fetchone()["cnt"]
        # Real OCR on a real NIT should produce many words
        assert word_count > 50

    def test_logs_audit_events(self, db_conn, tmp_pdf):
        conn, tender_id, _ = db_conn
        process_document(conn, tender_id, tmp_pdf, "nit")
        conn.commit()

        events = conn.execute(
            "SELECT event_type FROM audit_events WHERE tender_id = ? ORDER BY id",
            (tender_id,),
        ).fetchall()
        event_types = [e["event_type"] for e in events]
        assert "document_received" in event_types
        assert "ocr_completed" in event_types

    def test_handles_missing_file(self, db_conn):
        conn, tender_id, _ = db_conn
        # This will fail at compute_file_hash since file doesn't exist
        with pytest.raises(FileNotFoundError):
            process_document(conn, tender_id, "/tmp/nonexistent.pdf", "nit")

    def test_bidder_id_stored(self, db_conn, tmp_pdf):
        conn, tender_id, _ = db_conn
        bidder_id = str(uuid.uuid4())
        # Create bidder record first
        conn.execute(
            "INSERT INTO bidders (id, tender_id, company_name, status, debarment_status) "
            "VALUES (?, ?, ?, ?, ?)",
            (bidder_id, tender_id, "Test Corp", "pending", "clear"),
        )
        result = process_document(
            conn, tender_id, tmp_pdf, "bidder_submission", bidder_id=bidder_id
        )
        conn.commit()

        assert result["bidder_id"] == bidder_id
        row = conn.execute(
            "SELECT bidder_id FROM documents WHERE id = ?", (result["id"],)
        ).fetchone()
        assert row["bidder_id"] == bidder_id
