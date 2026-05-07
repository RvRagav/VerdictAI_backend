"""Unit tests for backend.utils.hash_utils module."""

import hashlib
import json
import os
import tempfile

import pytest

from backend.utils.hash_utils import compute_entry_hash, compute_file_hash


class TestComputeEntryHash:
    """Tests for compute_entry_hash function."""

    def test_returns_64_char_hex_string(self):
        result = compute_entry_hash("test_event", {"key": "value"}, "system", "2024-01-01T00:00:00Z", "0" * 64)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic_output(self):
        """Same inputs always produce the same hash."""
        args = ("document_received", {"doc_id": "abc"}, "officer_1", "2024-06-15T10:30:00Z", "a" * 64)
        hash1 = compute_entry_hash(*args)
        hash2 = compute_entry_hash(*args)
        assert hash1 == hash2

    def test_different_inputs_produce_different_hashes(self):
        base_args = ("document_received", {"doc_id": "abc"}, "system", "2024-01-01T00:00:00Z", "0" * 64)
        hash1 = compute_entry_hash(*base_args)
        # Change event_type
        hash2 = compute_entry_hash("ocr_completed", {"doc_id": "abc"}, "system", "2024-01-01T00:00:00Z", "0" * 64)
        assert hash1 != hash2

    def test_dict_key_order_does_not_affect_hash(self):
        """Event data with different insertion order should produce same hash."""
        data1 = {"b": 2, "a": 1}
        data2 = {"a": 1, "b": 2}
        hash1 = compute_entry_hash("test", data1, "system", "2024-01-01T00:00:00Z", "0" * 64)
        hash2 = compute_entry_hash("test", data2, "system", "2024-01-01T00:00:00Z", "0" * 64)
        assert hash1 == hash2

    def test_matches_manual_computation(self):
        """Verify hash matches manual SHA-256 of deterministic JSON."""
        event_type = "schema_approved"
        event_data = {"officer_id": "off1"}
        actor = "off1"
        timestamp = "2024-03-01T12:00:00Z"
        prev_hash = "f" * 64

        payload = json.dumps({
            "event_type": event_type,
            "event_data": event_data,
            "actor": actor,
            "timestamp": timestamp,
            "prev_hash": prev_hash,
        }, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        result = compute_entry_hash(event_type, event_data, actor, timestamp, prev_hash)
        assert result == expected


class TestComputeFileHash:
    """Tests for compute_file_hash function."""

    def test_returns_64_char_hex_string(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello world")
            f.flush()
            path = f.name
        try:
            result = compute_file_hash(path)
            assert len(result) == 64
            assert all(c in "0123456789abcdef" for c in result)
        finally:
            os.unlink(path)

    def test_matches_known_hash(self):
        """SHA-256 of 'hello world' is well-known."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello world")
            f.flush()
            path = f.name
        try:
            result = compute_file_hash(path)
            expected = hashlib.sha256(b"hello world").hexdigest()
            assert result == expected
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            path = f.name
        try:
            result = compute_file_hash(path)
            expected = hashlib.sha256(b"").hexdigest()
            assert result == expected
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            compute_file_hash("/nonexistent/path/file.txt")
