"""Tests for Granite language detection and embedding services.

These tests primarily mock external model calls to avoid heavy downloads and
network reliance. An optional integration test is provided that will run only
when HF_TOKEN is set in the environment.
"""

import os
import numpy as np
import pytest

from services.granite_service import GraniteLanguageDetectionService
from services.granite_embedding_service import GraniteEmbeddingService


class DummyPipeline:
    """Simple dummy pipeline to simulate transformers pipeline output."""
    def __call__(self, text):
        # Return a list with a single prediction dict similar to transformers output
        return [{"label": "en", "score": 0.95}]


def test_language_detection_ml_path(monkeypatch):
    """Language detection should use ML path when pipeline is available."""
    service = GraniteLanguageDetectionService()

    # Monkeypatch the language detector with our dummy pipeline
    service.language_detector = DummyPipeline()

    result = service.detect_language("This is a test sentence in English.")

    assert result["detected_language"] == "english"
    assert 0.9 <= result["confidence"] <= 1.0
    assert result["method"] == "ml_transformer"
    assert result["supported"] is True


def test_language_detection_rule_based_fallback():
    """Fallback rule-based detection should work when pipeline is unavailable."""
    service = GraniteLanguageDetectionService()

    # Force fallback by removing the language detector
    service.language_detector = None

    hindi_text = "मरीज़ का नाम: राम शर्मा. निदान: मधुमेह."
    result = service.detect_language(hindi_text)

    assert result["detected_language"] in {"hindi", "marathi", "english"}
    # For our sample, hindi indicators should be present
    assert result["detected_language"] == "hindi"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["method"] in {"rule_based", "fallback_default"}


def test_generate_embeddings_with_mock(monkeypatch):
    """Embedding generation should work with mocked sentence transformer backend."""
    service = GraniteEmbeddingService()

    # Mock sentence transformer backend to avoid heavy downloads
    class DummyST:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32):
            # Return deterministic embeddings: shape (len(texts), 384)
            rng = np.random.default_rng(42)
            embs = rng.normal(size=(len(texts), 384)).astype(np.float32)
            if normalize_embeddings:
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1e-8, norms)
                embs = embs / norms
            return embs

        def get_sentence_embedding_dimension(self):
            return 384

    # Ensure we use the sentence transformer code path
    service.model = None
    service.tokenizer = None
    service.sentence_transformer = DummyST()

    texts = ["hello world", "granite embeddings test"]
    embs = service.generate_embeddings(texts)

    assert isinstance(embs, np.ndarray)
    assert embs.shape == (2, 384)
    # Check normalized rows (approx 1.0)
    row_norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(row_norms, 1.0, atol=1e-3)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="HF_TOKEN not set; skipping live model integration test")
def test_live_embedding_generation_small_input():
    """Optional live test: run a tiny embedding through whichever backend is available."""
    service = GraniteEmbeddingService()
    texts = ["short text for embedding"]
    embs = service.generate_embeddings(texts, normalize=True)

    assert isinstance(embs, np.ndarray)
    assert embs.shape[0] == 1
    # Dimension may vary by backend, but should be > 0
    assert embs.shape[1] > 0
