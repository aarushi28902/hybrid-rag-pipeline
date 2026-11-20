import pytest
from src.rag.config import get_settings


def test_settings_load():
    settings = get_settings()
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.top_k == 6


def test_settings_are_cached():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2   # same object = lru_cache working


def test_reranking_defaults():
    settings = get_settings()
    assert settings.search_mode == "hybrid"
    assert settings.rerank_enabled is True
    assert settings.rerank_candidates == 30
    assert 0 < settings.abstain_threshold < 1
