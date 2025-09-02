# use pytest frameworks to write test cases for adapters.py
import pytest
import torch
from sentence_transformers import SentenceTransformer

from src.sbert_adapter.adapters import LinearAdapter, TwoLayerAdapter


@pytest.fixture
def embedding_dim():
    base_model_name = "BAAI/bge-small-en"
    return SentenceTransformer(base_model_name).get_sentence_embedding_dimension()


def test_linear_adapter(embedding_dim):
    adapter = LinearAdapter(embedding_dim=embedding_dim, adapter_dim=embedding_dim, bias=False)
    assert adapter is not None
    assert adapter.adapter_dim == embedding_dim


def test_two_layer_adapter(embedding_dim):
    adapter = TwoLayerAdapter(embedding_dim=embedding_dim, hidden_dim=128, adapter_dim=embedding_dim, bias=False,
                              add_residual=True)
    assert adapter is not None
    assert adapter.embedding_dim == embedding_dim
    assert adapter.adapter_dim == embedding_dim
    assert adapter.add_residual is True
