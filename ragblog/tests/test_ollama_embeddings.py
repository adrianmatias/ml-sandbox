from unittest.mock import MagicMock, patch

import pytest
import requests

from src.const import CONST
from src.ollama_embeddings import OllamaEmbeddings


@pytest.fixture
def embeddings():
    return OllamaEmbeddings()


def test_init_defaults(embeddings):
    assert embeddings.model == CONST.model.emb
    assert embeddings.base_url == "http://127.0.0.1:11434"


def test_init_custom():
    custom = OllamaEmbeddings(model="custom-model", base_url="http://custom:11434")
    assert custom.model == "custom-model"
    assert custom.base_url == "http://custom:11434"


def test_init_strips_v1_suffix():
    custom = OllamaEmbeddings(base_url="http://127.0.0.1:11434/v1")
    assert custom.base_url == "http://127.0.0.1:11434"


@patch("src.ollama_embeddings.requests.post")
def test_embed_single(mock_post, embeddings):
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    result = embeddings._embed("test text")

    mock_post.assert_called_once_with(
        "http://127.0.0.1:11434/api/embeddings",
        json={"model": CONST.model.emb, "prompt": "test text"},
        timeout=60,
    )
    assert result == [0.1, 0.2, 0.3]


@patch("src.ollama_embeddings.requests.post")
def test_embed_documents(mock_post, embeddings):
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2]}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    texts = ["doc1", "doc2", "doc3"]
    result = embeddings.embed_documents(texts)

    assert len(result) == 3
    assert all(r == [0.1, 0.2] for r in result)
    assert mock_post.call_count == 3


@patch("src.ollama_embeddings.requests.post")
def test_embed_query(mock_post, embeddings):
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.5, 0.6, 0.7]}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    result = embeddings.embed_query("query text")

    mock_post.assert_called_once_with(
        "http://127.0.0.1:11434/api/embeddings",
        json={"model": CONST.model.emb, "prompt": "query text"},
        timeout=60,
    )
    assert result == [0.5, 0.6, 0.7]


@patch("src.ollama_embeddings.requests.post")
def test_embed_http_error(mock_post, embeddings):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("500 Error")
    mock_post.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        embeddings._embed("test")
