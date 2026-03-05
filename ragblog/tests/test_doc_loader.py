from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.doc_loader import DocLoader


@pytest.fixture
def doc_loader():
    return DocLoader()


def test_load(doc_loader):
    mock_docs = [Document(page_content="test content")]

    with patch("src.doc_loader.JSONLoader") as mock_json_loader:
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = mock_docs
        mock_json_loader.return_value = mock_loader_instance

        result = doc_loader.load()

        assert len(result) == 1
        assert result[0].page_content == "test content"
        mock_json_loader.assert_called_once()
        mock_loader_instance.load.assert_called_once()


def test_split(doc_loader):
    long_content = "This is a long document. " * 200  # Make it long enough to split
    docs = [Document(page_content=long_content)]

    result = doc_loader.split(doc_list=docs)

    assert len(result) > 1  # Should split into multiple chunks
    assert all(isinstance(doc, Document) for doc in result)
    assert all(len(doc.page_content) <= 1000 for doc in result)
