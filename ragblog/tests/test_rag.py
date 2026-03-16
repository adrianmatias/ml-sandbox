from unittest.mock import MagicMock, patch

from src.const import CONST, LLM
from src.rag import Rag


@patch("src.rag.VectorDB")
@patch("src.rag.DocLoader")
@patch("src.rag.Crawler")
def test_init_index_exists(mock_crawler, mock_doc_loader, mock_vector_db):
    mock_vector_db_instance = MagicMock()
    mock_vector_db.return_value = mock_vector_db_instance
    mock_vector_db_instance.persist_directory.exists.return_value = True
    mock_vector_db_instance.get_vector_db.return_value = mock_vector_db_instance

    rag = Rag()

    mock_crawler.assert_not_called()
    mock_doc_loader.assert_not_called()
    mock_vector_db_instance.get_vector_db.assert_called_once_with(doc_list=None)
    assert rag.vector_db == mock_vector_db_instance


@patch("src.rag.VectorDB")
@patch("src.rag.DocLoader")
@patch("src.rag.Crawler")
def test_init_overwrite_index(mock_crawler, mock_doc_loader, mock_vector_db):
    mock_crawler_instance = MagicMock()
    mock_crawler.return_value = mock_crawler_instance
    mock_doc_loader_instance = MagicMock()
    mock_doc_loader.return_value = mock_doc_loader_instance
    mock_doc_loader_instance.load.return_value = "docs"
    mock_vector_db_instance = MagicMock()
    mock_vector_db.return_value = mock_vector_db_instance
    mock_vector_db_instance.get_vector_db.return_value = mock_vector_db_instance

    rag = Rag(is_overwrite_index=True)

    mock_crawler.assert_called_once_with(post_count_min=100)
    mock_crawler_instance.run.assert_called_once()
    mock_doc_loader.assert_called_once()
    mock_doc_loader_instance.load.assert_called_once()
    mock_vector_db_instance.get_vector_db.assert_called_once_with(doc_list="docs")
    assert rag.vector_db == mock_vector_db_instance


def test_format_docs():
    docs = [MagicMock()]
    docs[0].page_content = "content"

    result = Rag.format_docs(docs)

    assert "<|retrieved_doc|>" in result
    assert "content" in result


@patch("src.rag.VectorDB")
@patch("src.rag.ChatOpenAI")
def test_create_chain(mock_chat_openai, mock_vector_db):
    mock_vector_db_instance = MagicMock()
    mock_vector_db.return_value = mock_vector_db_instance
    mock_vector_db_instance.persist_directory.exists.return_value = True
    mock_vector_db_instance.get_vector_db.return_value = mock_vector_db_instance

    mock_llm_instance = MagicMock()
    mock_chat_openai.return_value = mock_llm_instance

    rag = Rag()
    _ = rag.chain

    mock_chat_openai.assert_called_once()
    call_kwargs = mock_chat_openai.call_args.kwargs
    assert call_kwargs["model"] == CONST.model.aug
    assert call_kwargs["base_url"] == CONST.api.base_url
    assert call_kwargs["api_key"] == CONST.api.api_key


@patch("src.rag.VectorDB")
def test_init_with_custom_aug(mock_vector_db):
    mock_vector_db_instance = MagicMock()
    mock_vector_db.return_value = mock_vector_db_instance
    mock_vector_db_instance.persist_directory.exists.return_value = True
    mock_vector_db_instance.get_vector_db.return_value = mock_vector_db_instance

    custom_aug = LLM.QWEN_2_5_14B
    rag = Rag(aug=custom_aug)

    assert rag.aug == custom_aug


@patch("src.rag.VectorDB")
def test_query(mock_vector_db):
    mock_vector_db_instance = MagicMock()
    mock_vector_db.return_value = mock_vector_db_instance
    mock_vector_db_instance.persist_directory.exists.return_value = True
    mock_vector_db_instance.get_vector_db.return_value = mock_vector_db_instance

    mock_retriever = MagicMock()
    mock_vector_db_instance.as_retriever.return_value = mock_retriever

    rag = Rag()

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "test response"
    rag.chain = mock_chain

    result = rag.query("test question")

    mock_chain.invoke.assert_called_once_with("test question")
    assert result == "test response"


@patch("src.rag.requests.post")
@patch("src.rag.VectorDB")
def test_query_fallback_to_reasoning_content(mock_vector_db, mock_post):
    mock_vector_db_instance = MagicMock()
    mock_vector_db.return_value = mock_vector_db_instance
    mock_vector_db_instance.persist_directory.exists.return_value = True
    mock_vector_db_instance.get_vector_db.return_value = mock_vector_db_instance

    mock_doc = MagicMock()
    mock_doc.page_content = "context A"
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc]
    mock_vector_db_instance.as_retriever.return_value = mock_retriever

    rag = Rag()
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = ""
    rag.chain = mock_chain

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": "fallback reasoning",
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    result = rag.query("test question")

    assert result == "fallback reasoning"
    mock_post.assert_called_once()


@patch("src.rag.VectorDB")
def test_get_contexts(mock_vector_db):
    mock_vector_db_instance = MagicMock()
    mock_vector_db.return_value = mock_vector_db_instance
    mock_vector_db_instance.persist_directory.exists.return_value = True
    mock_vector_db_instance.get_vector_db.return_value = mock_vector_db_instance

    mock_doc1 = MagicMock()
    mock_doc1.page_content = "context 1"
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "context 2"

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc1, mock_doc2]
    mock_vector_db_instance.as_retriever.return_value = mock_retriever

    rag = Rag()
    contexts = rag.get_contexts("test question")

    assert contexts == ["context 1", "context 2"]
    mock_retriever.invoke.assert_called_once_with("test question")
