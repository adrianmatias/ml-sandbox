from unittest.mock import MagicMock, patch

from src.rag import Rag, ThinkingOutputParser


def test_thinking_output_parser_strips_markdown_thinking():
    parser = ThinkingOutputParser()

    # Actual pattern from Qwen output
    text = """Thinking Process:

1. **Analyze the Request:**
   Some thinking here.

2. **Analyze the Context:**
   More thinking.

Based on the provided context, the answer is 42."""
    result = parser.parse(text)
    assert "Thinking Process:" not in result
    assert "Analyze the Request" not in result
    assert result.startswith("Based on the provided context")


def test_thinking_output_parser_strips_think_token():
    parser = ThinkingOutputParser()

    text_with_think_token = """<think>

</think>

The core thesis of the author is"""
    result = parser.parse(text_with_think_token)
    assert result == "The core thesis of the author is"


def test_thinking_output_parser_strips_unicode_tags():
    parser = ThinkingOutputParser()

    # Unicode thinking tags (U+16EE = ᛮ, U+16ED = ᛭)
    text_with_unicode = "\u16eeLet me think\u16edThe answer is 42."
    result = parser.parse(text_with_unicode)
    assert result == "The answer is 42."


def test_thinking_output_parser_keeps_thinking_content():
    parser = ThinkingOutputParser()

    # Generic  markers - content is kept
    text_with_generic = "Let me thinkThe answer is 42."
    result = parser.parse(text_with_generic)
    assert "Let me think" in result
    assert "The answer is 42" in result


def test_thinking_output_parser_handles_no_tags():
    parser = ThinkingOutputParser()

    text_without_thinking = "The answer is 42."
    result = parser.parse(text_without_thinking)
    assert result == "The answer is 42."


def test_thinking_output_parser_after_double_newline():
    parser = ThinkingOutputParser()

    text = "thinking the str\n\nThe answer is 42."
    result = parser.parse(text)
    assert result == "The answer is 42."


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
