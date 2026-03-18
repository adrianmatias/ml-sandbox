"""Unit tests for llama_server_manager and model_comparator."""

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.const import LLM
from src.llama_server_manager import ConfServer, LlamaServerManager


class TestConfServer:
    def test_default_path(self):
        conf = ConfServer()
        assert "llama-server" in str(conf.path_bin)

    def test_custom_path_via_env(self, monkeypatch):
        monkeypatch.setenv("LLAMA_SERVER_PATH", "/custom/path/llama-server")
        conf = ConfServer()
        assert conf.path_bin == Path("/custom/path/llama-server")


class TestLlamaServerManager:
    def test_init_defaults(self):
        manager = LlamaServerManager()
        assert manager.conf.port == 8080
        assert manager.conf.ctx_size == 32768
        assert manager.process_active is None

    def test_model_hf_mapping(self):
        manager = LlamaServerManager()
        assert (
            manager.map_model_hf[LLM.QWEN_3_5_9B_Q8] == "unsloth/Qwen3.5-9B-GGUF:Q8_0"
        )
        assert manager.map_model_hf[LLM.GPT_OSS_20B] == "unsloth/gpt-oss-20b-GGUF:Q8_0"
        assert (
            manager.map_model_hf[LLM.QWEN_3_5_27B_Q3]
            == "unsloth/Qwen3.5-27B-GGUF:Q3_K_S"
        )

    @patch("subprocess.Popen")
    @patch("time.sleep")
    def test_start_calls_kill_port(self, mock_sleep, mock_popen):
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        manager = LlamaServerManager()
        manager._kill_port = MagicMock()

        manager.start(LLM.QWEN_3_5_9B_Q8)

        manager._kill_port.assert_called_once_with(8080)

    @patch("subprocess.run")
    def test_kill_port_finds_process(self, mock_run):
        mock_run.return_value = MagicMock(stdout="12345\n")

        manager = LlamaServerManager()
        with patch("os.kill") as mock_kill:
            manager._kill_port(8080)
            mock_kill.assert_called()


class TestModelComparator:
    def test_aggregate_scores(self):
        from src.model_comparator import ModelComparator

        # Create mock comparator
        comparator = ModelComparator.__new__(ModelComparator)
        comparator.eval_set = None
        comparator.server_manager = None

        result_list = [
            {"context_precision": 0.8, "faithfulness": 0.9, "answer_relevancy": 0.7},
            {"context_precision": 0.6, "faithfulness": 0.7, "answer_relevancy": 0.5},
        ]

        scores = comparator.aggregate_scores(result_list)

        assert scores["context_precision"] == 0.7
        assert scores["faithfulness"] == 0.8
        assert scores["answer_relevancy"] == 0.6

    def test_to_dataframe(self):
        from src.model_comparator import ComparisonResult, ModelComparator

        comparator = ModelComparator.__new__(ModelComparator)
        comparator.comparison_map = {
            LLM.QWEN_3_5_9B_Q8: ComparisonResult(
                model=LLM.QWEN_3_5_9B_Q8,
                result_list=[],
                score_map={"context_precision": 0.8, "faithfulness": 0.9},
            ),
            LLM.GPT_OSS_20B: ComparisonResult(
                model=LLM.GPT_OSS_20B,
                result_list=[],
                score_map={"context_precision": 0.7, "faithfulness": 0.85},
            ),
        }

        df = comparator.to_dataframe()

        assert len(df) == 2
        assert "context_precision" in df.columns
        assert "faithfulness" in df.columns
