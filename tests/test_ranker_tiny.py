"""Ranker end to end on a tiny randomly initialised Qwen2 model: real tokenizer and offsets, no weights download."""
import dataclasses

import pytest
import torch

from aethis_helm.config import ModelConfig, VerbalizeConfig
from aethis_helm.model import PromptTooLongError, SpanActionRankModel, SpanScoringHead
from aethis_helm.ranker import Ranker
from aethis_helm.types import Step, ToolSpec

BACKBONE = ModelConfig().backbone


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(BACKBONE, local_files_only=True)
    except OSError:
        pytest.skip("backbone tokenizer not cached locally")


def _ranker(tokenizer, max_prompt_tokens: int = 1280) -> Ranker:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=2048)
    backbone = Qwen2ForCausalLM(config).eval()
    head = SpanScoringHead(hidden_dim=16, head_hidden=8, temperature=0.05)
    cfg = dataclasses.replace(ModelConfig(), max_prompt_tokens=max_prompt_tokens)
    return Ranker(SpanActionRankModel(tokenizer, backbone, head, cfg, VerbalizeConfig()))


TOOLS = [ToolSpec("get_weather", "Weather for a city"), ToolSpec("search_flights", "Find flights"),
         ToolSpec("Finish", "Stop and answer")]


def test_rank_scores_every_tool_once_best_first(tokenizer):
    ranked = _ranker(tokenizer).rank("Plan a trip", [Step("get_weather", "{}", "sunny")], TOOLS)
    assert sorted(r.name for r in ranked) == sorted(t.name for t in TOOLS)
    assert [r.score for r in ranked] == sorted((r.score for r in ranked), reverse=True)
    assert abs(sum(r.prob for r in ranked) - 1.0) < 1e-5


def test_rank_is_deterministic(tokenizer):
    ranker = _ranker(tokenizer)
    assert ranker.rank("Plan a trip", [], TOOLS) == ranker.rank("Plan a trip", [], TOOLS)


def test_long_task_is_truncated_silently_while_tools_fit(tokenizer):
    ranked = _ranker(tokenizer, max_prompt_tokens=64).rank("word " * 500, [], TOOLS)
    assert len(ranked) == 3


def test_prompt_that_would_cut_a_tool_line_raises(tokenizer):
    tools = [ToolSpec(f"tool_{i}", "a fairly long description of what this tool does " * 4) for i in range(6)]
    with pytest.raises(PromptTooLongError):
        _ranker(tokenizer, max_prompt_tokens=64).rank("task", [], tools)
