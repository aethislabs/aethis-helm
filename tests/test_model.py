import pytest
import torch

from aethis_helm.model import (PromptTooLongError, SpanScoringHead, build_candidate_mask, check_spans_kept,
                               load_span_head, pad_tool_vectors, pool_hidden, pool_spans, rank_logits,
                               resolve_device, save_span_head, span_token_mask)
from aethis_helm.types import Catalog, Example, ToolSpec


def test_mean_pool_respects_mask():
    h = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    m = torch.tensor([[1, 1, 0]])
    assert torch.allclose(pool_hidden(h, m, "mean"), torch.tensor([[2.0, 2.0]]))


def test_last_pool_picks_last_real_token_with_right_padding():
    h = torch.tensor([[[1.0], [3.0], [100.0]]])
    m = torch.tensor([[1, 1, 0]])
    assert pool_hidden(h, m, "last").item() == 3.0


def test_pool_returns_float32_even_for_half_input():
    h = torch.randn(2, 4, 8).half()
    assert pool_hidden(h, torch.ones(2, 4, dtype=torch.long), "last").dtype == torch.float32


def test_pool_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        pool_hidden(torch.zeros(1, 1, 1), torch.ones(1, 1), "max")


def test_resolve_device_cpu_always_ok():
    assert resolve_device("cpu").type == "cpu"


@pytest.mark.skipif(torch.cuda.is_available(), reason="cuda present")
def test_resolve_device_raises_when_unavailable():
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device("cuda")


def test_candidate_mask():
    cat = Catalog((ToolSpec("a", ""), ToolSpec("b", ""), ToolSpec("Finish", "")))
    ex = [Example("q", (), ("b", "Finish")), Example("q", (), ("a",))]
    assert build_candidate_mask(ex, cat).tolist() == [[False, True, True], [True, False, False]]


def test_rank_logits_excludes_masked_tools_when_fewer_than_k():
    logits = torch.tensor([[0.5, float("-inf"), 2.0, float("-inf")]])
    assert rank_logits(logits, k=5) == [(2, 0)]


def test_span_token_mask_marks_overlapping_tokens_only():
    offsets = torch.tensor([[0, 3], [3, 6], [6, 9], [9, 12], [12, 12]])  # last is padding
    mask = span_token_mask(offsets, ((0, 6), (7, 11)))
    assert mask.tolist() == [[True, True, False, False, False], [False, False, True, True, False]]


def test_span_token_mask_raises_on_zero_token_span():
    with pytest.raises(PromptTooLongError, match="zero tokens"):
        span_token_mask(torch.tensor([[0, 3], [3, 6]]), ((0, 3), (6, 9)))


def test_check_spans_kept_accepts_untruncated_prompt():
    check_spans_kept(torch.tensor([[0, 4], [4, 8], [8, 12], [0, 0]]), ((4, 8), (8, 12)))


def test_check_spans_kept_rejects_cut_first_tool_line():
    offsets = torch.tensor([[6, 9], [9, 12], [0, 0]])  # truncation dropped characters [0, 6)
    with pytest.raises(PromptTooLongError, match="too long"):
        check_spans_kept(offsets, ((4, 8), (8, 12)))


def test_pool_spans_means_over_masked_tokens():
    hidden = torch.tensor([[1.0, 0.0], [3.0, 0.0], [10.0, 5.0]])
    mask = torch.tensor([[True, True, False], [False, False, True]])
    assert torch.allclose(pool_spans(hidden, mask), torch.tensor([[2.0, 0.0], [10.0, 5.0]]))


def test_span_head_scatters_to_catalog_width_and_starts_as_cosine():
    head = SpanScoringHead(hidden_dim=4, head_hidden=8, temperature=0.5)
    q = torch.randn(2, 4)
    tools = torch.randn(2, 3, 4)
    tool_mask = torch.tensor([[True, True, False], [True, True, True]])
    tool_idx = torch.tensor([[5, 1, 0], [2, 7, 3]])
    logits = head(q, tools, tool_mask, tool_idx, num_tools=8)
    assert logits.shape == (2, 8)
    assert torch.isfinite(logits[0]).tolist() == [False, True, False, False, False, True, False, False]
    assert torch.isfinite(logits[1]).tolist() == [False, False, True, True, False, False, False, True]
    cos = torch.nn.functional.cosine_similarity(q[0], tools[0, 1], dim=0) / 0.5
    assert torch.allclose(logits[0, 1], cos, atol=1e-5)


def test_span_head_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        SpanScoringHead(hidden_dim=4, head_hidden=8, temperature=0.0)


def test_pad_tool_vectors_aligns_with_catalog_indices():
    cat = Catalog((ToolSpec("a", ""), ToolSpec("b", ""), ToolSpec("c", "")))
    ex = [Example("q", (), ("c", "a")), Example("q", (), ("b", "c", "a"))]
    padded, mask, idx = pad_tool_vectors([torch.ones(2, 4), torch.ones(3, 4)], ex, cat)
    assert padded.shape == (2, 3, 4)
    assert mask.tolist() == [[True, True, False], [True, True, True]]
    assert idx.tolist() == [[2, 0, 0], [1, 2, 0]]


def test_span_head_roundtrip(tmp_path):
    head = SpanScoringHead(hidden_dim=4, head_hidden=8, temperature=0.5)
    save_span_head(head, tmp_path / "s.pt")
    loaded = load_span_head(tmp_path / "s.pt")
    assert isinstance(loaded, SpanScoringHead) and loaded.temperature == 0.5
    for a, b in zip(head.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)


def test_research_checkpoint_with_catalog_fingerprint_loads(tmp_path):
    head = SpanScoringHead(hidden_dim=4, head_hidden=8, temperature=0.05)
    torch.save({"state_dict": head.state_dict(), "hidden_dim": 4, "head_hidden": 8, "temperature": 0.05,
                "catalog_sha": "0" * 64, "architecture": "span-cosine-v1"}, tmp_path / "head.pt")
    assert load_span_head(tmp_path / "head.pt").hidden_dim == 4


def test_load_span_head_rejects_other_architecture(tmp_path):
    head = SpanScoringHead(hidden_dim=4, head_hidden=8, temperature=0.5)
    torch.save({"state_dict": head.state_dict(), "hidden_dim": 4, "head_hidden": 8, "temperature": 0.5,
                "architecture": "residual-cosine-v1"}, tmp_path / "h.pt")
    with pytest.raises(ValueError, match="architecture"):
        load_span_head(tmp_path / "h.pt")
