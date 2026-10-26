"""Backbone (prefill only), span pooling, and the span scoring head.

Each candidate tool is represented by the hidden states over its own line inside the prompt, so the head has
no per-tool table and works for any tool list."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from .config import ModelConfig, VerbalizeConfig
from .types import Catalog, Example
from .verbalize import build_prompt_with_spans

log = logging.getLogger(__name__)

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
SPAN_HEAD_ARCHITECTURE = "span-cosine-v1"


class PromptTooLongError(ValueError):
    """The prompt exceeds max_prompt_tokens far enough that left truncation would cut a tool's line."""


def resolve_device(requested: str) -> torch.device:
    """Return the requested device, or raise if it is not available on this machine."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError(f"device {requested!r} is not available here; set HELM_DEVICE=cpu or fix the runtime")


def synchronize(device: torch.device) -> None:
    """Block until queued kernels finish (needed for honest latency timing)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_backbone(cfg: ModelConfig):
    """Load tokenizer + causal LM in cfg.dtype on cfg.device, eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.backbone, dtype=_DTYPES[cfg.dtype])
    model.to(device)  # pyright: ignore[reportArgumentType]  (transformers types .to() narrowly)
    model.eval()
    return tokenizer, model


def pool_hidden(hidden: Tensor, attention_mask: Tensor, pooling: str) -> Tensor:
    """Pool [B,T,D] hidden states to [B,D] float32 using the attention mask (right padding)."""
    mask = attention_mask.to(hidden.device)
    if pooling == "mean":
        weights = mask.unsqueeze(-1).to(torch.float32)
        summed = (hidden.to(torch.float32) * weights).sum(dim=1)
        return summed / weights.sum(dim=1).clamp(min=1.0)
    if pooling == "last":
        last_idx = (mask.sum(dim=1) - 1).clamp(min=0)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, last_idx].to(torch.float32)
    raise ValueError(f"unknown pooling {pooling!r}; choose 'mean' or 'last'")


def span_token_mask(offsets: Tensor, spans: Sequence[tuple[int, int]]) -> Tensor:
    """[K, T] bool: token t belongs to span k when their character ranges overlap.

    Raises PromptTooLongError if any span covers zero tokens (its line was truncated away).
    """
    span_t = torch.as_tensor(list(spans), dtype=offsets.dtype, device=offsets.device)
    starts, ends = offsets[:, 0].unsqueeze(0), offsets[:, 1].unsqueeze(0)
    mask = (starts < span_t[:, 1:2]) & (ends > span_t[:, 0:1])
    empty = (mask.sum(dim=1) == 0).nonzero().flatten().tolist()
    if empty:
        raise PromptTooLongError(f"candidate span(s) {empty} cover zero tokens; the prompt was truncated past the tool list")
    return mask


def check_spans_kept(offsets: Tensor, spans: Sequence[tuple[int, int]]) -> None:
    """Raise PromptTooLongError if left truncation removed the start of any candidate's line.

    `offsets` are one prompt's [T, 2] character offsets after truncation; padding tokens have (0, 0)."""
    real = offsets[offsets[:, 1] > offsets[:, 0]]
    first_kept = int(real[:, 0].min()) if len(real) else None
    if first_kept is None or spans[0][0] < first_kept:
        raise PromptTooLongError("the prompt is too long: truncation would cut the tool list; "
                                 "shorten the task, history or tool descriptions, or send fewer tools")


def pool_spans(hidden: Tensor, token_mask: Tensor) -> Tensor:
    """Mean of hidden[T, D] over each row of token_mask[K, T] -> [K, D] float32."""
    weights = token_mask.to(hidden.device, torch.float32)
    return (weights @ hidden.to(torch.float32)) / weights.sum(dim=1, keepdim=True).clamp(min=1.0)


def encode_with_spans(tokenizer, backbone, prompts: Sequence[str], spans: Sequence[Sequence[tuple[int, int]]],
                      cfg: ModelConfig) -> tuple[Tensor, list[Tensor]]:
    """One prefill per batch; returns the pooled query vectors [B, D] and, per example, its candidate
    span vectors [K_i, D]. Span pooling runs on CPU from a single copy of the hidden states."""
    device = next(backbone.parameters()).device
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    batch = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True,
                      max_length=cfg.max_prompt_tokens, return_offsets_mapping=True)
    offsets = batch.pop("offset_mapping")
    for i in range(len(prompts)):
        check_spans_kept(offsets[i], spans[i])
    inputs = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        hidden = backbone.get_decoder()(input_ids=inputs["input_ids"],
                                        attention_mask=inputs["attention_mask"]).last_hidden_state
    query = pool_hidden(hidden, inputs["attention_mask"], cfg.pooling)
    source = hidden.to(torch.float32).cpu()
    tools = [pool_spans(source[i], span_token_mask(offsets[i], spans[i])) for i in range(len(prompts))]
    return query, tools


def _residual_mlp(hidden_dim: int, head_hidden: int) -> nn.Sequential:
    """Linear -> GELU -> Linear with a zero-initialised output layer (state_dict keys 0.* and 2.*)."""
    out = nn.Linear(head_hidden, hidden_dim)
    nn.init.zeros_(out.weight)
    nn.init.zeros_(out.bias)
    return nn.Sequential(nn.Linear(hidden_dim, head_hidden), nn.GELU(), out)


class SpanScoringHead(nn.Module):
    """score(q, tool_k) = cos(q + mlp(q), t_k + mlp'(t_k)) / temperature for the K candidate spans of a prompt,
    scattered into catalog-width logits (-inf everywhere else). Both projections are residual with
    zero-initialised output layers, so an untrained head is plain cosine similarity."""

    def __init__(self, hidden_dim: int, head_hidden: int, temperature: float) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.proj = _residual_mlp(hidden_dim, head_hidden)
        self.tool_proj = _residual_mlp(hidden_dim, head_hidden)
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.head_hidden = head_hidden

    def forward(self, q: Tensor, tools: Tensor, tool_mask: Tensor, tool_idx: Tensor, num_tools: int) -> Tensor:
        q32, t32 = q.to(torch.float32), tools.to(torch.float32)
        qn = nn.functional.normalize(q32 + self.proj(q32), dim=-1)
        tn = nn.functional.normalize(t32 + self.tool_proj(t32), dim=-1)
        scores = torch.einsum("bd,bkd->bk", qn, tn) / self.temperature
        scores = scores.masked_fill(~tool_mask.to(scores.device), float("-inf"))
        logits = torch.full((q.shape[0], num_tools), float("-inf"), device=scores.device)
        # amax so a padded slot (idx 0, score -inf) can never clobber a real candidate at the same index
        return logits.scatter_reduce(1, tool_idx.to(scores.device), scores, reduce="amax", include_self=True)


def save_span_head(head: SpanScoringHead, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "hidden_dim": head.hidden_dim, "head_hidden": head.head_hidden,
                "temperature": head.temperature, "architecture": SPAN_HEAD_ARCHITECTURE}, path)


def load_span_head(path: Path) -> SpanScoringHead:
    """Load a span head checkpoint (tensors and plain values only; no pickled code is executed).

    Research checkpoints also carry a catalog fingerprint; it is ignored because the span head never
    indexes a catalog."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("architecture") != SPAN_HEAD_ARCHITECTURE:
        raise ValueError(f"head at {path} has architecture {saved.get('architecture')!r}, "
                         f"expected {SPAN_HEAD_ARCHITECTURE!r}")
    head = SpanScoringHead(saved["hidden_dim"], saved["head_hidden"], saved["temperature"])
    head.load_state_dict(saved["state_dict"])
    return head.eval()


def pad_tool_vectors(tools: Sequence[Tensor], examples: Sequence[Example], catalog: Catalog,
                     width: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Pad per-example [K_i, D] span vectors to [B, K, D] with a bool mask and catalog indices."""
    width = width or max(t.shape[0] for t in tools)
    dim, device = tools[0].shape[1], tools[0].device
    padded = torch.zeros(len(tools), width, dim, dtype=torch.float32, device=device)
    mask = torch.zeros(len(tools), width, dtype=torch.bool)
    idx = torch.zeros(len(tools), width, dtype=torch.long)
    for row, (vecs, ex) in enumerate(zip(tools, examples)):
        k = vecs.shape[0]
        padded[row, :k] = vecs.to(torch.float32)
        mask[row, :k] = True
        idx[row, :k] = torch.tensor([catalog.index(c) for c in ex.candidates])
    return padded, mask, idx


def build_candidate_mask(examples: Sequence[Example], catalog: Catalog) -> Tensor:
    mask = torch.zeros(len(examples), len(catalog), dtype=torch.bool)
    for row, ex in enumerate(examples):
        for name in ex.candidates:
            mask[row, catalog.index(name)] = True
    return mask


def rank_logits(logits: Tensor, k: int) -> list[tuple[int, ...]]:
    """Top-k catalog indices per row, excluding masked (-inf) tools even when fewer than k remain."""
    k = min(k, logits.shape[-1])
    values, indices = logits.topk(k, dim=-1)
    return [tuple(int(i) for i, v in zip(row_idx, row_val) if torch.isfinite(v))
            for row_idx, row_val in zip(indices.cpu(), values.cpu())]


class SpanActionRankModel(nn.Module):
    """Backbone + SpanScoringHead: candidates are scored from their own lines inside the prompt."""

    def __init__(self, tokenizer, backbone, head: SpanScoringHead, model_cfg: ModelConfig,
                 verbalize_cfg: VerbalizeConfig) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.head = head.to(next(backbone.parameters()).device)
        self.model_cfg = model_cfg
        self.verbalize_cfg = verbalize_cfg

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @torch.no_grad()
    def forward(self, examples: Sequence[Example], catalog: Catalog) -> Tensor:
        """Catalog-width logits [B, len(catalog)], -inf for every tool that is not a candidate."""
        rendered = [build_prompt_with_spans(ex, catalog, self.verbalize_cfg) for ex in examples]
        q, tools = encode_with_spans(self.tokenizer, self.backbone, [r[0] for r in rendered],
                                     [r[1] for r in rendered], self.model_cfg)
        padded, mask, idx = pad_tool_vectors(tools, examples, catalog)
        head_device = next(self.head.parameters()).device
        logits = self.head(q.to(head_device), padded.to(head_device), mask, idx, len(catalog))
        return logits.masked_fill(~build_candidate_mask(examples, catalog).to(logits.device), float("-inf"))
