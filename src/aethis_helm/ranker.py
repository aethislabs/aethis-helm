"""Ranker: the one object serving needs. Loads backbone + adapter + head once, then ranks tool lists."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch

from .config import HelmConfig
from .model import SpanActionRankModel, load_backbone, load_span_head, synchronize
from .types import Catalog, Example, Step, ToolSpec
from .weights import resolve_weights

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankedTool:
    name: str
    score: float  # cosine similarity / temperature; comparable within one request only
    prob: float  # softmax over this request's tools; a ranking signal, not a calibrated probability


class Ranker:
    def __init__(self, model: SpanActionRankModel) -> None:
        self._model = model

    @classmethod
    def load(cls, cfg: HelmConfig) -> "Ranker":
        from peft import PeftModel

        paths = resolve_weights(cfg.weights)
        tokenizer, backbone = load_backbone(cfg.model)
        peft_model = PeftModel.from_pretrained(backbone, str(paths.adapter_dir)).eval()
        head = load_span_head(paths.head_file)
        log.info("loaded %s + adapter %s on %s (%s)", cfg.model.backbone, paths.adapter_dir,
                 cfg.model.device, cfg.model.dtype)
        return cls(SpanActionRankModel(tokenizer, peft_model, head, cfg.model, cfg.verbalize))

    @property
    def device(self) -> str:
        return self._model.device.type

    def rank(self, task: str, history: Sequence[Step], tools: Sequence[ToolSpec]) -> list[RankedTool]:
        """Score every tool in `tools` (prompt order is the given order); best first.

        Raises PromptTooLongError when the prompt cannot fit without cutting a tool's line."""
        catalog = Catalog(tuple(tools))
        example = Example(query=task, history=tuple(history), candidates=catalog.names)
        logits = self._model([example], catalog)[0]
        synchronize(self._model.device)
        scores = logits.float().cpu()
        probs = torch.softmax(scores, dim=-1)
        ranked = sorted(range(len(catalog)), key=lambda i: -float(scores[i]))
        return [RankedTool(name=catalog.names[i], score=float(scores[i]), prob=float(probs[i])) for i in ranked]
