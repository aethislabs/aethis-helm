"""Typed, immutable settings. Defaults are the trained checkpoint's settings (ActionRank joint_s1);
only deployment knobs come from the environment, and only through HelmConfig.from_env()."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

DEVICES = ("cuda", "mps", "cpu")


@dataclass(frozen=True)
class VerbalizeConfig:
    full_history_steps: int = 3
    observation_chars: int = 200
    args_chars: int = 120
    description_chars: int = 200


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "cpu"
    pooling: str = "last"
    max_prompt_tokens: int = 1280

    @property
    def dtype(self) -> str:
        """Half precision on accelerators (as trained), full precision on CPU."""
        return "float32" if self.device == "cpu" else "float16"


@dataclass(frozen=True)
class WeightsConfig:
    local_dir: str | None = None  # directory with adapter/ and head.pt; wins over the Hub
    hf_repo: str | None = None
    hf_revision: str = "main"


@dataclass(frozen=True)
class HelmConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    verbalize: VerbalizeConfig = field(default_factory=VerbalizeConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    min_tools: int = 2
    max_tools: int = 10  # tools a caller sends; Finish may be added on top, for at most 11 as in training

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "HelmConfig":
        env = os.environ if env is None else env
        device = env.get("HELM_DEVICE", "cpu").strip().lower()
        if device not in DEVICES:
            raise ValueError(f"HELM_DEVICE must be one of {DEVICES}, got {device!r}")
        model = ModelConfig(backbone=env.get("HELM_BACKBONE") or ModelConfig.backbone, device=device)
        weights = WeightsConfig(local_dir=env.get("HELM_WEIGHTS_DIR") or None,
                                hf_repo=env.get("HELM_HF_REPO") or None,
                                hf_revision=env.get("HELM_HF_REVISION") or WeightsConfig.hf_revision)
        return cls(model=model, weights=weights)
