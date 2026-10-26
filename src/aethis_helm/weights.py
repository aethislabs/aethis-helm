"""Locate the trained weights: a local directory (HELM_WEIGHTS_DIR) or a Hugging Face Hub repo (HELM_HF_REPO).

Either source must contain `adapter/` (a PEFT LoRA adapter) and `head.pt` (the span scoring head)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import WeightsConfig

ADAPTER_DIR = "adapter"
HEAD_FILE = "head.pt"
HUB_PATTERNS = (f"{ADAPTER_DIR}/*", HEAD_FILE)


@dataclass(frozen=True)
class WeightPaths:
    adapter_dir: Path
    head_file: Path


def _validated(root: Path) -> WeightPaths:
    paths = WeightPaths(adapter_dir=root / ADAPTER_DIR, head_file=root / HEAD_FILE)
    missing = [str(p) for p in (paths.adapter_dir / "adapter_config.json", paths.head_file) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"weights at {root} are incomplete; missing {missing}")
    return paths


def resolve_weights(cfg: WeightsConfig) -> WeightPaths:
    """Local directory if set, otherwise download (or reuse the cached copy of) the Hub repo."""
    if cfg.local_dir:
        return _validated(Path(cfg.local_dir).expanduser())
    if not cfg.hf_repo:
        raise RuntimeError("no weights configured: set HELM_WEIGHTS_DIR to a local directory or "
                           "HELM_HF_REPO to a Hugging Face model repo")
    from huggingface_hub import snapshot_download

    root = snapshot_download(cfg.hf_repo, revision=cfg.hf_revision, allow_patterns=list(HUB_PATTERNS))
    return _validated(Path(root))
