"""Prompt builder: (task, compressed history, candidate tools) -> prompt text plus each candidate's span.

The format is byte-for-byte the one the checkpoint was trained on; changing it changes the model's inputs."""
from __future__ import annotations

import re

from .config import VerbalizeConfig
from .types import Catalog, Example, Step

ELLIPSIS = "…"
NO_HISTORY = "(no tool calls yet)"
SUFFIX = "Next tool:"
_WS = re.compile(r"\s+")


def truncate(text: str, limit: int) -> str:
    """Collapse whitespace and cut to `limit` chars, marking the cut with an ellipsis."""
    flat = _WS.sub(" ", text).strip()
    return flat if len(flat) <= limit else flat[:limit] + ELLIPSIS


def _render_step(index: int, step: Step, cfg: VerbalizeConfig) -> str:
    args = truncate(step.arguments, cfg.args_chars)
    obs = truncate(step.observation, cfg.observation_chars) or "(empty)"
    return f"Step {index}: called {step.action}({args}) -> {obs}"


def render_history(history: tuple[Step, ...], cfg: VerbalizeConfig) -> str:
    """Last `full_history_steps` steps in full; older steps collapsed to their tool names."""
    if not history:
        return NO_HISTORY
    keep = max(cfg.full_history_steps, 0)
    older, recent = (history[:-keep], history[-keep:]) if keep else (history, ())
    lines: list[str] = []
    if older:
        lines.append("Earlier calls: " + ", ".join(s.action for s in older))
    offset = len(older)
    lines.extend(_render_step(offset + i + 1, s, cfg) for i, s in enumerate(recent))
    return "\n".join(lines)


def render_catalog(candidates: tuple[str, ...], catalog: Catalog, cfg: VerbalizeConfig) -> str:
    """One `- name: description` line per candidate, in candidate order."""
    return "\n".join(
        f"- {name}: {truncate(catalog.describe(name), cfg.description_chars)}" for name in candidates
    )


def build_prompt(example: Example, catalog: Catalog, cfg: VerbalizeConfig) -> str:
    """Prefill-only prompt ending with 'Next tool:'."""
    return (
        f"Task: {truncate(example.query, 10_000)}\n\n"
        f"History:\n{render_history(example.history, cfg)}\n\n"
        f"Available tools:\n{render_catalog(example.candidates, catalog, cfg)}\n"
        f"\n{SUFFIX}"
    )


def build_prompt_with_spans(example: Example, catalog: Catalog,
                            cfg: VerbalizeConfig) -> tuple[str, tuple[tuple[int, int], ...]]:
    """The prompt plus the [start, end) character span of each candidate's catalog line.

    Spans are returned in candidate order and never include the trailing newline.
    """
    prompt = build_prompt(example, catalog, cfg)
    cursor = prompt.rindex(render_catalog(example.candidates, catalog, cfg))
    spans: list[tuple[int, int]] = []
    for name in example.candidates:
        start = prompt.index(f"- {name}: ", cursor)
        newline = prompt.find("\n", start)
        end = len(prompt) if newline < 0 else newline
        spans.append((start, end))
        cursor = end
    return prompt, tuple(spans)
