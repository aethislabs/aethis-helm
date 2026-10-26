"""Immutable value types shared by the prompt builder, the model and the API."""
from __future__ import annotations

from dataclasses import dataclass

FINISH = "Finish"
# Verbatim ToolBench description of Finish: the model was trained with exactly this line in every prompt.
FINISH_DESCRIPTION = (
    "If you think you get the result which can answer the task, call this function to give the final answer. "
    "Or, if you think you can't handle the task from this status, call this function to restart. Remember: you "
    "should ALWAYS call this function at the end of your try, and the final answer is the ONLY part that will be "
    "showed to user, so final answer should contain enough information."
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str


@dataclass(frozen=True)
class Step:
    """One earlier call: the tool, its serialized arguments, and what it returned."""

    action: str
    arguments: str
    observation: str


@dataclass(frozen=True)
class Example:
    """One decision point: the task, the calls so far, and the candidate tool names in prompt order."""

    query: str
    history: tuple[Step, ...]
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class Catalog:
    """Name -> description lookup with a stable index per tool. Built per request in serving."""

    tools: tuple[ToolSpec, ...]

    def __post_init__(self) -> None:
        names = [t.name for t in self.tools]
        if len(set(names)) != len(names):
            raise ValueError("catalog contains duplicate tool names")
        object.__setattr__(self, "_index", {n: i for i, n in enumerate(names)})

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    def index(self, name: str) -> int:
        return self._index[name]  # type: ignore[attr-defined]

    def __contains__(self, name: object) -> bool:
        return name in self._index  # type: ignore[attr-defined]

    def describe(self, name: str) -> str:
        return self.tools[self.index(name)].description
