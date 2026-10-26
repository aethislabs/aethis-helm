from aethis_helm.config import VerbalizeConfig
from aethis_helm.types import Catalog, Example, Step, ToolSpec
from aethis_helm.verbalize import build_prompt, build_prompt_with_spans, render_catalog, render_history, truncate

CFG = VerbalizeConfig(full_history_steps=2, observation_chars=20, args_chars=10, description_chars=15)
CAT = Catalog((ToolSpec("a_tool", "Alpha does alpha things indeed"), ToolSpec("b_tool", "Beta"), ToolSpec("Finish", "Stop")))

# Produced by ActionRank's original verbalize.py with the checkpoint's settings: the port must match it exactly.
GOLDEN_PROMPT = (
    'Task: Plan a trip to Paris\n\nHistory:\nEarlier calls: t0, t1\n'
    'Step 3: called t2({ "city": "Paris" }) -> sunny and warm\n'
    'Step 4: called t3({ "city": "Paris" }) -> sunny and warm\n'
    'Step 5: called t4({ "city": "Paris" }) -> sunny and warm\n\nAvailable tools:\n'
    '- get_weather: ' + 'Current weather for a city. ' * 7 + 'Curr…\n'
    '- search_flights: Search flights\n- Finish: Stop\n\nNext tool:'
)
GOLDEN_SPANS = ((248, 464), (465, 497), (498, 512))


def steps(n):
    return tuple(Step(f"tool{i}", '{"x": ' + str(i) + '}', "obs" * 20) for i in range(n))


def test_matches_research_prompt_byte_for_byte():
    cat = Catalog((ToolSpec("get_weather", "Current weather for a city. " * 12),
                   ToolSpec("search_flights", "Search flights"), ToolSpec("Finish", "Stop")))
    history = tuple(Step(f"t{i}", '{\n  "city": "Paris"\n}', "sunny  and\n warm") for i in range(5))
    ex = Example("Plan a trip to Paris", history, ("get_weather", "search_flights", "Finish"))
    prompt, spans = build_prompt_with_spans(ex, cat, VerbalizeConfig())
    assert prompt == GOLDEN_PROMPT
    assert spans == GOLDEN_SPANS


def test_truncate_marks_cut():
    assert truncate("abcdef", 3) == "abc…"
    assert truncate("ab", 3) == "ab"
    assert truncate("a\nb  c", 10) == "a b c"


def test_history_keeps_last_k_full_and_summarizes_older():
    text = render_history(steps(4), CFG)
    assert "Earlier calls: tool0, tool1" in text
    assert "tool2(" in text and "tool3(" in text
    assert text.count("->") == 2
    assert "obsobsobsobsobsobsob…" in text
    assert "tool0(" not in text


def test_history_no_summary_line_when_short():
    text = render_history(steps(2), CFG)
    assert "Earlier calls" not in text and text.count("->") == 2


def test_history_empty():
    assert render_history((), CFG) == "(no tool calls yet)"


def test_catalog_lines_follow_candidate_order_and_truncate():
    text = render_catalog(("b_tool", "a_tool"), CAT, CFG)
    assert text.splitlines() == ["- b_tool: Beta", "- a_tool: Alpha does alph…"]


def test_prompt_contains_all_sections():
    ex = Example("Find alpha", steps(1), ("a_tool", "Finish"))
    p = build_prompt(ex, CAT, CFG)
    assert p.startswith("Task: Find alpha")
    assert "History:" in p and "Available tools:" in p
    assert p.rstrip().endswith("Next tool:")
    assert p.index("History:") < p.index("Available tools:")


def test_prompt_with_spans_matches_prompt_and_locates_each_candidate_line():
    ex = Example("Find alpha", steps(1), ("b_tool", "a_tool", "Finish"))
    prompt, spans = build_prompt_with_spans(ex, CAT, CFG)
    assert prompt == build_prompt(ex, CAT, CFG)
    assert len(spans) == 3
    for name, (start, end) in zip(ex.candidates, spans):
        assert prompt[start:end].startswith(f"- {name}: ")
        assert "\n" not in prompt[start:end]
    assert spans[0][1] <= spans[1][0] <= spans[2][0]


def test_spans_are_not_fooled_by_a_name_inside_another_description():
    cat = Catalog((ToolSpec("a", "see - b: for details"), ToolSpec("b", "Beta")))
    prompt, spans = build_prompt_with_spans(Example("q", (), ("a", "b")), cat, CFG)
    assert prompt[spans[1][0]:spans[1][1]] == "- b: Beta"
