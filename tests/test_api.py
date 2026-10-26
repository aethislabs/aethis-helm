import pytest
from fastapi.testclient import TestClient

from aethis_helm.model import PromptTooLongError
from aethis_helm.ranker import RankedTool
from aethis_helm.serve import FINISH_NOTE, create_app
from aethis_helm.types import FINISH, FINISH_DESCRIPTION


class FakeRanker:
    """Scores tools by position (first tool best) and records what it was asked."""

    device = "cpu"

    def __init__(self, error: Exception | None = None) -> None:
        self.calls = []
        self.error = error

    def rank(self, task, history, tools):
        self.calls.append((task, list(history), list(tools)))
        if self.error:
            raise self.error
        n = len(tools)
        return [RankedTool(name=t.name, score=float(n - i), prob=1.0 / n) for i, t in enumerate(tools)]


@pytest.fixture
def ranker():
    return FakeRanker()


@pytest.fixture
def client(ranker):
    with TestClient(create_app(lambda: ranker)) as c:
        yield c


def tools(n, with_finish=False):
    out = [{"name": f"tool_{i}", "description": f"does thing {i}"} for i in range(n)]
    return out + ([{"name": FINISH, "description": "stop"}] if with_finish else [])


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "device": "cpu",
                                            "version": "0.1.0"}


def test_response_shape_sorted_best_first(client):
    res = client.post("/rank", json={"task": "do it", "tools": tools(3, with_finish=True)})
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"ranked", "finish_added", "notes", "latency_ms"}
    assert [r["name"] for r in body["ranked"]] == ["tool_0", "tool_1", "tool_2", FINISH]
    assert all(set(r) == {"name", "score", "prob"} for r in body["ranked"])
    scores = [r["score"] for r in body["ranked"]]
    assert scores == sorted(scores, reverse=True)
    assert body["finish_added"] is False and body["notes"] == []


def test_finish_is_added_last_with_the_trained_description(client, ranker):
    body = client.post("/rank", json={"task": "do it", "tools": tools(2)}).json()
    assert body["finish_added"] is True and body["notes"] == [FINISH_NOTE]
    assert FINISH in [r["name"] for r in body["ranked"]]
    sent = ranker.calls[0][2]
    assert sent[-1].name == FINISH and sent[-1].description == FINISH_DESCRIPTION
    assert [t.name for t in sent[:-1]] == ["tool_0", "tool_1"]


def test_history_is_converted_with_pretty_printed_args(client, ranker):
    payload = {"task": "t", "tools": tools(2),
               "history": [{"tool": "tool_0", "args": {"city": "Paris"}, "result": "sunny"}, {"tool": "tool_1"}]}
    assert client.post("/rank", json=payload).status_code == 200
    history = ranker.calls[0][1]
    assert history[0].action == "tool_0" and history[0].arguments == '{\n  "city": "Paris"\n}'
    assert history[0].observation == "sunny"
    assert history[1].arguments == "{}" and history[1].observation == ""


@pytest.mark.parametrize("payload, needle", [
    ({"task": "t", "tools": tools(1)}, "at least 2"),
    ({"task": "t", "tools": tools(11)}, "at most 10"),
    ({"task": "t", "tools": tools(10, with_finish=True)}, "at most 10"),
    ({"task": "t", "tools": tools(2) + tools(1)}, "unique"),
    ({"task": "", "tools": tools(2)}, "at least 1 character"),
    ({"tools": tools(2)}, "Field required"),
    ({"task": "t", "tools": [{"name": "a\nb", "description": ""}, {"name": "c"}]}, "single printable line"),
    ({"task": "t", "tools": [{"name": "  ", "description": ""}, {"name": "c"}]}, "must not be empty"),
    ({"task": "t", "tools": tools(2), "history": [{"tool": "x", "args": [1, 2]}]}, "dictionary"),
])
def test_validation_errors_are_422(client, ranker, payload, needle):
    res = client.post("/rank", json=payload)
    assert res.status_code == 422
    assert needle in res.text
    assert ranker.calls == []


def test_ten_tools_plus_added_finish_is_the_maximum(client, ranker):
    body = client.post("/rank", json={"task": "t", "tools": tools(10)}).json()
    assert body["finish_added"] is True and len(body["ranked"]) == 11
    assert len(ranker.calls[0][2]) == 11


def test_long_description_is_accepted_and_left_to_the_prompt_builder(client, ranker):
    payload = {"task": "t", "tools": [{"name": "a", "description": "x" * 5000}, {"name": "b"}]}
    assert client.post("/rank", json=payload).status_code == 200
    assert len(ranker.calls[0][2][0].description) == 5000  # verbalize cuts it to 200 chars, as in training


def test_prompt_too_long_is_a_clear_422():
    with TestClient(create_app(lambda: FakeRanker(PromptTooLongError("the prompt is too long")))) as c:
        res = c.post("/rank", json={"task": "t", "tools": tools(2)})
    assert res.status_code == 422 and res.json() == {"detail": "the prompt is too long"}
