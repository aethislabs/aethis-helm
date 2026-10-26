"""Loads the real backbone and trained weights. Run with: HELM_WEIGHTS_DIR=... pytest -m slow"""
import os

import pytest

from aethis_helm.config import HelmConfig
from aethis_helm.ranker import Ranker
from aethis_helm.types import FINISH, FINISH_DESCRIPTION, Step, ToolSpec

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def ranker():
    if not (os.environ.get("HELM_WEIGHTS_DIR") or os.environ.get("HELM_HF_REPO")):
        pytest.skip("set HELM_WEIGHTS_DIR or HELM_HF_REPO")
    return Ranker.load(HelmConfig.from_env())


TOOLS = [ToolSpec("get_current_weather", "Get the current weather for a city."),
         ToolSpec("search_flights", "Search flights between two airports on a date."),
         ToolSpec("convert_currency", "Convert an amount between two currencies."),
         ToolSpec(FINISH, FINISH_DESCRIPTION)]


def test_picks_the_obvious_tool(ranker):
    ranked = ranker.rank("What is the weather in Paris right now?", [], TOOLS)
    assert ranked[0].name == "get_current_weather"


def test_finishes_once_the_answer_is_in_the_history(ranker):
    history = [Step("get_current_weather", '{\n  "city": "Paris"\n}', '{"temp_c": 18, "conditions": "sunny"}')]
    ranked = ranker.rank("What is the weather in Paris right now?", history, TOOLS)
    assert ranked[0].name == FINISH
