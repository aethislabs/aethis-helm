import pytest

from aethis_helm.config import HelmConfig, ModelConfig, VerbalizeConfig
from aethis_helm.types import Catalog, ToolSpec


def test_catalog_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        Catalog((ToolSpec("a", ""), ToolSpec("a", "")))


def test_catalog_lookup():
    cat = Catalog((ToolSpec("a", "Alpha"), ToolSpec("b", "Beta")))
    assert cat.names == ("a", "b") and len(cat) == 2
    assert cat.index("b") == 1 and cat.describe("a") == "Alpha"
    assert "a" in cat and "z" not in cat


def test_defaults_are_the_checkpoint_settings():
    cfg = HelmConfig()
    assert cfg.model == ModelConfig(backbone="Qwen/Qwen2.5-1.5B-Instruct", device="cpu", pooling="last",
                                    max_prompt_tokens=1280)
    assert cfg.verbalize == VerbalizeConfig(full_history_steps=3, observation_chars=200, args_chars=120,
                                            description_chars=200)
    assert (cfg.min_tools, cfg.max_tools) == (2, 15)


def test_config_is_immutable():
    with pytest.raises(AttributeError):
        HelmConfig().model.device = "cuda"  # type: ignore[misc]


def test_from_env_reads_only_deployment_knobs():
    cfg = HelmConfig.from_env({"HELM_DEVICE": "MPS", "HELM_WEIGHTS_DIR": "/w", "HELM_HF_REPO": "u/r",
                               "HELM_HF_REVISION": "v1"})
    assert cfg.model.device == "mps" and cfg.model.dtype == "float16"
    assert (cfg.weights.local_dir, cfg.weights.hf_repo, cfg.weights.hf_revision) == ("/w", "u/r", "v1")
    assert cfg.verbalize == VerbalizeConfig()


def test_from_env_defaults_to_cpu_in_full_precision():
    cfg = HelmConfig.from_env({})
    assert cfg.model.device == "cpu" and cfg.model.dtype == "float32"
    assert cfg.weights.local_dir is None and cfg.weights.hf_repo is None


def test_from_env_rejects_unknown_device():
    with pytest.raises(ValueError, match="HELM_DEVICE"):
        HelmConfig.from_env({"HELM_DEVICE": "tpu"})
