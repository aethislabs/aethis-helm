import pytest

from aethis_helm import weights
from aethis_helm.config import WeightsConfig


def _make_weights(root):
    (root / "adapter").mkdir(parents=True)
    (root / "adapter" / "adapter_config.json").write_text("{}")
    (root / "head.pt").write_bytes(b"x")
    return root


def test_local_dir_wins_over_hub(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not download")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    root = _make_weights(tmp_path / "w")
    paths = weights.resolve_weights(WeightsConfig(local_dir=str(root), hf_repo="u/r"))
    assert paths.adapter_dir == root / "adapter" and paths.head_file == root / "head.pt"


def test_incomplete_local_dir_is_reported(tmp_path):
    (tmp_path / "adapter").mkdir()
    with pytest.raises(FileNotFoundError, match="head.pt"):
        weights.resolve_weights(WeightsConfig(local_dir=str(tmp_path)))


def test_no_source_configured_is_a_clear_error():
    with pytest.raises(RuntimeError, match="HELM_WEIGHTS_DIR"):
        weights.resolve_weights(WeightsConfig())


def test_hub_download_requests_only_the_weight_files(tmp_path, monkeypatch):
    seen = {}

    def fake_download(repo_id, revision, allow_patterns):
        seen.update(repo_id=repo_id, revision=revision, allow_patterns=allow_patterns)
        return str(_make_weights(tmp_path / "snap"))

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    paths = weights.resolve_weights(WeightsConfig(hf_repo="u/r", hf_revision="v1"))
    assert seen == {"repo_id": "u/r", "revision": "v1", "allow_patterns": ["adapter/*", "head.pt"]}
    assert paths.head_file == tmp_path / "snap" / "head.pt"
