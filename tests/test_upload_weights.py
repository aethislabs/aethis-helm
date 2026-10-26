import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("upload_weights", Path(__file__).parents[1] / "scripts" / "upload_weights.py")
upload_weights = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upload_weights)


def test_plan_maps_checkpoint_files_to_repo_paths(tmp_path):
    (tmp_path / "adapter").mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (tmp_path / "adapter" / name).write_text("x")
    (tmp_path / "head.pt").write_text("x")
    remote = [r for _, r in upload_weights.plan(tmp_path)]
    assert remote == ["adapter/adapter_config.json", "adapter/adapter_model.safetensors", "head.pt"]


def test_plan_refuses_incomplete_checkpoint(tmp_path):
    (tmp_path / "head.pt").write_text("x")
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        upload_weights.plan(tmp_path)
