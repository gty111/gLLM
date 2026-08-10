import json

import pytest
import torch
from safetensors.torch import save_file

from gllm.model_loader import ModelLoader


def _loader() -> ModelLoader:
    return ModelLoader.__new__(ModelLoader)


def test_safetensors_index_ignores_alternate_shard_set(tmp_path):
    chosen = tmp_path / "model-00001-of-00001.safetensors"
    stale = tmp_path / "alternate-00001-of-00001.safetensors"
    save_file({"weight": torch.ones(4)}, chosen)
    save_file({"weight": torch.zeros(4)}, stale)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": chosen.name}}), encoding="utf-8"
    )

    loader = _loader()
    assert loader.load_safetensors(str(tmp_path))
    assert torch.equal(loader.weights["weight"], torch.ones(4))
    assert loader.weights._index["weight"][0] == str(chosen)
    loader.weights.close()


def test_unindexed_duplicate_safetensors_key_is_rejected(tmp_path):
    save_file({"weight": torch.ones(4)}, tmp_path / "a.safetensors")
    save_file({"weight": torch.zeros(4)}, tmp_path / "b.safetensors")

    with pytest.raises(ValueError, match="Duplicate tensor key"):
        _loader().load_safetensors(str(tmp_path))
