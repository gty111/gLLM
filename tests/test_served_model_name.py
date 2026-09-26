from types import SimpleNamespace

from gllm.entrypoints import api_server


def test_served_model_name_keeps_stable_api_id(monkeypatch):
    old_id = "/models/Qwen3.8-27B"
    new_id = "/models/Qwen3.8-27B-FP8"
    args = api_server.build_arg_parser().parse_args([
        "--model-path", new_id, "--served-model-name", old_id,
    ])
    monkeypatch.setattr(api_server, "llm", SimpleNamespace(model_path=new_id))
    monkeypatch.setattr(api_server, "served_model_names", args.served_model_name)

    assert api_server._validate_model(old_id) is None
    assert api_server._validate_model(new_id) is None
    assert api_server._public_model_id() == old_id
    assert api_server._validate_model("/models/other").status_code == 404
