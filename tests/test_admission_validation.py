import json
from types import SimpleNamespace

import pytest

from gllm.distributed.comm import IPCPackage
from gllm.runtime.sequence import GenerationSequence
from gllm.structured_output import StructuredOutput, normalize_structural_tag


def test_legacy_structural_tag_becomes_triggered_tags():
    legacy = {"structures": [{"begin": "<a>", "schema": {"type": "object"}, "end": "</a>"}],
              "triggers": ["<a>"]}
    tag = json.loads(normalize_structural_tag(json.dumps(legacy)))
    assert tag["type"] == "structural_tag"
    assert tag["format"]["type"] == "triggered_tags"
    assert tag["format"]["tags"][0]["content"]["json_schema"] == {"type": "object"}


@pytest.mark.parametrize("bad", ["[]", "not json", '{"format": 1}'])
def test_malformed_structural_tags_are_refused(bad):
    with pytest.raises(ValueError):
        normalize_structural_tag(bad)


def _seq(seq_id, spec):
    return GenerationSequence(seq_id, [1, 2], [3], 8, structured_output=spec)


def test_driver_refuses_requests_whose_constraint_does_not_compile(monkeypatch):
    from gllm.workers import worker as worker_mod

    def validate(spec, tokenizer, vocab_size, stop_tokens):
        if spec.schema == "bad":
            raise ValueError("Invalid structured output constraint: nope")
        return spec

    monkeypatch.setattr("gllm.structured_output.validate_structured_output", validate)
    sent = []
    w = worker_mod.Worker.__new__(worker_mod.Worker)
    w.model_runner = SimpleNamespace(tokenizer=None, model_loader=SimpleNamespace(vocab_size=10))
    w.comm = SimpleNamespace(send_output=sent.append)
    ok, bad, plain = _seq(1, StructuredOutput("{}")), _seq(2, StructuredOutput("bad")), _seq(3, None)
    assert w._refuse_invalid_requests([ok, bad, plain]) == [ok, plain]
    (pkg,) = sent
    assert isinstance(pkg, IPCPackage) and pkg.free_ids == [2]
    assert pkg.request_errors == {2: "Invalid structured output constraint: nope"}
    assert pkg.request_error_codes == {2: "invalid_request"}
