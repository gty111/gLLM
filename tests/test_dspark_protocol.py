"""Conformance of the DSpark adapters with the shared contracts.

The contract module is deliberately torch-free (its annotations are lazy), so
the abstractness checks run anywhere.  Everything that has to import a real
adapter pulls in the runtime stack, so those imports stay inside the test body
-- collecting this module stays cheap, matching the local convention.
"""


def test_forward_protocol_cannot_be_instantiated():
    import pytest

    from gllm.models.dspark_protocol import DSparkForwardProtocol

    # One abstract method left unimplemented must refuse construction; this is
    # what turns "an adapter forgot a stage" into an import-time failure.
    with pytest.raises(TypeError):
        DSparkForwardProtocol()


def test_checkpoint_mapping_cannot_be_instantiated():
    import pytest

    from gllm.models.dspark_protocol import DSparkCheckpointMapping

    with pytest.raises(TypeError):
        DSparkCheckpointMapping()


def test_deepseek_v4_dspark_implements_forward_protocol():
    from gllm.models.deepseek_v4_dspark import DeepseekV4DSpark
    from gllm.models.dspark_protocol import DSparkForwardProtocol

    assert issubclass(DeepseekV4DSpark, DSparkForwardProtocol)
    for stage in ("prefill", "forward_draft", "forward_head"):
        assert callable(getattr(DeepseekV4DSpark, stage, None)), stage
    # An empty abstract set is the proof that every stage is provided.
    assert DeepseekV4DSpark.__abstractmethods__ == frozenset()


def test_deepseek_v4_mapping_conforms_and_maps_to_mtp_keys():
    from types import SimpleNamespace

    from gllm.models.deepseek_v4_dspark import DeepseekV4DSparkMapping
    from gllm.models.dspark_protocol import DSparkCheckpointMapping

    # ``checkpoint_key`` only needs ``num_stages``; the real module would drag
    # in CUDA-tensor construction, and the mapping is pure string work.
    mapping = DeepseekV4DSparkMapping(
        parent=None, dspark=SimpleNamespace(num_stages=3)
    )
    assert issubclass(DeepseekV4DSparkMapping, DSparkCheckpointMapping)
    assert DeepseekV4DSparkMapping.__abstractmethods__ == frozenset()

    cases = {
        # The draft stages live under mtp.0/1/2 ...
        "blocks.0.attn.projections.wq_a.weight": "mtp.0.attn.wq_a.weight",
        "blocks.2.ffn.experts.w13_weight": "mtp.2.ffn.experts.w13_weight",
        # ... the target-hidden projection on the first stage ...
        "main_proj.weight": "mtp.0.main_proj.weight",
        "main_proj.weight_scale_inv": "mtp.0.main_proj.scale",
        "main_norm.weight": "mtp.0.main_norm.weight",
        # ... and the joint head on the last one.
        "markov_w1.weight": "mtp.2.markov_head.markov_w1.weight",
        "markov_w2.weight": "mtp.2.markov_head.markov_w2.weight",
        "confidence_proj.weight": "mtp.2.confidence_head.proj.weight",
        "norm.weight": "mtp.2.norm.weight",
    }
    for name, expected in cases.items():
        assert mapping.checkpoint_key(name) == expected, name