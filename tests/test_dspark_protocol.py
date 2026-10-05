"""Conformance of the DSpark adapters with the shared execution contract.

The contract module is deliberately torch-free (its annotations are lazy), so
the abstractness check runs anywhere.  The adapter conformance check has to
import the DeepSeek-V4 model, which pulls in the whole runtime stack -- that
import stays inside the test body so collecting this module stays cheap.
"""


def test_protocol_cannot_be_instantiated():
    import pytest

    from gllm.models.dspark_protocol import DSparkForwardProtocol

    # One abstract method left unimplemented must refuse construction; this is
    # what turns "an adapter forgot a stage" into an import-time failure.
    with pytest.raises(TypeError):
        DSparkForwardProtocol()


def test_deepseek_v4_dspark_implements_protocol():
    from gllm.models.deepseek_v4_dspark import DeepseekV4DSpark
    from gllm.models.dspark_protocol import DSparkForwardProtocol

    assert issubclass(DeepseekV4DSpark, DSparkForwardProtocol)
    for stage in ("prefill", "forward_draft", "forward_head"):
        assert callable(getattr(DeepseekV4DSpark, stage, None)), stage
    # An empty abstract set is the proof that every stage is provided.
    assert DeepseekV4DSpark.__abstractmethods__ == frozenset()