import json

import pytest

from gllm.tokenizers.tool_parsers import DeepSeekToolParser


@pytest.mark.parametrize("outer_tag", ["function_calls", "tool_calls"])
@pytest.mark.parametrize("prefix", ["｜DSML｜", ""])
def test_deepseek_parser_accepts_v32_and_v4_outer_tags(outer_tag, prefix):
    text = (
        f"<thinking>done</thinking>\n<{prefix}{outer_tag}>\n"
        f'<{prefix}invoke name="calculate">\n'
        f'<{prefix}parameter name="expression" string="true">'
        f"3*9*19</{prefix}parameter>\n"
        f"</{prefix}invoke>\n</{prefix}{outer_tag}>"
    )

    content, calls = DeepSeekToolParser().parse(text)

    assert content == "<thinking>done</thinking>"
    assert len(calls) == 1
    assert calls[0].function.name == "calculate"
    assert json.loads(calls[0].function.arguments) == {"expression": "3*9*19"}


def test_deepseek_v4_stream_parser_emits_structured_tool_call():
    parser = DeepSeekToolParser().stream_parser()
    marker = "<｜DSML｜tool_calls>"
    for end in range(1, len(marker) + 1):
        assert parser.process(marker[:end]) is None

    partial = marker + "\n<｜DSML｜invoke name=\"calculate\">"
    assert parser.process(partial) is None

    full = (
        partial
        + "\n<｜DSML｜parameter name=\"expression\" string=\"true\">"
        + "3*9*19</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
        + "</｜DSML｜tool_calls>"
    )
    delta = parser.process(full)

    assert delta.tool_calls[0].index == 0
    assert delta.tool_calls[0].function.name == "calculate"
    assert json.loads(delta.tool_calls[0].function.arguments) == {
        "expression": "3*9*19"
    }
    assert parser.has_tool_calls()
