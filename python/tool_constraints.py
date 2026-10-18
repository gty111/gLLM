"""Generation constraints for Qwen tool calls, including raw custom inputs.

Use XGrammar's public structural-tag and Lark frontends. Only serialized
specifications cross the worker boundary; no mutable matcher is shared.
"""
import json
from functools import lru_cache


@lru_cache(maxsize=64)
def custom_grammar(syntax, definition):
    import xgrammar as xgr

    if syntax == "lark":
        return str(xgr.Grammar.from_lark(definition))
    if syntax == "regex":
        return str(xgr.Grammar.from_regex(definition))
    raise ValueError(f"Unsupported custom tool grammar syntax: {syntax}.")


def build_tool_tag(tools, parser_name, *, custom_formats=None, schema=None,
                   parallel_tool_calls=True):
    """Bind constraints to the selected parser, not just model architecture.

    Raw custom grammars currently require Qwen XML parameters. Reject other
    envelopes explicitly rather than claiming to constrain JSON string contents.
    """
    if not tools:
        return None
    custom_formats = custom_formats or {}
    models = {"qwen": "qwen_3", "qwen3": "qwen_3_5"}
    if parser_name not in models:
        if custom_formats or schema is not None:
            raise ValueError("Constrained tools require a Qwen tool-call parser.")
        return None  # Preserve existing non-Qwen function calling.
    if custom_formats and parser_name != "qwen3":
        raise ValueError("Custom tool grammar constraints require the Qwen XML tool-call parser.")
    import xgrammar as xgr

    wire = [t.model_dump(by_alias=True, exclude_none=True)
            if hasattr(t, "model_dump") else t for t in tools]
    tag = xgr.get_model_structural_tag(
        models[parser_name], wire, "auto", reasoning="disabled",
        any_order=False, exclude_special_tokens=True,
        parallel_tool_calls=parallel_tool_calls,
    ).model_dump(mode="json")
    suffix = tag["format"]
    # The upstream XML trigger includes the function prefix. Trigger at the
    # opening marker instead, so malformed/JSON calls cannot bypass the mask.
    if suffix.get("type") != "triggered_tags":
        raise ValueError("Unsupported XGrammar tool-tag layout.")
    suffix["triggers"] = ["<tool_call>"]
    for item, tool in zip(suffix["tags"], wire, strict=True):
        name = tool["function"]["name"]
        fmt = custom_formats.get(name)
        if fmt is None:
            continue
        content = ({"type": "grammar", "grammar": custom_grammar(fmt["syntax"], fmt["definition"])}
                   if fmt.get("type") == "grammar" else {"type": "any_text"})
        # Exactly one framing newline on each side. The custom parser removes
        # only those bytes, preserving meaningful whitespace in the input.
        item["content"] = {"type": "tag", "begin": "<parameter=input>\n",
                           "content": content, "end": "\n</parameter>"}
    if schema is not None:
        # A turn may return schema-conforming text OR one/more tool calls.
        # Never silently drop either constraint when both were requested.
        call = {"type": "or", "elements": suffix["tags"]}
        calls = call if not parallel_tool_calls else {
            "type": "sequence", "elements": [call, {"type": "star", "content": {
                "type": "sequence", "elements": [
                    {"type": "regex", "pattern": r"[ \t\r\n]*"}, call]}}]}
        whitespace = {"type": "regex", "pattern": r"[ \t\r\n]*"}
        tag["format"] = {"type": "sequence", "elements": [whitespace,
            {"type": "or", "elements": [
                {"type": "json_schema", "json_schema": json.loads(schema)}, calls]}, whitespace]}
    return json.dumps(tag)
