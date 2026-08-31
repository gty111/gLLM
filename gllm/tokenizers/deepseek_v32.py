"""Compatibility wrappers for the official DeepSeek-V3.2 encoder."""

from gllm.tokenizers.deepseek_official import (
    apply_deepseek_chat_template,
    load_deepseek_encoder,
)


def load_dsv32_encoder(model_path: str):
    return load_deepseek_encoder(model_path, "dsv32")


def apply_dsv32_chat_template(encoder, messages, tokenizer, **kwargs):
    return apply_deepseek_chat_template(
        encoder, messages, tokenizer, **kwargs
    )


__all__ = ["apply_dsv32_chat_template", "load_dsv32_encoder"]
