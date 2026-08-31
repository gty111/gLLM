"""Compatibility wrappers for the official DeepSeek-V4 encoder."""

from gllm.tokenizers.deepseek_official import (
    apply_deepseek_chat_template,
    load_deepseek_encoder,
)


def load_dsv4_encoder(model_path: str):
    return load_deepseek_encoder(model_path, "dsv4")


def apply_dsv4_chat_template(encoder, messages, tokenizer, **kwargs):
    return apply_deepseek_chat_template(
        encoder, messages, tokenizer, **kwargs
    )


__all__ = ["apply_dsv4_chat_template", "load_dsv4_encoder"]
