"""Tokenizer encode/decode logic for the model runner.

Method bodies here were moved verbatim out of ``gllm.runtime.model_runner``;
:class:`TokenizerMixin` is mixed into ``ModelRunner`` (and thereby
``OverlapModelRunner``) so every ``self`` reference and call site keeps its
original meaning.
"""

from typing import Dict, Optional

from gllm.tokenizers.tool_parsers import normalize_chat_template_messages
from gllm.utils import unify_decode


class TokenizerMixin:
    def encode(
        self,
        messages,
        chat: bool = False,
        has_mm: bool = False,
        chat_template_kwargs: Optional[Dict] = None,
        tools: Optional[list] = None,
    ):
        # Per-request chat-template variables (e.g. ``{"thinking": False}`` /
        # ``{"enable_thinking": False}``) forwarded straight from the request's
        # ``chat_template_kwargs``. Different chat templates gate "thinking"
        # mode via different variable names (Qwen3/3.5 read ``enable_thinking``,
        # Kimi-K2.5 reads ``thinking``); Jinja silently ignores undefined
        # template variables, so a client can send both. When omitted, the
        # model's own chat-template default applies (there is no server-wide
        # thinking flag anymore).
        #
        # ``tools`` (the request's OpenAI-style function schemas) are forwarded
        # so the chat template renders the model's tool-declaration block (e.g.
        # Kimi's ``<|im_system|>tool_declare<|im_middle|>...``). Without this the
        # model never learns the tools exist and answers as if it had none.
        template_kwargs = dict(chat_template_kwargs or {})
        if tools:
            template_kwargs["tools"] = tools
        if chat:
            normalize_chat_template_messages(messages)
            # OpenAI-style requests may send ``content: null`` (e.g. an assistant
            # turn that only carries ``tool_calls``). Many chat templates assume
            # ``content`` is a str or list and iterate it in the non-string
            # branch, so a None surfaces as ``TypeError: 'NoneType' object is not
            # iterable`` mid-render. Normalize null content to "" before render.
            for message in messages:
                if isinstance(message, dict) and message.get("content") is None:
                    message["content"] = ""
            deepseek_encoder = None
            if self._deepseek_encoder_variant is not None:
                from gllm.tokenizers.deepseek_official import load_deepseek_encoder

                deepseek_encoder = load_deepseek_encoder(
                    self.model_path, self._deepseek_encoder_variant
                )
            if deepseek_encoder is not None and (not self.use_mm or not has_mm):
                # DeepSeek-V3.2/V4: render with the checkpoint's official encoder
                # (reference DSML format) instead of a Jinja chat template.
                from gllm.tokenizers.deepseek_official import (
                    apply_deepseek_chat_template,
                )

                out = apply_deepseek_chat_template(
                    deepseek_encoder,
                    messages,
                    self.tokenizer,
                    tokenize=True,
                    **template_kwargs,
                )
            elif not self.use_mm or not has_mm:
                out = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **template_kwargs,
                )
            elif self.is_kimi_mm:
                # Kimi's chat template renders one ``<|media_pad|>`` per image
                # and a ``<|kimi_k25_video_placeholder|>`` per video, neither of
                # which its processor expands (unlike Qwen-VL). Render the text,
                # then ``build_kimi_input_ids`` splices video placeholders into
                # per-chunk prompts and expands every ``<|media_pad|>`` to the
                # exact per-item embedding count, so the downstream
                # ``is_multimodal`` mask has one True per produced vision
                # embedding. Counts come from the processor's own calculator,
                # guaranteeing they match the grids the vision tower will emit.
                out = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **template_kwargs,
                )
                if isinstance(out, (list, tuple)):
                    out = out[0]
                from gllm.models.kimi_k25 import build_kimi_input_ids

                return build_kimi_input_ids(
                    out,
                    messages,
                    self.processor,
                    self.tokenizer,
                    self.model_loader.config.media_placeholder_token_id,
                )
            else:
                out = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    **template_kwargs,
                )[0]
        else:
            out = self.tokenizer.encode(messages)
        # transformers >= 5.x ``apply_chat_template`` returns a
        # ``BatchEncoding`` (dict-like) when ``return_dict`` defaults to
        # True; older versions returned a flat ``List[int]``. Normalize
        # here so downstream code can always treat the result as a token
        # id list.
        if hasattr(out, "input_ids"):
            out = out.input_ids
        elif isinstance(out, dict) and "input_ids" in out:
            out = out["input_ids"]
        return out

    def decode(self, token_ids):
        return unify_decode(self.tokenizer, token_ids)

    def encode_skeleton(self, messages, chat_template_kwargs: Optional[Dict] = None):
        """Text-only tokenization with one sentinel per mm item (design §5.4).

        Used by the disaggregated LM frontend instead of the multimodal
        ``processor.apply_chat_template``: no pixels are opened or processed
        here, and each image/video collapses to a single placeholder id that
        the LM PP0 later expands to ``N_vis_i`` tokens. Returns the skeleton
        token-id list. ``chat_template_kwargs`` carries per-request chat-template
        variables (e.g. ``{"thinking": False}``) straight from the request.
        """
        from gllm.multimodal.common import tokenize_text_only

        cfg = self.model_loader.config
        skel = tokenize_text_only(
            self.tokenizer,
            messages,
            image_token_id=int(cfg.image_token_id),
            video_token_id=int(cfg.video_token_id),
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs,
        )
        return skel.token_ids
