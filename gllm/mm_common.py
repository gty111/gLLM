"""Shared multimodal helpers for encoder disaggregation.

Phase 2a: the Frontend must tokenize the *text only* (no image IO, no
processor, no pixel work -- design §3.1 / §4.4) and emit a **skeleton**
token-id list with exactly one placeholder sentinel per mm item. The LM PP0
later expands each sentinel into ``N_vis_i`` ``<|image_pad|>`` tokens once the
encoder reports ``num_tokens_i`` (design §5.4).

Validated invariant (Qwen3.5-VL): expanding the skeleton's i-th sentinel into
``N_vis_i`` copies of the same placeholder id byte-reconstructs the monolith's
``processor.apply_chat_template(..., tokenize=True)`` output exactly, so the LM
stays token-for-token equivalent with the monolith.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import List, Optional


def processor_config_hash(model_path: str) -> str:
    """Deterministic hash of a model's multimodal-processor config (§5.4.4).

    The Encoder runs the full image processor while the LM only keeps config
    scalars; both must agree on pixel preprocessing for the per-item content
    hashes (and thus prefix-cache keys) to match the monolith. We hash the
    on-disk processor/config JSON, which is byte-identical wherever the same
    checkpoint is loaded, and compare it during the discovery handshake;
    a mismatch rejects the connection instead of silently diverging.
    """
    candidates = (
        "preprocessor_config.json",
        "processor_config.json",
        "config.json",
    )
    h = hashlib.sha256()
    found = False
    for fn in candidates:
        p = os.path.join(model_path, fn)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "rb") as f:
                data = f.read()
        except OSError:
            continue
        h.update(fn.encode())
        h.update(b"\0")
        h.update(data)
        found = True
    if not found:
        # No config on disk (e.g. dummy load) -- fall back to the path so two
        # processes pointed at the same checkpoint still agree.
        h.update(os.path.abspath(model_path).encode())
    return h.hexdigest()


@dataclass
class MmSkeleton:
    """Frontend output for a (possibly) multimodal prompt.

    ``token_ids`` is the text-only tokenization with one sentinel per item.
    ``sentinel_positions[i]`` / ``modalities[i]`` describe the i-th item in
    prompt order; ``num_items`` == K.
    """

    token_ids: List[int]
    sentinel_positions: List[int] = field(default_factory=list)
    modalities: List[str] = field(default_factory=list)

    @property
    def num_items(self) -> int:
        return len(self.sentinel_positions)


def tokenize_text_only(
    tokenizer,
    messages,
    *,
    image_token_id: int,
    video_token_id: int,
    add_generation_prompt: bool = True,
    enable_thinking: Optional[bool] = None,
) -> MmSkeleton:
    """Tokenize ``messages`` with the *text* chat template (no image expansion).

    Uses ``tokenizer.apply_chat_template`` (NOT the multimodal
    ``processor.apply_chat_template``): the plain tokenizer template inserts a
    single placeholder token per image/video reference and never opens or
    processes the pixels. Returns the skeleton + per-item sentinel positions.

    ``enable_thinking`` is only forwarded when not ``None``. For multimodal
    prompts leave it ``None`` so the output matches the monolith's mm template
    (which passes no thinking flag); for pure-text prompts the caller may pass
    the server's ``use_thinking`` to match the monolith text path.
    """
    kwargs = dict(tokenize=True, add_generation_prompt=add_generation_prompt)
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    out = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(out, "input_ids"):
        token_ids = out.input_ids
    elif isinstance(out, dict) and "input_ids" in out:
        token_ids = out["input_ids"]
    else:
        token_ids = out
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    token_ids = list(token_ids)

    positions: List[int] = []
    modalities: List[str] = []
    for pos, tid in enumerate(token_ids):
        if tid == image_token_id:
            positions.append(pos)
            modalities.append("image")
        elif tid == video_token_id:
            positions.append(pos)
            modalities.append("video")
    return MmSkeleton(
        token_ids=token_ids,
        sentinel_positions=positions,
        modalities=modalities,
    )


def expand_sentinel(
    token_ids: List[int],
    sentinel_pos: int,
    num_tokens: int,
    pad_id: int,
) -> List[int]:
    """Return a copy of ``token_ids`` with the single sentinel at
    ``sentinel_pos`` replaced by ``num_tokens`` copies of ``pad_id``.

    The LM PP0 applies this in item order; after applying item ``i`` the
    positions of later sentinels shift by ``num_tokens_i - 1`` (handled by the
    aggregator that re-locates each sentinel before expanding -- design
    §5.4.2). This standalone helper expands exactly one sentinel.
    """
    assert 0 <= sentinel_pos < len(token_ids)
    return token_ids[:sentinel_pos] + [pad_id] * num_tokens + token_ids[sentinel_pos + 1:]
