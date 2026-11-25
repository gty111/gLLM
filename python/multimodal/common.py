"""Shared multimodal helpers for encoder disaggregation.

The front end of an LM node renders a *skeleton* prompt with one placeholder
per media item (no image IO); the LM PP0 expands each placeholder into
``N_vis_i`` ``<|image_pad|>`` tokens once the encoder reports ``num_tokens_i``.
Expanding the skeleton this way reproduces the monolith's prompt exactly.
"""

from __future__ import annotations

import hashlib
import os


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
