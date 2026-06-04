"""Explicit configuration for encoder disaggregation.

Replaces the ``GLLM_DISAGG_*`` / ``GLLM_SKIP_*`` environment variables that
``lm_server`` / ``encoder_server`` used to set from parsed CLI args and that the
worker + model loader read back across the ``spawn`` process boundary. The
object is built once in the entrypoint and threaded explicitly:

* ``LLM -> ModelRunner -> ModelLoader`` for the role flags
  (``skip_visual`` / ``skip_language``), and
* ``LLM -> Worker`` for the LM-side manager params, where it is pickled to the
  spawned worker subprocess together with the rest of the worker state.

Operator / test runtime knobs that have no CLI argument and are set directly in
the environment at launch time (``GLLM_DISAGG_OVERLAP``,
``GLLM_DISAGG_REDISPATCH_TIMEOUT_S``, ``GLLM_DISAGG_MAX_REDISPATCH``,
``GLLM_ENC_FAIL_FIRST_N``) intentionally stay as env vars and are NOT part of
this config.

Kept dependency-free (stdlib only) so the monolith import path can pull it in
without dragging in NIXL / torch.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DisaggConfig:
    """Encoder-disaggregation config threaded from the entrypoint downward."""

    # --- Model-loader role flags (consumed in ModelLoader.load_config) ---
    # LM node: do not construct / load the vision tower; visual embeddings
    # arrive over NIXL from the encoder process.
    skip_visual: bool = False
    # Encoder node: construct ONLY the vision tower (no language model, KV
    # cache, scheduler, or sampler).
    skip_language: bool = False

    # --- LM-side manager (built on the PP0 driver in
    # ``Worker._maybe_init_disagg``). When ``is_lm`` is False the worker runs as
    # a plain monolith / text-only LM and never builds an ``LMDisaggManager``.
    is_lm: bool = False
    discovery_endpoint: str = ""
    # ``None`` -> the worker derives ``f"lm{rank}"`` (the rank is only known in
    # the subprocess).
    lm_id: Optional[str] = None
    processor_config_hash: str = ""
    advertise_host: str = "127.0.0.1"
    meta_bind: str = "tcp://0.0.0.0:0"
    # ``None`` -> fall back to the ``LMDisaggManager`` default.
    num_slots: Optional[int] = None
    max_vis_tokens: Optional[int] = None
    encoder_dp: int = 1
    nixl_backend: str = "UCX"
