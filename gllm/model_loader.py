import glob
import os
from typing import Tuple

import torch
from huggingface_hub import snapshot_download
from logger import logger
from safetensors import safe_open
from transformers import AutoConfig, GenerationConfig

from gllm.models.chatglm import ChatGLMForCausalLM
from gllm.models.deepseek_v2 import DeepseekV2ForCausalLM
from gllm.models.llama import LlamaForCausalLM
from gllm.models.mixtral import MixtralForCausalLM
from gllm.models.qwen2 import Qwen2ForCausalLM
from gllm.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from gllm.models.qwen2_moe import Qwen2MoeForCausalLM
from gllm.models.qwen3 import Qwen3ForCausalLM
from gllm.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from gllm.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
from gllm.models.qwen3_moe import Qwen3MoeForCausalLM
from gllm.models.qwen3_vl import Qwen3VLForConditionalGeneration
from gllm.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from gllm.utils import get_lock


def _is_quantized_param(param: torch.Tensor) -> bool:
    """True for the per-tensor weight of a FP8 / INT8 quantized linear.

    Identified by dtype rather than by name: gllm Linear classes register
    the FP8 weight as ``self.weight`` and the per-block scale as
    ``self.weight_scale_inv``. We key on the low-precision dtype directly
    so the check is also robust to future int8 / int4 layouts that may
    not carry a sibling ``_scale_inv``.
    """
    return param.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8)


def _validate_ignored_layers(model, quantization_config) -> None:
    """Warn if any parameter in ``ignored_layers`` ended up quantized.

    Naming is intentionally a best-effort suffix match: the checkpoint
    typically writes entries like ``model.language_model.layers.0.mlp.gate``
    while gllm's ``named_parameters()`` may emit
    ``language_model.model.layers.0.mlp.gate.weight`` (PP / VL wrapping
    can re-arrange the common ancestor). Stripping the trailing
    ``.weight`` / ``.weight_scale_inv`` and doing an ``endswith`` against
    each ignored entry catches every reasonable layout we ship without
    requiring per-model prefix bookkeeping. False positives here are
    cheap (a warning); false negatives would degrade accuracy silently,
    which is exactly what we want to avoid.
    """
    ignored = quantization_config.get("ignored_layers") if isinstance(
        quantization_config, dict
    ) else getattr(quantization_config, "ignored_layers", None)
    if not ignored:
        return
    # Strip the ``model.`` / ``model.language_model.`` style root prefixes
    # so suffix-matching is robust to gllm's PP/VL parameter naming. Keep
    # both the original and stripped variants — different checkpoints use
    # different leading prefixes.
    ignored_suffixes = []
    for entry in ignored:
        ignored_suffixes.append(entry)
        for stripped_prefix in ("model.language_model.", "model.", "language_model."):
            if entry.startswith(stripped_prefix):
                ignored_suffixes.append(entry[len(stripped_prefix):])
    ignored_suffixes = tuple(set(ignored_suffixes))

    violations = []
    for name, param in model.named_parameters():
        if not _is_quantized_param(param):
            continue
        # Param names end in ``.weight`` (and sometimes ``.weight_scale_inv``
        # for FP8 scales — but those have float32 dtype so the dtype filter
        # above already excludes them). Strip the trailing dot-suffix once.
        module_name = name.rsplit(".", 1)[0] if "." in name else name
        if module_name.endswith(ignored_suffixes):
            violations.append((name, str(param.dtype)))

    if violations:
        sample = ", ".join(f"{n} ({d})" for n, d in violations[:5])
        more = f" (and {len(violations) - 5} more)" if len(violations) > 5 else ""
        logger.warning(
            f"quantization_config['ignored_layers'] contains {len(ignored)} "
            f"entries but {len(violations)} matching parameters are still "
            f"quantized: {sample}{more}. The model definition needs to "
            "construct these layers without a ``quant_config`` (e.g. via "
            "``ParallelLMHead`` or plain ``nn.Linear``) or route them to "
            "a sub-config whose ``quantization_config`` is None."
        )


def get_attr_from_config(config, attr_name):
    if hasattr(config, attr_name):
        return getattr(config, attr_name)
    elif hasattr(config, 'text_config') and hasattr(getattr(config, 'text_config'), attr_name):
        return getattr(getattr(config, 'text_config'), attr_name)
    elif hasattr(config, 'vision_config') and hasattr(getattr(config, 'vision_config'), attr_name):
        return getattr(getattr(config, 'vision_config'), attr_name)
    else:
        raise KeyError(f"Failed to get attribute '{attr_name}' from config.")


def propagate_tie_word_embeddings(
    config,
    propagate_to: Tuple[str, ...] = ("text_config",),
) -> None:
    """Align ``tie_word_embeddings`` between the top-level config and listed
    nested sub-configs (default: ``text_config``).

    Why this is necessary: HuggingFace's ``PretrainedConfig`` defaults
    ``tie_word_embeddings`` to ``True``. For multimodal/VL configs the
    top-level config carries the authoritative value (which matches the
    checkpoint's actual ``lm_head.weight`` layout), but ``text_config`` is
    instantiated separately and silently keeps the HF default ``True``.
    VL wrappers (``Qwen2_5_VL``, ``Qwen3VL``, ``Qwen3VLMoe``, …) construct
    their language sub-model from ``config.text_config``, so without this
    propagation the LM ties ``lm_head`` to ``embed_tokens`` even when the
    checkpoint ships a *separate* ``lm_head.weight`` (i.e. top-level
    ``tie_word_embeddings: false``). The model loader then loads
    ``embed_tokens`` into the tied weight and the real ``lm_head.weight``
    in the checkpoint is silently ignored — predictions become garbage
    (often deterministic-looking nonsense like ``.fetchone``) even though
    every transformer layer is computed correctly.

    Top-level wins: if the top-level has an explicit value (``True`` or
    ``False``) it overwrites the sub-config's HF-default value. This is
    safe because checkpoints store the LM-head layout decision exactly
    once at the top level.

    Mutates ``config`` in place. Safe to call repeatedly.
    """
    top = getattr(config, "tie_word_embeddings", None)
    if top is None:
        return
    for name in propagate_to:
        sub = getattr(config, name, None)
        if sub is None:
            continue
        sub_val = getattr(sub, "tie_word_embeddings", None)
        if sub_val != top:
            sub.tie_word_embeddings = top


def propagate_quantization_config(
    config,
    propagate_to: Tuple[str, ...] = ("text_config",),
) -> None:
    """Mirror ``quantization_config`` between the top-level :class:`PretrainedConfig`
    and the listed nested sub-configs (default: only ``text_config``).

    Why this is necessary: every VL wrapper in this repo (``Qwen2_5_VL``,
    ``Qwen3VL``, ``Qwen3_5``, ``Qwen3VLMoe``, ``Qwen3_5MoeForConditionalGeneration``)
    constructs its language sub-model from ``config.text_config``. But some
    checkpoints (notably Qwen3.5-MoE-FP8) store the FP8 ``quantization_config``
    only on the top-level config — the LM would then never observe the quant
    field and would silently fall back to bf16 weights, doubling memory and
    corrupting every FP8 matmul. The mirror also handles the reverse case
    (checkpoint puts the field on a nested sub-config only) so the bare
    ``getattr(config, 'quantization_config', None)`` further below in
    :meth:`ModelLoader.load_config` always sees a non-None value when any
    listed sub-config has one.

    The default ``propagate_to=("text_config",)`` is intentionally
    conservative: in the Qwen3.5-MoE-FP8 layout the vision tower stays in
    bf16 and would not understand the FP8 ``weight_block_size`` directives,
    so we must not silently force FP8 on it. A future fully-quantized VL
    checkpoint (vision + language both quantized) should opt-in explicitly
    via ``propagate_to=("text_config", "vision_config")`` — typically from
    an architecture-specific branch in :meth:`ModelLoader.load_config`.

    Explicit per-sub-config settings always win: this helper only fills
    sub-configs whose ``quantization_config`` is currently ``None``. A
    checkpoint that deliberately sets *different* quant configs on
    different sub-configs (mixed-precision) is therefore preserved.

    Mutates ``config`` in place. Safe to call repeatedly. No-op when
    neither the top-level config nor any listed sub-config carries a
    ``quantization_config``.
    """
    # Resolve "the" quant config: prefer the top-level field, else the first
    # non-None field on a listed sub-config (reverse propagation so callers
    # can read it back off the top-level config uniformly).
    resolved = getattr(config, "quantization_config", None)
    if resolved is None:
        for name in propagate_to:
            sub = getattr(config, name, None)
            sub_quant = (
                getattr(sub, "quantization_config", None) if sub is not None else None
            )
            if sub_quant is not None:
                resolved = sub_quant
                break
    if resolved is None:
        return

    # Forward propagate: top-level -> each listed sub-config, but only when
    # the sub-config doesn't already carry its own (explicit wins).
    if getattr(config, "quantization_config", None) is None:
        config.quantization_config = resolved
    for name in propagate_to:
        sub = getattr(config, name, None)
        if sub is None:
            continue
        if getattr(sub, "quantization_config", None) is None:
            sub.quantization_config = resolved


class ModelLoader:
    def __init__(self, load_format, model_path, max_num_batched_tokens):
        self.model_path = model_path
        self.max_num_batched_tokens = max_num_batched_tokens
        self.load_config()
        self.load_format = load_format

    def load_safetensors(self, path):
        weights_path = glob.glob(f"{path}/*.safetensors")
        for weight_path in weights_path:
            with safe_open(weight_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    self.weights[k] = f.get_tensor(k)
        return len(self.weights) != 0

    def load_bin(self, path):
        weights_path = glob.glob(f"{path}/*.bin")
        for weight_path in weights_path:
            self.weights.update(torch.load(weight_path, weights_only=True))
        return len(self.weights) != 0

    def load_weights_from_local(self, path):
        if self.load_safetensors(path):
            return True

        if self.load_bin(path):
            return True

        return False

    def load_weights_from_huggingface(self, path):
        try:
            with get_lock(path, None):
                cached_path = snapshot_download(
                    path,
                    allow_patterns=["*.safetensors", "*.bin"],
                    ignore_patterns=["original/**/*"],
                )
                return self.load_weights_from_local(cached_path)
        except Exception as e:
            raise Exception(f"Failed to load {self.model_path} because of {e}!")

    def load_weights(self):
        self.weights = {}

        if self.load_weights_from_local(self.model_path):
            return

        if self.load_weights_from_huggingface(self.model_path):
            return

        raise Exception(f"Failed to load {self.model_path} from local or huggingface!")

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        except Exception:
            self.generation_config = GenerationConfig.from_model_config(self.config)
        if self.config.dtype is not None:
            self.dtype = self.config.dtype
        else:
            assert hasattr(self.config, "text_config")
            self.dtype = self.config.text_config.dtype
        self.architecture = self.config.architectures[0]
        self.vocab_size = get_attr_from_config(self.config, "vocab_size")
        self.hidden_size = get_attr_from_config(self.config, "hidden_size")
        # Mirror ``quantization_config`` across the top-level config and
        # ``config.text_config`` so that VL wrappers — which pass
        # ``text_config`` into the language sub-model — always observe the
        # quant field. See :func:`propagate_quantization_config`.
        propagate_quantization_config(self.config)
        # Same problem for ``tie_word_embeddings``: VL configs often have
        # top-level ``tie_word_embeddings: false`` (separate ``lm_head.weight``
        # in the checkpoint) while ``text_config`` keeps HF's default ``True``,
        # which would cause VL LMs to silently tie ``lm_head`` to ``embed_tokens``
        # and ignore the real ``lm_head.weight``. See
        # :func:`propagate_tie_word_embeddings`.
        propagate_tie_word_embeddings(self.config)
        self.quantization_config = getattr(self.config, "quantization_config", None)
        self.config.use_mla = self.use_mla
        self.config.use_hybrid_state = self.use_hybrid_state
        self.config.max_num_batched_tokens = self.max_num_batched_tokens
        # Encoder-disaggregation role flag (docs/encoder_disaggregation_design.md
        # §4.3 / §7.2.1). The LM node sets ``GLLM_SKIP_VISUAL=1`` so the VL
        # wrapper does not construct / load the vision tower; the visual
        # embeddings instead arrive over NIXL from the encoder process. Read
        # from the environment so the monolith path stays untouched and we
        # avoid threading a flag through LLM -> ModelRunner -> ModelLoader.
        self.config.skip_visual = bool(
            int(os.environ.get("GLLM_SKIP_VISUAL", "0"))
        )
        # Complementary flag for the encoder process: construct ONLY the
        # vision tower (no language model, KV cache, scheduler, or sampler).
        self.config.skip_language = bool(
            int(os.environ.get("GLLM_SKIP_LANGUAGE", "0"))
        )

    @property
    def use_mla(self):
        return self.architecture in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]

    @property
    def use_mm(self):
        return self.architecture in ["Qwen2_5_VLForConditionalGeneration",
                                     "Qwen3VLForConditionalGeneration",
                                     "Qwen3VLMoeForConditionalGeneration",
                                     "Qwen3_5ForConditionalGeneration",
                                     "Qwen3_5MoeForConditionalGeneration"]

    @property
    def use_hybrid_state(self):
        """Whether the model has linear-attention (Mamba/GDN) layers that need
        a recurrent-state cache *in addition to* the regular KV cache.
        """
        return self.architecture in ["Qwen3_5ForConditionalGeneration",
                                     "Qwen3_5MoeForConditionalGeneration"]

    def get_model_type(self):
        model_type = None
        if self.architecture == "LlamaForCausalLM":
            model_type = LlamaForCausalLM
        elif self.architecture == "ChatGLMModel":
            model_type = ChatGLMForCausalLM
        elif self.architecture == "Qwen2ForCausalLM":
            model_type = Qwen2ForCausalLM
        elif self.architecture == "Qwen3ForCausalLM":
            model_type = Qwen3ForCausalLM
        elif self.architecture == "Qwen2MoeForCausalLM":
            model_type = Qwen2MoeForCausalLM
        elif self.architecture == "Qwen3MoeForCausalLM":
            model_type = Qwen3MoeForCausalLM
        elif self.architecture == "MixtralForCausalLM":
            model_type = MixtralForCausalLM
        elif (
            self.architecture == "DeepseekV2ForCausalLM"
            or self.architecture == "DeepseekV3ForCausalLM"
        ):
            model_type = DeepseekV2ForCausalLM
        elif self.architecture == "Qwen2_5_VLForConditionalGeneration":
            model_type = Qwen2_5_VLForConditionalGeneration
        elif self.architecture == "Qwen3VLForConditionalGeneration":
            model_type = Qwen3VLForConditionalGeneration
        elif self.architecture == "Qwen3VLMoeForConditionalGeneration":
            model_type = Qwen3VLMoeForConditionalGeneration
        elif self.architecture == "Qwen3_5ForConditionalGeneration":
            model_type = Qwen3_5ForConditionalGeneration
        elif self.architecture == "Qwen3_5MoeForConditionalGeneration":
            model_type = Qwen3_5MoeForConditionalGeneration
        else:
            raise Exception(f"Unsupported model: {self.architecture}")
        return model_type

    def load_model(self, mp_load_progress=None):
        model_type = self.get_model_type()

        torch.set_default_dtype(self.dtype)

        # Load weights to CPU memory
        if self.load_format == "auto":
            self.load_weights()

        torch.set_default_device("cuda")

        # Init model whose weights are on GPU memory
        free_gpu_memory_before, _ = torch.cuda.mem_get_info()
        model = model_type(self.config)
        free_gpu_memory_after, _ = torch.cuda.mem_get_info()
        model_size_gb = round(
            (free_gpu_memory_before - free_gpu_memory_after) / (2**30), 2
        )

        quant_method_log = (
            f", Quant method: {self.quantization_config['quant_method']}"
            if self.quantization_config
            else ""
        )
        logger.info(
            f"Model architecture: {self.architecture}, "
            f"Default dtype: {self.dtype}{quant_method_log}, "
            f"Model weights {model_size_gb} GB"
        )

        # Load weights from CPU memory to GPU memory
        if self.load_format == "auto":
            model.load_weights(self.weights, mp_load_progress)

        # Best-effort validation that ``quantization_config["ignored_layers"]``
        # is honored. gllm does not consume the field directly (no per-layer
        # ``prefix`` plumbing through the Linear/MoE classes); instead every
        # model is expected to either route ignored layers to a sub-config
        # without ``quantization_config`` (e.g. the vision tower) or
        # construct them as non-quantizable classes (``ParallelLMHead``,
        # plain ``nn.Linear`` for MoE routers). This check catches future
        # checkpoints whose ``ignored_layers`` list does not match the
        # conventions baked into the model definitions, so a silent FP8 of
        # an "ignored" layer surfaces as a clear warning at load time
        # rather than mysterious downstream NaNs / garbage tokens.
        if self.quantization_config is not None:
            _validate_ignored_layers(model, self.quantization_config)

        return model
