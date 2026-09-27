"""Multimodal preprocessing for :class:`ModelRunner`.

Method bodies here were moved verbatim out of ``gllm.runtime.model_runner``;
:class:`MmMixin` is mixed into ``ModelRunner`` (and thereby
``OverlapModelRunner``) so every ``self`` reference and call site keeps its
original meaning.
"""

import hashlib
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from attr import dataclass
from transformers.image_utils import load_images
from transformers.video_utils import load_video

from gllm.distributed.parallel_state import is_first_pp_rank
from gllm.layers.rotary_embedding import MRotaryEmbedding
from gllm.runtime.sequence import GenerationSequence


@dataclass
class EmbeddingInfo:
    prompt_positions: torch.Tensor = None
    mrope_position_delta: torch.Tensor = None
    # Only encoded visual rows are retained between prefill chunks. Text and
    # deepstack embeddings are materialized for the scheduled span each time.
    multimodal_embeddings: Optional[torch.Tensor] = None
    is_multimodal_cpu: Optional[torch.Tensor] = None
    # Disaggregated encoders may have delivered only a ready prefix. Refresh
    # the visual rows when the scheduler advances beyond this coverage.
    coverage_len: Optional[int] = None


# High-id offset for the synthetic ``pad_id``s spliced into the prefix-cache
# key. The flag bit ``1 << 30`` keeps these well above any real vocab id (the
# largest model in this repo, Qwen3.5, tops out around 250k) and below the
# default ``int64`` tokenizer ceiling. The low 30 bits carry 30 bits of the
# multimodal content hash so two distinct images produce different pad ids
# with overwhelming probability.
_MM_PAD_ID_BASE = 1 << 30
_MM_PAD_ID_MASK = _MM_PAD_ID_BASE - 1


def _concat_mrope_positions_pinned(
    positions: List[torch.Tensor],
) -> torch.Tensor:
    """Concatenate ``[3, tokens]`` MRoPE positions into pinned CPU memory.

    ``torch.concat`` returns pageable memory even when every input is on the
    CPU.  The following H2D copy is therefore synchronizing despite being
    requested with ``non_blocking=True``.  Materializing the final layout
    directly in pinned memory keeps that metadata upload asynchronous with the
    preceding model forward.
    """
    if not positions:
        return torch.empty(
            (3, 0), dtype=torch.long, device="cpu", pin_memory=True
        )
    total_tokens = sum(int(position.shape[1]) for position in positions)
    output = torch.empty(
        (3, total_tokens),
        dtype=positions[0].dtype,
        device="cpu",
        pin_memory=True,
    )
    offset = 0
    for position in positions:
        num_tokens = int(position.shape[1])
        output[:, offset : offset + num_tokens].copy_(position)
        offset += num_tokens
    return output


def _mm_pad_id_from_hash(mm_hash: bytes) -> int:
    return _MM_PAD_ID_BASE | (int.from_bytes(mm_hash[:4], "big") & _MM_PAD_ID_MASK)


def _hash_tensor_bytes(*tensors: torch.Tensor) -> bytes:
    """Stable digest over the concatenated raw bytes of one or more tensors.

    Vision-tower inputs (pixel_values, grid_thw, timestamps, ...) are CPU-
    side when this runs (forced by ``_mm_prepare_cpu``), so we can lift the
    underlying storage directly without an extra D2H copy.
    """
    h = hashlib.sha256()
    for t in tensors:
        if t is None:
            h.update(b"\x00")
            continue
        if t.device.type != "cpu":
            t = t.detach().cpu()
        t = t.contiguous()
        # Mix dtype + shape so two tensors with identical bytes but
        # different reinterpretations can't collide.
        h.update(str(t.dtype).encode())
        h.update(repr(tuple(t.shape)).encode())
        h.update(memoryview(t.numpy().tobytes()))
    return h.digest()


def _build_item_content_hash(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
) -> bytes:
    """Per-item content hash, byte-identical to the monolith's i-th item hash.

    The monolith (:meth:`ModelRunner._build_mm_content_hashes`) computes each
    item's digest as ``_hash_tensor_bytes(pixel_chunk_i, image_grid_thw[i])``
    where ``pixel_chunk_i`` is this item's slice of the concatenated
    ``pixel_values`` and ``image_grid_thw[i]`` is the 1-D ``[3]`` grid row.

    The encoder runs the processor on a *single* image, so its ``pixel_values``
    already equals ``pixel_chunk_i`` and its ``grid_thw`` is ``[1, 3]``; we take
    row 0 to match the monolith's 1-D grid tensor exactly. This determinism is
    what lets the LM's prefix-cache pad ids agree across the two paths.
    """
    if isinstance(grid_thw, torch.Tensor) and grid_thw.ndim == 2:
        thw = grid_thw[0]
    else:
        thw = grid_thw
    return _hash_tensor_bytes(pixel_values, thw)


class MultiModalEmbeddingCache:
    """LRU cache over ``model.embed_multimodal(**mm_input)`` outputs.

    Key is the prompt-level digest of all of a sequence's multimodal items
    (concatenation of per-item sha256s, computed once in
    :meth:`_mm_prepare_cpu`). Value is the per-item embedding tuple that
    ``embed_multimodal`` returns — i.e. the same shape the model expects to
    splice back into the input embeddings.

    Eviction is byte-aware so a single huge ViT output can't squat on the
    pool indefinitely; once the running total exceeds ``max_bytes`` we evict
    LRU until back under the cap.
    """

    def __init__(self, max_entries: int = 64, max_mb: float = 256.0):
        self._cache: "OrderedDict[bytes, tuple]" = OrderedDict()
        self.max_entries = max_entries
        self.max_bytes = int(max_mb * 1024 * 1024)
        self._cur_bytes = 0
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _size_of(value) -> int:
        if value is None:
            return 0
        total = 0
        for t in value:
            if isinstance(t, torch.Tensor):
                total += t.element_size() * t.numel()
        return total

    def get(self, key: Optional[bytes]):
        if key is None:
            return None
        v = self._cache.get(key)
        if v is None:
            self.misses += 1
            return None
        self.hits += 1
        self._cache.move_to_end(key)
        return v

    def put(self, key: Optional[bytes], value) -> None:
        if key is None or value is None:
            return
        sz = self._size_of(value)
        if sz > self.max_bytes:
            # Don't even try to cache something that wouldn't fit; the
            # eviction loop would just thrash.
            return
        if key in self._cache:
            self._cur_bytes -= self._size_of(self._cache[key])
            self._cache.move_to_end(key)
        self._cache[key] = value
        self._cur_bytes += sz
        # Evict by entry count first, then by byte budget.
        while len(self._cache) > self.max_entries or self._cur_bytes > self.max_bytes:
            _, evicted = self._cache.popitem(last=False)
            self._cur_bytes -= self._size_of(evicted)


class MmMixin:
    def extract_modify_mm(self, messages: Dict):
        mm_contents = {"image": [], "video": []}
        for message in messages:
            contents = message["content"]
            if type(contents) != list:
                continue
            for content in contents:
                if content["type"] == "image":
                    mm_contents["image"].append(content["image"])
                elif content["type"] == "video":
                    mm_contents["video"].append(content["video"])
                elif content["type"] == "image_url":
                    content["type"] = "image"
                    data = content["image_url"]
                    del content["image_url"]
                    if type(data) == dict:
                        data = data["url"]
                    content["image"] = data
                    mm_contents["image"].append(data)
                elif content["type"] == "video_url":
                    content["type"] = "video"
                    data = content["video_url"]
                    del content["video_url"]
                    if type(data) == dict:
                        data = data["url"]
                    content["video"] = data
                    mm_contents["video"].append(data)
        return (
            mm_contents
            if len(mm_contents["image"]) + len(mm_contents["video"]) != 0
            else None
        )

    def extract_mm_items_ordered(self, messages: List[Dict]):
        """Return the mm items as an ordered ``[(modality, content), ...]`` list.

        Encoder disaggregation needs the items in *prompt order* (matching the
        skeleton's sentinel order) so the LM can pair the i-th sentinel with the
        i-th encoder job. Call *after* :meth:`extract_modify_mm`, which has
        already normalized ``image_url``/``video_url`` -> ``image``/``video``.
        """
        items = []
        for message in messages:
            contents = message["content"]
            if type(contents) != list:
                continue
            for content in contents:
                if content["type"] == "image":
                    items.append(("image", content["image"]))
                elif content["type"] == "video":
                    items.append(("video", content["video"]))
        return items

    @torch.inference_mode()
    def _mm_prepare_cpu(self, seqs: List[GenerationSequence]) -> Dict:
        """CPU phase of :meth:`mm_prepare_inputs`.

        Computes mrope positions and collects per-seq prefill work to run in
        :meth:`_mm_prepare_gpu`. Decode seqs (``seq.computed_prompt``) only
        contribute positions and a token count: their embedding rows are
        re-written in one fused call by
        :meth:`OverlapModelRunner._fixup_vl_decode_embeddings` on the forward
        stream, so we skip the per-seq ``embed_input_ids`` launch (and the
        attendant ``aten::any`` / ``aten::clamp`` sync points) entirely.

        Returning a context dict (instead of going straight to GPU work) lets
        the overlap scheduler run this phase concurrently with the previous
        batch's GPU forward.
        """
        batch_positions: List[torch.Tensor] = []
        prefill_works: List[Dict] = []
        num_decode_tokens = 0
        in_decode = True

        for seq in seqs:
            if seq.computed_prompt:
                # Decode token: positions only; embed is deferred to fixup.
                # The scheduler places decode seqs before prefill seqs, so
                # the contiguous decode block always sits at the front.
                assert in_decode, (
                    "scheduler invariant violated: decode seqs must precede "
                    "prefill seqs within a batch"
                )
                if self.uses_mrope:
                    # mrope (Qwen-VL): decode positions are extrapolated from
                    # the prefill-time ``mrope_position_delta`` stashed in the
                    # embedding cache, so the entry must exist here.
                    embedding_info = self.embedding_cache[seq.seq_id]
                    position = MRotaryEmbedding.get_next_input_positions(
                        embedding_info.mrope_position_delta,
                        seq.computed_token_num,
                        seq.seq_len,
                    )
                    batch_positions.append(torch.tensor(position, device="cpu"))
                else:
                    # Kimi: plain 1-D positions. These are discarded by the
                    # caller (``set_mrope_position`` is skipped for Kimi; the
                    # real positions come from ``cal_and_set_input``), but we
                    # still append a correctly-shaped tensor to keep the
                    # downstream ``torch.concat`` happy. We deliberately do NOT
                    # read ``embedding_cache`` here: a Kimi decode seq does not
                    # need its prefill embedding (positions are a plain
                    # ``arange`` and the row is re-embedded by the decode fixup),
                    # and a text-only prompt may never have created an entry --
                    # touching it would raise ``KeyError`` and crash the engine.
                    batch_positions.append(
                        torch.arange(seq.computed_token_num, seq.seq_len, device="cpu")
                    )
                num_decode_tokens += seq.to_compute_token_num
                continue

            in_decode = False
            if seq.seq_id in self.disagg_embeds:
                # Encoder-disaggregation overlap: this seq was
                # admitted before all its visual embeddings landed. Embed only
                # the span-aligned *ready prefix*; rebuild when more items land.
                self._mm_disagg_collect(
                    seq,
                    self.disagg_embeds[seq.seq_id],
                    prefill_works,
                    batch_positions,
                )
                continue
            if seq.mm_contents is None:
                # Text has no cross-chunk visual embeddings or mrope offsets.
                # Materialize only the scheduled span, including on prefix hits
                # and re-prefill after preemption. A full-prompt embedding can
                # otherwise consume gigabytes even with a small prefill budget.
                input_ids_cpu, is_multimodal_cpu = self._mm_build_is_multimodal_cpu(
                    seq, seq.computed_token_num, seq.seq_len
                )
                positions = torch.arange(
                    seq.computed_token_num, seq.seq_len, device="cpu"
                )
                batch_positions.append(
                    positions.unsqueeze(0).expand(3, -1)
                    if self.uses_mrope
                    else positions
                )
                prefill_works.append(
                    {
                        "kind": "text",
                        "seq": seq,
                        "input_ids_cpu": input_ids_cpu,
                        "is_multimodal_cpu": is_multimodal_cpu,
                    }
                )
                continue
            cached_info = self.embedding_cache.get(seq.seq_id)
            if cached_info is None or cached_info.is_multimodal_cpu is None:
                # If the scheduler already ran ``_mm_precompute_hash`` for
                # this seq (required for multimodal prefix-cache correctness
                # -- see that method's docstring), reuse the cached
                # image_processor output and is_multimodal mask. Otherwise
                # build them now (non-prefix-cache configs and the
                # never-cached scheduler in tests land here).
                pre = getattr(seq, "_mm_precomputed", None)
                if pre is not None:
                    mm_input = pre["mm_input"]
                    image_grid_thw = pre["image_grid_thw"]
                    video_grid_thw = pre["video_grid_thw"]
                    input_ids_cpu = pre["input_ids_cpu"]
                    is_multimodal_cpu = pre["is_multimodal_cpu"]
                    mm_bundle_key = pre["mm_bundle_key"]
                    # Encoder-disaggregation: the per-item visual
                    # embeddings were produced on the encoder and NIXL-written
                    # into the LM slot pool, then cloned into this tuple by the
                    # LM disagg manager. When present, ``_mm_prepare_gpu`` uses
                    # them verbatim instead of running the (absent) local ViT.
                    mm_embeddings = pre.get("mm_embeddings")
                    # Single-use: drop the stash so a re-scheduled seq
                    # (preempt + resume) doesn't accidentally read stale
                    # tensors. The work it represents is now folded into
                    # ``embedding_cache[seq.seq_id]`` below.
                    seq._mm_precomputed = None
                else:
                    mm_embeddings = None
                    mm_input, image_grid_thw, video_grid_thw = self._mm_run_processor(
                        seq
                    )
                    input_ids_cpu, is_multimodal_cpu = self._mm_build_is_multimodal_cpu(
                        seq
                    )
                    mm_bundle_key, item_hashes = self._build_mm_content_hashes(
                        mm_input, image_grid_thw, video_grid_thw
                    )
                    if item_hashes:
                        seq.hash_token_ids = self._splice_mm_pad_ids(
                            seq.token_ids,
                            is_multimodal_cpu,
                            item_hashes,
                        )
                    else:
                        seq.hash_token_ids = None

                if self.uses_mrope:
                    prompt_positions, mrope_position_delta = (
                        MRotaryEmbedding.get_input_positions(
                            input_tokens=seq.token_ids,
                            hf_config=self.model.config,
                            image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw,
                            second_per_grid_ts=None,
                        )
                    )
                    batch_positions.append(
                        prompt_positions[:, seq.computed_token_num : seq.seq_len]
                    )
                else:
                    # Kimi: plain 1-D positions over the full prompt. Stored in
                    # EmbeddingInfo so decode can extrapolate; ``mrope_position
                    # _delta`` is unused (decode uses ``torch.arange``).
                    prompt_positions = torch.arange(len(seq.token_ids), device="cpu")
                    mrope_position_delta = None
                    batch_positions.append(
                        prompt_positions[seq.computed_token_num : seq.seq_len]
                    )

                prefill_works.append(
                    {
                        "kind": "uncached",
                        "seq": seq,
                        "input_ids_cpu": input_ids_cpu,
                        "is_multimodal_cpu": is_multimodal_cpu,
                        "mm_input": mm_input,
                        "mm_embeddings": mm_embeddings,
                        "prompt_positions": prompt_positions,
                        "mrope_position_delta": mrope_position_delta,
                        "mm_bundle_key": mm_bundle_key,
                    }
                )
            else:
                embedding_info = self.embedding_cache[seq.seq_id]
                if self.uses_mrope:
                    batch_positions.append(
                        embedding_info.prompt_positions[
                            :, seq.computed_token_num : seq.seq_len
                        ]
                    )
                else:
                    batch_positions.append(
                        embedding_info.prompt_positions[
                            seq.computed_token_num : seq.seq_len
                        ]
                    )
                prefill_works.append(
                    {
                        "kind": "cached",
                        "seq": seq,
                        "embedding_info": embedding_info,
                    }
                )

        # Qwen-VL packs positions as (3, N) and concatenates on the token axis
        # (dim=1); Kimi uses 1-D positions (dim=0). Kimi's result is discarded
        # by callers (``set_mrope_position`` is skipped) but we still build a
        # well-formed tensor.
        if self.uses_mrope:
            mrope_positions = _concat_mrope_positions_pinned(batch_positions)
        elif batch_positions:
            mrope_positions = torch.concat(batch_positions, dim=0)
        else:
            mrope_positions = None
        return {
            "prefill_works": prefill_works,
            "mrope_positions": mrope_positions,
            "num_decode_tokens": num_decode_tokens,
        }

    @staticmethod
    def _disagg_ready_len(st: "DisaggSeqState") -> int:
        """Length of the span-aligned ready prefix ``[0, ready_len)``.

        Stops at the first *not-yet-ready* image span start (in token order),
        regardless of whether a later item happens to be ready: a prefix that
        spanned an unready item would have more ``is_multimodal`` positions
        than gathered embeddings and the merge would misalign.
        """
        rl = st.prompt_len
        for i in range(st.num_items):
            if not st.item_ready[i]:
                rl = min(rl, st.item_span[i][0])
        return rl

    def _mm_disagg_collect(
        self,
        seq: GenerationSequence,
        st: "DisaggSeqState",
        prefill_works: List[Dict],
        batch_positions: List[torch.Tensor],
    ) -> None:
        """Build the prefill work for an overlap disagg seq.

        Positions come from the full-prompt mrope grid (all grids known once
        meta arrived). Encoded visual rows cover the ready prefix; refresh them
        (kind ``uncached``) whenever the scheduler advances past the cached
        ``coverage_len`` because more items became ready, otherwise the cached
        rows are re-sliced for the current chunk (kind ``cached``).
        """
        batch_positions.append(
            st.prompt_positions[:, seq.computed_token_num : seq.seq_len]
        )
        info = self.embedding_cache.get(seq.seq_id)
        need_build = info is None or info.is_multimodal_cpu is None or (
            info.coverage_len is not None and seq.seq_len > info.coverage_len
        )
        if not need_build:
            prefill_works.append({"kind": "cached", "seq": seq, "embedding_info": info})
            return
        ready_len = self._disagg_ready_len(st)
        # Gather the ready-prefix items in token-span order so the concatenated
        # embeddings line up 1-1 with the ``is_multimodal`` True positions.
        ready_items = [
            i for i in range(st.num_items) if st.item_span[i][1] <= ready_len
        ]
        ready_items.sort(key=lambda i: st.item_span[i][0])
        ready_embeds = tuple(st.item_embed[i] for i in ready_items)
        prefill_works.append(
            {
                "kind": "uncached",
                "seq": seq,
                "input_ids_cpu": st.input_ids_cpu[:ready_len],
                "is_multimodal_cpu": st.is_multimodal_cpu[:ready_len],
                "mm_input": {},
                "mm_embeddings": ready_embeds if ready_embeds else None,
                "prompt_positions": st.prompt_positions,
                "mrope_position_delta": st.mrope_position_delta,
                "mm_bundle_key": None,
                "coverage_len": ready_len,
            }
        )

    def _mm_run_processor(
        self, seq: GenerationSequence
    ) -> Tuple[Dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run image/video processors for ``seq.mm_contents``.

        Returns ``(mm_input, image_grid_thw, video_grid_thw)``. The
        grid tensors are forced to CPU because ``get_input_positions``
        (and our content hashing) does per-element Python indexing on
        them; leaving CUDA tensors there would trigger a D2H sync per
        element and serialize the prepare-input stage against the
        previous batch's forward.
        """
        mm_input: Dict = {}
        image_grid_thw: Optional[torch.Tensor] = None
        video_grid_thw: Optional[torch.Tensor] = None
        if seq.mm_contents is not None and self.is_kimi_mm:
            return self._mm_run_processor_kimi(seq)
        if seq.mm_contents is not None:
            if len(seq.mm_contents["image"]) != 0:
                images = load_images(seq.mm_contents["image"])
                images_input = self.image_processor(images=images)
                mm_input.update(images_input)
                image_grid_thw = images_input["image_grid_thw"]
            if len(seq.mm_contents["video"]) != 0:
                videos = []
                video_metadata = []
                for video_content in seq.mm_contents["video"]:
                    video_data, metadata = load_video(video_content)
                    videos.append(video_data)
                    video_metadata.append(metadata)
                videos_input = self.video_processor(
                    videos=videos,
                    video_metadata=video_metadata,
                )
                mm_input.update(videos_input)
                video_grid_thw = videos_input["video_grid_thw"]
        if isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.cpu()
        if isinstance(video_grid_thw, torch.Tensor):
            video_grid_thw = video_grid_thw.cpu()
        return mm_input, image_grid_thw, video_grid_thw

    def _mm_run_processor_kimi(
        self, seq: GenerationSequence
    ) -> Tuple[Dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Kimi-K2.5 image + video preprocessing.

        ``KimiK25VisionProcessor.preprocess`` takes a list of media dicts
        (``{"type":"image",...}`` or ``{"type":"video_chunk",...}``) and returns
        ``pixel_values`` (patchified, ``[sum(t*h*w), 3, ps, ps]``) plus
        ``grid_thws`` (``[num_items, 3]``; video chunks have ``t>1``). We build
        one combined media list in embed order -- all images first, then every
        video's temporal chunks -- matching ``build_kimi_input_ids``'s
        placeholder order and ``embed_multimodal``'s iteration. ``grid_thws`` is
        surfaced as ``image_grid_thw`` so the generic content-hashing path
        (``prod(dim=-1)`` + ``split``) covers every item, while ``grid_thws``
        stays in ``mm_input`` for ``embed_multimodal``.
        """
        from PIL import Image as _PILImage
        from transformers.image_utils import load_image as _hf_load_image

        from gllm.models.kimi_k25_vision import split_video_chunks

        medias = []
        for img_ref in seq.mm_contents["image"]:
            pil = (
                img_ref
                if isinstance(img_ref, _PILImage.Image)
                else _hf_load_image(img_ref)
            )
            medias.append({"type": "image", "image": pil})
        cfg = self.processor.media_processor.media_proc_cfg
        for vid_ref in seq.mm_contents["video"]:
            for chunk in split_video_chunks(vid_ref, cfg):
                medias.append(chunk)

        mm_input: Dict = {}
        image_grid_thw: Optional[torch.Tensor] = None
        if medias:
            preprocessed = self.processor.media_processor.preprocess(
                medias, return_tensors="pt"
            )
            mm_input["pixel_values"] = preprocessed["pixel_values"]
            mm_input["grid_thws"] = preprocessed["grid_thws"]
            image_grid_thw = preprocessed["grid_thws"]
        if isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.cpu()
        return mm_input, image_grid_thw, None

    def _mm_build_is_multimodal_cpu(
        self, seq: GenerationSequence, start: int = 0, end: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build CPU IDs and placeholder mask for the requested token span.

        Explicitly CPU-side: the repo sets the default device to CUDA
        via ``ModelLoader``, so a bare ``torch.tensor(...)`` would
        silently allocate on GPU and the ``torch.isin`` below would
        launch a kernel on the default stream -- defeating overlap with
        the previous batch's forward.
        """
        input_ids_cpu = torch.tensor(seq.token_ids[start:end], device="cpu")
        placeholder_token_id_cpu = torch.tensor(
            self.model.get_mm_placeholder_token_ids(), device="cpu"
        )
        is_multimodal_cpu = torch.isin(input_ids_cpu, placeholder_token_id_cpu)
        return input_ids_cpu, is_multimodal_cpu

    def _mm_precompute_hash(self, seq: GenerationSequence) -> None:
        """Pre-build ``seq.hash_token_ids`` before the scheduler's prefix
        cache lookup, so distinct multimodal items don't collide on the
        raw ``<|image_pad|>`` placeholder id.

        The scheduler calls ``pre_allocate_computed_page`` for every new
        seq *before* ``_mm_prepare_cpu`` runs. The prefix-cache hash reads
        ``seq.hash_token_ids`` when present and otherwise falls back to
        ``seq.token_ids``; without this hook the fallback makes every
        image-bearing request share the same placeholder ids at the image
        span, so a second request would silently reuse the first request's
        KV pages there (wrong-image answers).

        Side effects:
            * ``seq.hash_token_ids`` is populated when the prompt has
              at least one mm item, ``None`` otherwise.
            * ``seq._mm_precomputed`` stashes the heavy outputs of the
              image/video processor + the cpu masks so the later
              ``_mm_prepare_cpu`` pass does not repeat the work.
        """
        if not self.use_mm:
            return
        if seq.mm_contents is None:
            return
        if (
            seq.hash_token_ids is not None
            or getattr(seq, "_mm_precomputed", None) is not None
        ):
            return  # already built (e.g. preemption + re-schedule).

        mm_input, image_grid_thw, video_grid_thw = self._mm_run_processor(seq)
        input_ids_cpu, is_multimodal_cpu = self._mm_build_is_multimodal_cpu(seq)
        mm_bundle_key, item_hashes = self._build_mm_content_hashes(
            mm_input, image_grid_thw, video_grid_thw
        )
        if item_hashes:
            seq.hash_token_ids = self._splice_mm_pad_ids(
                seq.token_ids, is_multimodal_cpu, item_hashes
            )
        else:
            seq.hash_token_ids = None

        seq._mm_precomputed = {
            "mm_input": mm_input,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "input_ids_cpu": input_ids_cpu,
            "is_multimodal_cpu": is_multimodal_cpu,
            "mm_bundle_key": mm_bundle_key,
        }

    @staticmethod
    def _build_mm_content_hashes(
        mm_input: Dict,
        image_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
    ) -> Tuple[Optional[bytes], List[bytes]]:
        """Hash each MM item's content, return (prompt-level key, per-item).

        Per-item hash mixes pixel bytes + grid shape so two crops of the
        same image with different processor settings still differ. The
        prompt-level key is the concatenation of all per-item digests in
        the order they appear in ``mm_input`` (image items, then video
        items, mirroring :meth:`embed_multimodal`'s iteration). ``None`` is
        returned when there's nothing multimodal in the prompt — that
        signals downstream that the seq is text-only and falls back to the
        cheap ``token_ids`` cache key.
        """
        item_hashes: List[bytes] = []

        pixel_values = mm_input.get("pixel_values")
        if pixel_values is not None and image_grid_thw is not None:
            sizes = image_grid_thw.prod(dim=-1).tolist()
            if isinstance(pixel_values, torch.Tensor):
                chunks = pixel_values.split(sizes, dim=0)
            else:
                chunks = pixel_values
            for chunk, thw in zip(chunks, image_grid_thw):
                item_hashes.append(_hash_tensor_bytes(chunk, thw))

        pixel_values_videos = mm_input.get("pixel_values_videos")
        if pixel_values_videos is not None and video_grid_thw is not None:
            sizes = video_grid_thw.prod(dim=-1).tolist()
            if isinstance(pixel_values_videos, torch.Tensor):
                chunks = pixel_values_videos.split(sizes, dim=0)
            else:
                chunks = pixel_values_videos
            for chunk, thw in zip(chunks, video_grid_thw):
                item_hashes.append(_hash_tensor_bytes(chunk, thw))

        if not item_hashes:
            return None, []
        bundle = hashlib.sha256()
        for h in item_hashes:
            bundle.update(h)
        return bundle.digest(), item_hashes

    @staticmethod
    def _splice_mm_pad_ids(
        token_ids: List[int],
        is_multimodal_cpu: torch.Tensor,
        item_hashes: List[bytes],
    ) -> List[int]:
        """Return a copy of ``token_ids`` with placeholder spans rewritten.

        Each contiguous run of multimodal placeholders is replaced by a
        single ``pad_id`` derived from the next item's content hash, so the
        downstream :class:`PrefixSegment` key naturally diverges between
        prompts whose only difference is the image content. Mirrors
        sglang's ``pad_input_tokens`` trick adapted to gllm's flat-page
        cache layout.
        """
        mask = (
            is_multimodal_cpu.tolist()
            if isinstance(is_multimodal_cpu, torch.Tensor)
            else list(is_multimodal_cpu)
        )
        out = list(token_ids)
        n = len(out)
        i = 0
        item_idx = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < n and mask[j]:
                j += 1
            # ``item_hashes`` exhaustion would mean the processor produced
            # fewer MM items than there are placeholder spans, which
            # indicates a tokenizer/processor mismatch. We leave excess
            # spans untouched (falls back to the raw token id), which is
            # the safe-but-conservative behavior — at worst it widens the
            # cache hit set, never causing a false hit.
            if item_idx < len(item_hashes):
                pad_id = _mm_pad_id_from_hash(item_hashes[item_idx])
                for k in range(i, j):
                    out[k] = pad_id
                item_idx += 1
            i = j
        return out

    @torch.inference_mode()
    def _mm_prepare_gpu(self, ctx: Dict) -> Optional[torch.Tensor]:
        """GPU phase of :meth:`mm_prepare_inputs`.

        Runs each prefill seq's multimodal+text embed and produces a single
        ``input_embeddings`` tensor laid out as ``[decode_rows, prefill_rows]``.
        Decode rows are an uninitialized placeholder; they will be overwritten
        by :meth:`OverlapModelRunner._fixup_vl_decode_embeddings` on the
        forward stream right before the model runs, so the placeholder content
        is irrelevant.
        """
        device = self.input_hidden_states.device
        batch_embeddings: List[torch.Tensor] = []
        # Per-chunk deepstack tensors aligned 1-1 with ``batch_embeddings``.
        # ``None`` means "no deepstack contribution for this chunk" (decode
        # rows, text-only prompts, non-deepstack VL models). A single
        # buffer-write at the end of this method stitches the non-``None``
        # chunks into the right rows of the model's deepstack buffer.
        batch_deepstack: List[Optional[torch.Tensor]] = []
        for work in ctx["prefill_works"]:
            seq = work["seq"]
            visual_chunk = None
            if work["kind"] == "text":
                # Text takes the same embedding path, without encoded media.
                input_ids_cpu = work["input_ids_cpu"]
                mask_cpu = work["is_multimodal_cpu"]
                embedding_info = EmbeddingInfo(mrope_position_delta=0)
                self.embedding_cache[seq.seq_id] = embedding_info
            else:
                if work["kind"] == "uncached":
                    from gllm.models.utils import _flatten_embeddings

                    # These may come from a local encoder, the visual feature
                    # cache, or the ready prefix of a disaggregated encoder.
                    mm_embeddings = work.get("mm_embeddings")
                    mm_input = work["mm_input"]
                    if mm_embeddings is None and mm_input:
                        bundle_key = work.get("mm_bundle_key")
                        mm_embeddings = self.mm_embed_cache.get(bundle_key)
                        if mm_embeddings is None:
                            mm_embeddings = self.model.embed_multimodal(**mm_input)
                            if bundle_key is not None:
                                self.mm_embed_cache.put(bundle_key, mm_embeddings)
                    visual_rows = (
                        _flatten_embeddings(mm_embeddings)
                        if mm_embeddings is not None and len(mm_embeddings) > 0
                        else None
                    )
                    mask = work["is_multimodal_cpu"]
                    expected = int(mask.sum())
                    actual = visual_rows.shape[0] if visual_rows is not None else 0
                    if actual != expected:
                        raise ValueError(
                            f"Expected {expected} visual embedding rows, got {actual}"
                        )
                    embedding_info = EmbeddingInfo(
                        prompt_positions=work["prompt_positions"],
                        mrope_position_delta=work["mrope_position_delta"],
                        multimodal_embeddings=visual_rows,
                        is_multimodal_cpu=mask,
                        coverage_len=work.get("coverage_len"),
                    )
                    self.embedding_cache[seq.seq_id] = embedding_info
                else:
                    embedding_info = work["embedding_info"]

                start, end = seq.computed_token_num, seq.seq_len
                mask = embedding_info.is_multimodal_cpu
                if mask is None or end > mask.numel():
                    raise RuntimeError(
                        f"Visual metadata does not cover prefill span [{start}, {end})"
                    )
                input_ids_cpu = torch.tensor(seq.token_ids[start:end], device="cpu")
                mask_cpu = mask[start:end]
                visual_start = int(mask[:start].sum())
                visual_count = int(mask_cpu.sum())
                if visual_count:
                    # Slice by placeholder ordinal, not by image boundaries:
                    # chunks and prefix hits may start in the middle of an image.
                    visual_chunk = (
                        embedding_info.multimodal_embeddings[
                            visual_start : visual_start + visual_count
                        ],
                    )

            # One path for text, images and video: embed only this chunk, then
            # merge its visual rows (and produce only this chunk's deepstack).
            embed_result = self.model.embed_input_ids(
                input_ids_cpu.to(device, non_blocking=True),
                visual_chunk,
                mask_cpu.to(device, non_blocking=True),
            )
            if isinstance(embed_result, tuple):
                embedding, deepstack_chunk = embed_result
            else:
                embedding, deepstack_chunk = embed_result, None

            if seq.seq_len == seq.prompt_len:
                # Retain position metadata for decode, but no request-owned
                # visual tensors after prefill has completed.
                embedding_info.multimodal_embeddings = None
                embedding_info.is_multimodal_cpu = None
                self.disagg_embeds.pop(seq.seq_id, None)

            batch_embeddings.append(embedding)
            batch_deepstack.append(deepstack_chunk)

        num_decode_tokens = ctx["num_decode_tokens"]
        if num_decode_tokens > 0:
            # Placeholder rows; ``_fixup_vl_decode_embeddings`` re-embeds these
            # token positions in a single fused launch on the forward stream
            # after future-token resolution. ``empty`` is fine since the
            # contents are dead-on-arrival.
            placeholder = torch.empty(
                (num_decode_tokens, self.hidden_size),
                device=self.input_hidden_states.device,
                dtype=self.input_hidden_states.dtype,
            )
            batch_embeddings.insert(0, placeholder)
            # Decode rows must contribute zero deepstack residual (they
            # represent already-prefilled tokens whose visual residuals
            # are baked into the KV cache, not the input embedding).
            batch_deepstack.insert(0, None)

        if not batch_embeddings:
            return None

        # Stitch per-chunk deepstack tensors into the model's per-batch
        # buffer at the offsets matching the final concatenated layout.
        # This makes ``model._get_deepstack_input_embeds(num_tokens)``
        # return rows aligned 1-1 with ``hidden_states`` regardless of
        # prefix-cache hits or chunked prefill -- the deepstack residual
        # for a token T at batch row R will land exactly at buffer row R.
        if any(d is not None for d in batch_deepstack) and hasattr(
            self.model, "_set_deepstack_input_embeds"
        ):
            total_tokens = sum(e.shape[0] for e in batch_embeddings)
            # Zero positions that no chunk will write to (decode rows,
            # text-only chunks). ``_clear_deepstack_input_embeds`` after
            # the previous forward only zeroed up to that batch's row
            # count, so anything beyond it could still hold stale values.
            self.model._clear_deepstack_input_embeds(total_tokens)
            offset = 0
            for chunk, emb in zip(batch_deepstack, batch_embeddings):
                n = emb.shape[0]
                if chunk is not None:
                    self.model._set_deepstack_input_embeds(chunk, offset=offset)
                offset += n

        return torch.concat(batch_embeddings)

    @torch.inference_mode()
    def mm_prepare_inputs(self, seqs: List[GenerationSequence]):
        """Single-shot wrapper kept for the non-overlap worker path."""
        ctx = self._mm_prepare_cpu(seqs)
        input_embeddings = self._mm_prepare_gpu(ctx)
        return input_embeddings, ctx["mrope_positions"]

    def _fixup_vl_decode_embeddings(self, num_decode_tokens: int) -> None:
        """Re-embed decode-token IDs into the front of ``input_hidden_states``.

        ``_mm_prepare_gpu`` inserts an ``torch.empty()`` placeholder for the
        decode rows of every VL batch and relies on this method to overwrite
        those rows with the real text embeddings *before* the model forward
        reads them. Both the no-overlap base path (:meth:`forward`) and the
        overlap path (:meth:`OverlapModelRunner.run_batch_async`) must call
        this; otherwise the model consumes uninitialized memory for every
        decode token and silently produces garbage from the first decode
        step onward.
        """
        if (
            self.use_mm
            and is_first_pp_rank()
            and self.input_data.embedding_size > 0
            and num_decode_tokens > 0
        ):
            decode_embeds = self.model.language_model.model.embed_tokens(
                self.input_data.tokens[:num_decode_tokens]
            )
            self.input_hidden_states[:num_decode_tokens] = decode_embeds
