"""The in-memory LoRA loader against vLLM's real LoRA classes.

The staged tensors are handed to ``LoRAModel.from_lora_tensors`` by reference
and kept for LRU rebuilds, while vLLM scales ``lora_b`` in place at load. The
trainer therefore publishes ``lora_alpha == r`` with the scale folded into
``lora_B``; this test pins that the loader refuses anything else, and that an
adapter in vLLM's flat 3D MoE layout is stamped ``is_3d_lora_weight`` on both
load paths (under ``enable_mixed_moe_lora_format`` vLLM otherwise hands the
unsplit ``experts`` tensor to ``FusedMoEWithLoRA.set_lora``).

Run with: uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/weight_sync/test_lora_in_memory_load.py
"""

from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.patches.vllm import patch_lora_in_memory as patch
from skyrl.backends.skyrl_train.weight_sync.lora_target import in_memory_lora_path

pytest.importorskip("vllm", reason="drives vLLM's LoRAModel and LoRALayerWeights directly")
pytestmark = pytest.mark.vllm

from vllm.lora.lora_model import LoRAModel, MoEEPLoadSpec  # noqa: E402
from vllm.lora.peft_helper import PEFTHelper  # noqa: E402

RANK, HIDDEN, INTER, EXPERTS = 4, 16, 8, 4


@pytest.fixture(autouse=True)
def _clear_staged():
    for name in patch.staged_adapter_names():
        patch.discard_in_memory_adapter(name)
    yield
    for name in patch.staged_adapter_names():
        patch.discard_in_memory_adapter(name)


def _tensors(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    t = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(RANK, HIDDEN, generator=g),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(HIDDEN, RANK, generator=g),
    }
    for expert in range(4):
        t[f"base_model.model.model.layers.0.mlp.experts.{expert}.down_proj.lora_A.weight"] = torch.randn(
            RANK, INTER, generator=g
        )
        t[f"base_model.model.model.layers.0.mlp.experts.{expert}.down_proj.lora_B.weight"] = torch.randn(
            HIDDEN, RANK, generator=g
        )
    return t


def _flat_3d_tensors(seed: int = 0) -> dict[str, torch.Tensor]:
    """The layout ``_convert_moe_experts_lora_to_vllm`` emits for fused-expert checkpoints."""
    g = torch.Generator().manual_seed(seed)
    prefix = "base_model.model.model.layers.0.mlp.experts"
    return {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(RANK, HIDDEN, generator=g),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(HIDDEN, RANK, generator=g),
        # gate_up_proj under experts.base_layer, down_proj under bare experts.
        f"{prefix}.base_layer.lora_A.weight": torch.randn(RANK * EXPERTS, HIDDEN, generator=g),
        f"{prefix}.base_layer.lora_B.weight": torch.randn(2 * INTER, RANK * EXPERTS, generator=g),
        f"{prefix}.lora_A.weight": torch.randn(RANK * EXPERTS, INTER, generator=g),
        f"{prefix}.lora_B.weight": torch.randn(HIDDEN, RANK * EXPERTS, generator=g),
    }


def _peft_config(lora_alpha: int) -> dict:
    return {
        "r": RANK,
        "lora_alpha": lora_alpha,
        "target_modules": ["q_proj", "down_proj"],
        "bias": "none",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
    }


def _manager(fused_experts: bool = False, gpt_oss_mixed: bool = False):
    class _AdapterManager:
        # A 2D MoE model lists each expert's projections under "experts"; a
        # fused-expert (3D) model has no expert entry in the packed mapping.
        supported_lora_modules = ["qkv_proj", "experts"]
        packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
        if not fused_experts:
            packed_modules_mapping["experts"] = [f"experts.{e}.down_proj" for e in range(4)]
        model = SimpleNamespace(
            hf_to_vllm_mapper=None,
            lora_skip_prefixes=None,
            config=SimpleNamespace(architectures=["GptOssForCausalLM" if gpt_oss_mixed else "Qwen3MoeForCausalLM"]),
        )
        _enable_mixed_moe_lora_format = gpt_oss_mixed
        # EP rank 1 of 2 with 2 local experts owns experts 2 and 3.
        moe_ep_load_spec = (
            None if fused_experts else MoEEPLoadSpec(ep_rank=1, local_num_experts=2, global_num_experts=4)
        )

    return SimpleNamespace(
        _adapter_manager=_AdapterManager(),
        _lora_model_cls=LoRAModel,
        lora_config=SimpleNamespace(
            lora_dtype=torch.float32, max_lora_rank=RANK, lora_extra_vocab_size=0, fully_sharded_loras=False
        ),
        device="cpu",
        vocab_size=32000,
    )


def _request(name: str):
    return SimpleNamespace(lora_path=in_memory_lora_path(name), lora_name=name, lora_int_id=7, is_3d_lora_weight=False)


def test_unfolded_alpha_is_refused_before_anything_is_built():
    tensors = _tensors()
    patch.stage_in_memory_adapter("t", tensors, _peft_config(lora_alpha=2 * RANK))
    with pytest.raises(ValueError, match="lora_alpha=8 != r=4"):
        patch._patched_load_adapter(_manager(), _request("t"))
    # Refusal leaves the stage in place for a corrected resync.
    assert patch.staged_adapter_names() == ["t"]


def test_flat_3d_moe_layout_is_stamped_on_the_in_memory_path():
    patch.stage_in_memory_adapter("t", _flat_3d_tensors(), _peft_config(lora_alpha=RANK))
    lora = patch._patched_load_adapter(_manager(fused_experts=True), _request("t"))
    assert "model.layers.0.mlp.experts.base_layer" in lora.loras
    assert lora.is_3d_lora_weight is True


def test_per_expert_layout_is_not_stamped():
    patch.stage_in_memory_adapter("t", _tensors(), _peft_config(lora_alpha=RANK))
    lora = patch._patched_load_adapter(_manager(), _request("t"))
    assert lora.is_3d_lora_weight is False


def test_flat_3d_moe_layout_is_stamped_on_the_directory_path(monkeypatch):
    # The directory path is vLLM's own loader; only the stamp is ours.
    def _original(self, lora_request):
        lora = LoRAModel.from_lora_tensors(
            lora_model_id=lora_request.lora_int_id,
            tensors=_flat_3d_tensors(),
            peft_helper=PEFTHelper.from_dict(_peft_config(lora_alpha=RANK)),
            device="cpu",
            dtype=torch.float32,
        )
        lora.is_3d_lora_weight = lora_request.is_3d_lora_weight
        return lora

    monkeypatch.setattr(patch, "_ORIGINAL_LOAD_ADAPTER", _original)
    request = SimpleNamespace(lora_path="/tmp/adapter", lora_name="t", lora_int_id=7, is_3d_lora_weight=False)
    assert patch._patched_load_adapter(_manager(fused_experts=True), request).is_3d_lora_weight is True


def test_gpt_oss_gate_up_rows_land_positionally_in_mixed_mode():
    # vLLM's mixed path splits w1 = B[::2], w3 = B[1::2] and the 2D wrapper adds them
    # to the contiguous halves of the interleaved w13 output; after the permutation
    # that split must give back the HF rows [0, I) and [I, 2I) unchanged.
    tensors = _flat_3d_tensors()
    key = "base_model.model.model.layers.0.mlp.experts.base_layer.lora_B.weight"
    hf_rows = tensors[key].clone()
    patch.stage_in_memory_adapter("t", tensors, _peft_config(lora_alpha=RANK))
    lora = patch._patched_load_adapter(_manager(fused_experts=True, gpt_oss_mixed=True), _request("t"))
    b = lora.loras["model.layers.0.mlp.experts.base_layer"].lora_b
    torch.testing.assert_close(torch.cat([b[::2], b[1::2]]), hf_rows)
    # The staged tensor is untouched, so an LRU rebuild permutes from HF order again.
    torch.testing.assert_close(tensors[key], hf_rows)


def test_gpt_oss_gate_up_rows_untouched_outside_mixed_mode():
    tensors = _flat_3d_tensors()
    key = "base_model.model.model.layers.0.mlp.experts.base_layer.lora_B.weight"
    patch.stage_in_memory_adapter("t", tensors, _peft_config(lora_alpha=RANK))
    lora = patch._patched_load_adapter(_manager(fused_experts=True), _request("t"))
    torch.testing.assert_close(lora.loras["model.layers.0.mlp.experts.base_layer"].lora_b, tensors[key])
