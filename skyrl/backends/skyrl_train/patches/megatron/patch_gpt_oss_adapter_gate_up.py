"""Runtime backport of Megatron-Bridge PR #6244 (651bf8dc5) for the pinned Bridge revision.

Megatron stores GPT-OSS's fused expert ``linear_fc1`` as ``[all gates; all ups]``
while HF (and vLLM) interleave the rows (gate0, up0, gate1, ...). The base weight
mapping converts between the two, but the adapter export used generic mappings,
so expert ``gate_up_proj`` ``lora_B`` left the trainer in Megatron order and every
consumer applied the trained gate/up update to the wrong rows.
"""

from __future__ import annotations

import torch


def _uninterleave(elem: torch.Tensor) -> torch.Tensor:
    gate, up = torch.chunk(elem, 2, dim=0)
    output = torch.empty_like(elem)
    output[::2, ...] = gate
    output[1::2, ...] = up
    return output


def apply_gpt_oss_adapter_gate_up_patch() -> None:
    """Install #6244's ``GPTOSSBridge._get_fused_adapter_linear_out_slices``."""
    from megatron.bridge.models.gpt_oss.gpt_oss_bridge import GPTOSSBridge

    # Skip the backport when Bridge includes #6244.
    if "_get_fused_adapter_linear_out_slices" in GPTOSSBridge.__dict__:
        return

    def _get_fused_adapter_linear_out_slices(
        self, megatron_model, base_hf_weight_names, linear_out_tensor, is_expert=False
    ):
        if is_expert and len(base_hf_weight_names) == 1 and base_hf_weight_names[0].endswith(".experts.gate_up_proj"):
            from megatron.core.utils import unwrap_model

            # Each expert-TP shard carries its own [gate, up] halves.
            config = unwrap_model(megatron_model)[0].config
            shards = torch.chunk(linear_out_tensor, config.expert_tensor_parallel_size or 1, dim=0)
            return {base_hf_weight_names[0]: torch.cat([_uninterleave(shard) for shard in shards], dim=0)}
        return super(GPTOSSBridge, self)._get_fused_adapter_linear_out_slices(
            megatron_model, base_hf_weight_names, linear_out_tensor, is_expert=is_expert
        )

    GPTOSSBridge._get_fused_adapter_linear_out_slices = _get_fused_adapter_linear_out_slices
