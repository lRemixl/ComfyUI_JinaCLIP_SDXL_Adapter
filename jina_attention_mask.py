import logging
import math

import torch
import comfy.ldm.modules.attention as attention


logger = logging.getLogger("JinaCLIP-SDXL-Adapter")


def _merge_attention_mask(args, kwargs, ca_mask):
    if len(args) > 3:
        new_args = list(args)
        new_args[3] = ca_mask if new_args[3] is None else new_args[3] + ca_mask
        return tuple(new_args), kwargs

    new_kwargs = dict(kwargs)
    mask = new_kwargs.get("mask")
    new_kwargs["mask"] = ca_mask if mask is None else mask + ca_mask
    return args, new_kwargs


def _format_cross_attention_mask(x, context, attn2):
    if context is None or not isinstance(context, torch.Tensor) or not hasattr(attn2, "to_k"):
        return context, None

    expected_dim = attn2.to_k.in_features
    if context.shape[-1] != expected_dim + 1:
        return context, None

    ca_mask = context[..., -1]
    context = context[..., :-1]

    batch_size, query_len = x.shape[:2]
    key_len = ca_mask.shape[-1]
    mask_batch = ca_mask.shape[0]

    if mask_batch != batch_size and mask_batch > 0:
        repeats = math.ceil(batch_size / mask_batch)
        ca_mask = ca_mask.repeat(repeats, 1)[:batch_size]
        mask_batch = batch_size

    additive_mask = (1.0 - ca_mask) * -10000.0
    ca_mask = additive_mask.view(mask_batch, 1, 1, key_len)
    ca_mask = ca_mask.expand(mask_batch, 1, query_len, key_len)
    return context, ca_mask.to(device=x.device, dtype=x.dtype)


def install_cross_attention_mask_patch():
    block_cls = attention.BasicTransformerBlock

    if getattr(block_cls, "_jina_mask_patch_installed", False):
        return

    if not hasattr(block_cls, "_jina_original_forward"):
        block_cls._jina_original_forward = block_cls.forward

    def jina_btb_forward(self, x, context=None, transformer_options=None, *args, **kwargs):
        if transformer_options is None:
            transformer_options = {}

        context, ca_mask = _format_cross_attention_mask(x, context, getattr(self, "attn2", None))
        if ca_mask is None:
            return self._jina_original_forward(x, context, transformer_options, *args, **kwargs)

        original_attn2_forward = self.attn2.forward

        def wrapped_attn2_forward(*attn_args, **attn_kwargs):
            attn_args, attn_kwargs = _merge_attention_mask(attn_args, attn_kwargs, ca_mask)
            return original_attn2_forward(*attn_args, **attn_kwargs)

        self.attn2.forward = wrapped_attn2_forward
        try:
            return self._jina_original_forward(x, context, transformer_options, *args, **kwargs)
        finally:
            self.attn2.forward = original_attn2_forward

    block_cls.forward = jina_btb_forward
    block_cls._jina_mask_patch_installed = True
    logger.info("Patched ComfyUI BasicTransformerBlock for Jina cross-attention masks.")


def build_conditioning(prompt_embeds, pooled_output, attention_mask, use_cross_attention_mask):
    if not use_cross_attention_mask:
        return [[prompt_embeds, {"pooled_output": pooled_output}]], ""

    mask = attention_mask.to(dtype=prompt_embeds.dtype, device=prompt_embeds.device).unsqueeze(-1)
    prompt_embeds = torch.cat([prompt_embeds, mask], dim=-1)
    return [[prompt_embeds, {"pooled_output": pooled_output}]], " (+1D cross-attention mask)"