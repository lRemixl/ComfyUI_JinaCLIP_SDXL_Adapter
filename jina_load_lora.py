import copy
import logging
import re

import torch
import folder_paths
import comfy.sd
import comfy.utils


logger = logging.getLogger("JinaCLIP-SDXL-Adapter")


class JinaAdapterLoRALoader:
    """Apply a LoRA to the SDXL UNet and the custom Jina adapter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "jina_adapter": ("JINA_ADAPTER",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "adapter_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "JINA_ADAPTER", "STRING")
    RETURN_NAMES = ("model", "jina_adapter", "info")
    FUNCTION = "load_lora"
    CATEGORY = "llm_sdxl/jina/advanced"

    def load_lora(self, model, jina_adapter, lora_name, model_weight, adapter_weight):
        info_lines = [f"LoRA: {lora_name}"]

        if model_weight == 0 and adapter_weight == 0:
            info_lines.append("Bypassed because both weights are 0.")
            return (model, jina_adapter, "\n".join(info_lines))

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise ValueError(f"LoRA file not found: {lora_name}")

        lora_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
        patched_model, _ = comfy.sd.load_lora_for_models(model, None, lora_dict, model_weight, 0)
        patched_adapter = copy.deepcopy(jina_adapter)

        lora_modules = self.group_adapter_lora_modules(lora_dict)
        applied_count, failed_stems = self.apply_adapter_lora(
            patched_adapter,
            lora_modules,
            adapter_weight,
        )

        info_lines.append(f"UNet weight applied: {model_weight}")
        info_lines.append(f"Adapter weight applied: {adapter_weight}")
        info_lines.append(f"Found {len(lora_modules)} adapter LoRA modules.")
        info_lines.append(f"Applied {applied_count} adapter modules.")

        if failed_stems:
            info_lines.append(f"Failed to apply {len(failed_stems)} adapter modules:")
            info_lines.extend(f"  - {stem}" for stem in failed_stems[:15])

        logger.info("Jina Adapter LoRA applied %s modules from %s.", applied_count, lora_name)
        return (patched_model, patched_adapter, "\n".join(info_lines))

    def group_adapter_lora_modules(self, lora_dict):
        lora_modules = {}
        suffixes = (
            ".lora_down.weight",
            ".lora_up.weight",
            ".alpha",
            ".scale",
            ".lora_down",
            ".lora_up",
        )

        for key, value in lora_dict.items():
            match = re.match(r"^lora_te\d*_(.+)$", key)
            if not match:
                continue

            rest = match.group(1)
            stem = rest
            suffix = ""
            for candidate in suffixes:
                if rest.endswith(candidate):
                    stem = rest[:-len(candidate)]
                    suffix = candidate
                    break

            module_tensors = lora_modules.setdefault(stem, {})
            if "down" in suffix:
                module_tensors["down"] = value
            elif "up" in suffix:
                module_tensors["up"] = value
            elif "alpha" in suffix:
                module_tensors["alpha"] = value

        return lora_modules

    def apply_adapter_lora(self, adapter, lora_modules, adapter_weight):
        if adapter_weight == 0:
            return 0, []

        applied_count = 0
        failed_stems = []

        for stem, tensors in lora_modules.items():
            down = tensors.get("down")
            up = tensors.get("up")
            alpha = tensors.get("alpha")
            if down is None or up is None:
                continue

            module_name = self.map_stem_to_module_name(stem)
            try:
                target_module = adapter
                for part in module_name.split("."):
                    target_module = getattr(target_module, part)

                if not hasattr(target_module, "weight"):
                    failed_stems.append(f"{stem} -> {module_name} (no weight attribute)")
                    continue

                if down.dim() == 4:
                    down = down.squeeze(-1).squeeze(-1)
                if up.dim() == 4:
                    up = up.squeeze(-1).squeeze(-1)

                rank = down.shape[0]
                scale = alpha.item() / rank if alpha is not None else 1.0
                delta = torch.mm(up.to(torch.float32), down.to(torch.float32))
                delta = delta * scale * adapter_weight
                target_module.weight.data += delta.to(
                    dtype=target_module.weight.dtype,
                    device=target_module.weight.device,
                )
                applied_count += 1
            except AttributeError:
                failed_stems.append(f"{stem} -> {module_name} (module path not found)")
            except Exception as exc:
                failed_stems.append(f"{stem} -> {module_name} ({exc})")

        return applied_count, failed_stems

    def map_stem_to_module_name(self, stem):
        stem = stem.replace("text_model_", "").replace("llm_adapter_", "").replace("model_", "")

        if stem == "seq_projection_0":
            return "seq_projection.0"
        if stem == "seq_projection_4":
            return "seq_projection.4"
        if stem == "pooled_projection":
            return "pooled_projection"

        match = re.match(r"attention_blocks_(\d+)_attn_(q_proj|k_proj|v_proj|out_proj)", stem)
        if match:
            return f"attention_blocks.{match.group(1)}.attn.{match.group(2)}"

        match = re.match(r"attention_blocks_(\d+)_mlp_(\d+)", stem)
        if match:
            return f"attention_blocks.{match.group(1)}.mlp.{match.group(2)}"

        match = re.match(r"attention_pooler_attn_(q_proj|k_proj|v_proj|out_proj)", stem)
        if match:
            return f"attention_pooler.attn.{match.group(1)}"

        return stem


NODE_CLASS_MAPPINGS = {
    "JinaAdapterLoRALoader": JinaAdapterLoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JinaAdapterLoRALoader": "Jina Adapter LoRA Loader",
}