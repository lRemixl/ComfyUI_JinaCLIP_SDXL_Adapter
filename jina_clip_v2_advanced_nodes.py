import torch
import logging
import gc
import os
import re
import math
from safetensors.torch import load_file

from .utils import get_llm_adapters, get_llm_adapter_path
from .jina_to_sdxl_adapter_v2 import JinaToSDXLAdapterV2
from .jina_attention_mask import build_conditioning, install_cross_attention_mask_patch

logger = logging.getLogger("JinaCLIP-SDXL-Adapter")

install_cross_attention_mask_patch()

class JinaAdapterLoaderAdvanced:
    """ComfyUI node that loads the Jina-clip-v2 adapter."""

    def __init__(self):
        self.adapter = None
        self.current_adapter_path = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.current_attn_pooling = None
        self.current_use_positional = None
        self.current_max_seq_length = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter_name": (get_llm_adapters(), {
                    "default": get_llm_adapters()[0] if get_llm_adapters() else None
                }),
            },
            "optional": {
                "device": (["auto", "cuda:0", "cuda:1", "cpu"], {"default": "auto"}),
                "max_seq_length": (["539", "1078"], {"default": "539"}),
                "force_reload": ("BOOLEAN", {"default": False}),
                "attn_pooling": ("BOOLEAN", {"default": True}),
                "use_positional_embeddings": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("JINA_ADAPTER", "STRING")
    RETURN_NAMES = ("jina_adapter", "info")
    FUNCTION = "load_adapter"
    CATEGORY = "llm_sdxl/jina/advanced"

    def load_adapter(
        self,
        adapter_name,
        device="auto",
        max_seq_length="539",
        force_reload=False,
        attn_pooling=True,
        use_positional_embeddings=True,
    ):
        if device == "auto":
            device = self.device

        adapter_path = get_llm_adapter_path(adapter_name)
        max_seq_len_int = 1078 if max_seq_length == "1078" else 539

        try:
            should_reload = (
                force_reload
                or self.adapter is None
                or self.current_adapter_path != adapter_path
                or self.current_attn_pooling != attn_pooling
                or self.current_use_positional != use_positional_embeddings
                or self.current_max_seq_length != max_seq_len_int
            )

            if should_reload:
                if self.adapter is not None:
                    del self.adapter
                    gc.collect()
                    torch.cuda.empty_cache()

                logger.info(f"Loading JinaToSDXL adapter from {adapter_path}")
                self.adapter = JinaToSDXLAdapterV2(
                    llm_dim=1024,
                    sdxl_seq_dim=2048,
                    sdxl_pooled_dim=1280,
                    n_attention_blocks=4,
                    num_heads=16,
                    dropout=0,
                    max_seq_len=max_seq_len_int,
                    attn_pooling=attn_pooling,
                    use_positional=use_positional_embeddings,
                )

                if os.path.exists(adapter_path):
                    checkpoint = load_file(adapter_path)
                    self.adapter.load_state_dict(checkpoint, strict=True)
                    logger.info(f"Loaded adapter weights from: {adapter_path}")
                else:
                    logger.warning(f"Adapter file not found: {adapter_path}, using initialized weights")

                self.adapter.to(device)
                self.adapter.eval()
                self.current_adapter_path = adapter_path
                self.current_attn_pooling = attn_pooling
                self.current_use_positional = use_positional_embeddings
                self.current_max_seq_length = max_seq_len_int
                logger.info("Jina Adapter loaded successfully")

            info = (
                f"Adapter: {adapter_path}\n"
                f"Device: {device}\n"
                f"Attention pooling: {attn_pooling}\n"
                f"Max sequence length: {max_seq_len_int}\n"
                f"Positional embeddings: {use_positional_embeddings}"
            )
            return (self.adapter, info)

        except Exception as e:
            logger.error(f"Failed to load Jina adapter: {str(e)}")
            raise Exception(f"Adapter loading failed: {str(e)}")

class JinaTextEncoderAdvanced:
    """
    Encodes text using Jina-Clip-V2 and converts it to prompt + pooled embedding for SDXL
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "jina_model": ("JINA_MODEL",),
                "jina_adapter": ("JINA_ADAPTER",),
                "text": ("STRING", {"multiline": True, "default": "masterpiece, (best quality:1.2)"}),
                "weighting_mode": (["comfy", "A1111", "comfy++", "skip"], {"default": "comfy"}),
                "custom_dtype": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
                "Padding_Mode": (["none", "Nearest-77", "539", "1078"], {"default": "Nearest-77"}),
                "format_text": ("BOOLEAN", {"default": True}),
                "cross_attention_mask": ("BOOLEAN", {"default": True}),
                "unmask_sink_padding": ("BOOLEAN", {"default": False}),
                "max_seq_length_string": (["512", "1024"], {"default": "512"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("CONDITIONING", "info")
    FUNCTION = "encode"
    CATEGORY = "llm_sdxl/jina/advanced"

    def parse_weights(self, text):
        out_text =[]
        out_weights = []
        stack =[] 
        current_weight = 1.0
        
        i = 0
        while i < len(text):
            char = text[i]

            if char == "\\" and i + 1 < len(text):
                out_text.append(text[i+1])
                out_weights.append(current_weight if not stack else stack[-1][0])
                i += 2
                continue

            if char == "(":
                stack.append((current_weight, len(out_text)))
                i += 1
                continue

            if char == ")":
                if len(stack) > 0:
                    old_weight, start_idx = stack.pop()
                    new_weight = None
                    colon_index = -1

                    for k in range(len(out_text) - 1, start_idx - 1, -1):
                        c = out_text[k]
                        if c == ":":
                            weight_str = "".join(out_text[k+1:])
                            try:
                                new_weight = float(weight_str)
                                colon_index = k
                            except ValueError:
                                pass
                            break
                        if c not in "0123456789. ":
                            break

                    if new_weight is not None:
                        remove_count = len(out_text) - colon_index
                        for _ in range(remove_count):
                            out_text.pop()
                            out_weights.pop()

                        seg_len = len(out_text) - start_idx
                        for k in range(seg_len):
                            out_weights[start_idx + k] = out_weights[start_idx + k] * new_weight
                    else:
                        seg_len = len(out_text) - start_idx
                        for k in range(seg_len):
                            out_weights[start_idx + k] = out_weights[start_idx + k] * 1.1

                    current_weight = old_weight
                else:
                    out_text.append(char)
                    out_weights.append(current_weight)
                i += 1
                continue

            out_text.append(char)
            out_weights.append(current_weight if not stack else stack[-1][0])
            i += 1

        return "".join(out_text), out_weights

    def get_padding_target(self, seq_len, do_pad):
        if do_pad == "none":
            return seq_len, ""

        if do_pad == "539":
            target_len = 539
            info_str = f"Original token length: {seq_len}, adapter padded to {target_len} Tokens"
        elif do_pad == "1078":
            target_len = 1078
            info_str = f"Original token length: {seq_len}, adapter padded to {target_len} Tokens"
        else:
            target_len = math.ceil(seq_len / 77) * 77
            info_str = f"Original token length: {seq_len}, adapter padded to nearest multiple of 77: {target_len} Tokens"

        if target_len < seq_len:
            info_str += f" (padding target is shorter than token length; using {seq_len})"
            target_len = seq_len

        return target_len, info_str

    def resize_adapter_inputs(self, hidden_states, attention_mask, target_len, token_weights=None):
        seq_len = hidden_states.shape[1]

        if seq_len > target_len:
            hidden_states = hidden_states[:, :target_len, :]
            attention_mask = attention_mask[:, :target_len]
            if token_weights is not None:
                token_weights = token_weights[:target_len]
            return hidden_states, attention_mask, token_weights

        if seq_len < target_len:
            pad_len = target_len - seq_len
            hidden_pad = torch.zeros(
                hidden_states.shape[0],
                pad_len,
                hidden_states.shape[2],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            mask_pad = torch.zeros(
                attention_mask.shape[0],
                pad_len,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            hidden_states = torch.cat([hidden_states, hidden_pad], dim=1)
            attention_mask = torch.cat([attention_mask, mask_pad], dim=1)

            if token_weights is not None:
                weight_pad = torch.ones(pad_len, dtype=token_weights.dtype, device=token_weights.device)
                token_weights = torch.cat([token_weights, weight_pad])

        return hidden_states, attention_mask, token_weights

    def get_token_data(self, tokenizer, clean_text, char_weights, device, do_pad):
        inputs = tokenizer(
            clean_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_seq_length_int,
            return_offsets_mapping=True,
        )

        input_ids = inputs.input_ids[0].to(device)
        attention_mask = inputs.attention_mask[0].to(device)
        offset_mapping = inputs.offset_mapping[0]

        seq_len = input_ids.shape[0]
        token_weights = torch.ones(seq_len, dtype=torch.float32, device=device)

        for token_index in range(seq_len):
            start = int(offset_mapping[token_index][0].item())
            end = int(offset_mapping[token_index][1].item())
            if start == end:
                continue

            if start < len(char_weights) and end <= len(char_weights):
                segment_weights = char_weights[start:end]
                if segment_weights:
                    token_weights[token_index] = sum(segment_weights) / len(segment_weights)

        target_len, info_str = self.get_padding_target(seq_len, do_pad)
        return input_ids.unsqueeze(0), attention_mask.unsqueeze(0), token_weights, target_len, info_str

    def encode(
        self,
        jina_model,
        jina_adapter,
        text,
        weighting_mode,
        custom_dtype,
        Padding_Mode="none",
        format_text=False,
        cross_attention_mask=True,
        unmask_sink_padding=False,
        max_seq_length_string="512",
    ):
        self.max_seq_length_int = 1024 if max_seq_length_string == "1024" else 512
        original_text = text

        if format_text:
            pattern = r'(^|,|\n)\s*(@[^,\.\n]+)'
            extracted_tags = []

            def extract_tag(match):
                prefix = match.group(1)
                tag_content = re.sub(r'\s+', ' ', match.group(2).strip())
                extracted_tags.append(re.sub(r'^@\s*', '@ ', tag_content))
                return '\n' if prefix == '\n' else ''

            clean_string = re.sub(pattern, extract_tag, text)
            clean_string = re.sub(r'(,\s*){2,}', ', ', clean_string)
            clean_string = re.sub(r'(^|\n)[ \t,]+', r'\1', clean_string)
            clean_string = re.sub(r'[ \t,]+($|\n)', r'\1', clean_string)

            if extracted_tags:
                prefix_str = ", ".join(extracted_tags)
                text = f"{prefix_str}, {clean_string}" if clean_string else prefix_str

        if custom_dtype == "auto":
            custom_dtype = torch.float32
        elif custom_dtype == "fp16":
            custom_dtype = torch.float16
        elif custom_dtype == "bf16":
            custom_dtype = torch.bfloat16
        elif custom_dtype == "fp32":
            custom_dtype = torch.float32

        try:
            device = jina_model.device
            clean_text, char_weights = self.parse_weights(text)
            input_ids, attention_mask, token_weights, target_len, info_str = self.get_token_data(
                jina_model.tokenizer,
                clean_text,
                char_weights,
                device,
                do_pad=Padding_Mode,
            )

            def run_jina_states(ids, mask):
                with torch.no_grad():
                    jina_model.hidden_states_cache = None
                    if hasattr(jina_model.model, 'get_text_features'):
                        pooled = jina_model.model.get_text_features(input_ids=ids, attention_mask=mask)
                    else:
                        out = jina_model.model.text_model(input_ids=ids, attention_mask=mask)
                        if hasattr(out, 'text_embeds'):
                            pooled = out.text_embeds
                        elif hasattr(out, 'pooler_output'):
                            pooled = out.pooler_output
                        elif isinstance(out, tuple):
                            pooled = out[1] if len(out) > 1 else out[0]
                        else:
                            pooled = out

                    if not isinstance(pooled, torch.Tensor):
                        pooled = jina_model.mean_pooling(jina_model.hidden_states_cache, mask)

                    return jina_model.hidden_states_cache.clone().to(custom_dtype), pooled.clone().to(custom_dtype)

            with torch.no_grad():
                empty_prompt_embeds = None
                if weighting_mode == "comfy":
                    empty_inputs = jina_model.tokenizer("", return_tensors="pt")
                    e_ids = empty_inputs.input_ids.to(device)
                    e_mask = empty_inputs.attention_mask.to(device)
                    e_hidden, e_pooled = run_jina_states(e_ids, e_mask)
                    e_hidden, e_adapter_mask, _ = self.resize_adapter_inputs(e_hidden, e_mask, target_len)
                    empty_prompt_embeds, _ = jina_adapter(
                        e_hidden.to(torch.float32),
                        e_pooled.to(torch.float32),
                        e_adapter_mask.to(torch.float32),
                    )

                base_hidden, base_pooled = run_jina_states(input_ids, attention_mask)
                adapter_hidden, adapter_mask, token_weights = self.resize_adapter_inputs(
                    base_hidden,
                    attention_mask,
                    target_len,
                    token_weights,
                )
                prompt_embeds, final_pooled = jina_adapter(
                    adapter_hidden.to(torch.float32),
                    base_pooled.to(torch.float32),
                    adapter_mask.to(torch.float32),
                )

                w_tensor = token_weights.unsqueeze(0).unsqueeze(-1).to(prompt_embeds.device)

                if weighting_mode == "A1111":
                    prompt_embeds = prompt_embeds * w_tensor
                elif weighting_mode == "comfy":
                    diff = prompt_embeds - empty_prompt_embeds
                    prompt_embeds = empty_prompt_embeds + (diff * w_tensor)
                elif weighting_mode == "comfy++":
                    w_list = token_weights.tolist()
                    spans = []

                    if w_list:
                        current_start = 0
                        current_w = w_list[0]
                        for index in range(1, len(w_list)):
                            if abs(w_list[index] - current_w) > 1e-5:
                                if abs(current_w - 1.0) > 1e-5:
                                    spans.append((current_start, index, current_w))
                                current_start = index
                                current_w = w_list[index]

                        if abs(current_w - 1.0) > 1e-5:
                            spans.append((current_start, len(w_list), current_w))

                    base_seq_len = attention_mask.shape[1]
                    for start, end, weight in spans:
                        mask_start = min(start, base_seq_len)
                        mask_end = min(end, base_seq_len)
                        if mask_start >= mask_end:
                            continue

                        span_mask = attention_mask.clone()
                        span_mask[0, mask_start:mask_end] = 0

                        masked_hidden, masked_pooled = run_jina_states(input_ids, span_mask)
                        masked_hidden, masked_adapter_mask, _ = self.resize_adapter_inputs(
                            masked_hidden,
                            span_mask,
                            target_len,
                        )
                        masked_embeds, _ = jina_adapter(
                            masked_hidden.to(torch.float32),
                            masked_pooled.to(torch.float32),
                            masked_adapter_mask.to(torch.float32),
                        )

                        full_slice = prompt_embeds[:, start:end, :]
                        masked_slice = masked_embeds[:, start:end, :]
                        prompt_embeds[:, start:end, :] = masked_slice + ((full_slice - masked_slice) * weight)
                elif weighting_mode == "skip":
                    pass

            prompt_embeds = prompt_embeds.to(custom_dtype).cpu().contiguous()
            final_pooled = final_pooled.to(custom_dtype).cpu().contiguous()

            unet_attention_mask = adapter_mask
            sink_applied = False
            if cross_attention_mask and unmask_sink_padding:
                unet_attention_mask = adapter_mask.clone()
                for batch_index in range(unet_attention_mask.shape[0]):
                    if unet_attention_mask[batch_index, -1] == 0:
                        unet_attention_mask[batch_index, -1] = 1
                        sink_applied = True

            conditioning, mask_info = build_conditioning(
                prompt_embeds,
                final_pooled,
                unet_attention_mask,
                cross_attention_mask,
            )

            info_str = info_str + f"\nJina model sequence length: {input_ids.shape[1]}"
            info_str = info_str + f"\nprompt_embeds shape: {prompt_embeds.shape}{mask_info}"
            if sink_applied:
                info_str += "\n[Attention Sink]: Unmasked final padding token for SDXL UNet."
            if original_text != text:
                info_str = info_str + f"\nclean text (input to adapter) is: {text}"
            info_str = info_str + f"\nmax_seq_length_int: {self.max_seq_length_int}"
            return (conditioning, info_str)

        except Exception as e:
            logger.error(f"Jina Encoding error: {e}")
            raise e

# ComfyUI Registration
NODE_CLASS_MAPPINGS = {
    "JinaAdapterLoaderAdvanced": JinaAdapterLoaderAdvanced,
    "JinaTextEncoderAdvanced": JinaTextEncoderAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JinaAdapterLoaderAdvanced": "Jina Adapter Loader (Advanced)",
    "JinaTextEncoderAdvanced": "Jina Text Encode (SDXL) (Advanced)",
}
