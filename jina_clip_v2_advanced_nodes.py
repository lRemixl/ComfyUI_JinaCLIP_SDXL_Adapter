import torch
import logging
import gc
import os
import sys
import types
import re
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file

from .utils import get_llm_checkpoints, get_llm_checkpoint_path, get_llm_adapters, get_llm_adapter_path
from .jina_to_sdxl_adapter_v2 import JinaToSDXLAdapterV2

logger = logging.getLogger("JinaCLIP-SDXL-Adapter")

class JinaAdapterLoaderAdvanced:
    """ComfyUI node that loads the Jina-clip-v2 adapter."""

    def __init__(self):
        self.adapter = None
        self.current_adapter_path = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.current_attn_pooling = None
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
                "force_reload": ("BOOLEAN", {"default": False}),
                "attn_pooling": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("JINA_ADAPTER", "STRING")
    RETURN_NAMES = ("jina_adapter", "info")
    FUNCTION = "load_adapter"
    CATEGORY = "llm_sdxl/jina/advanced"

    def load_adapter(self, adapter_name, device="auto", force_reload=False, attn_pooling=True):
        if device == "auto":
            device = self.device
        
        adapter_path = get_llm_adapter_path(adapter_name)

        try:
            if force_reload or self.adapter is None or self.current_adapter_path != adapter_path or self.current_attn_pooling != attn_pooling:
                if self.adapter is not None:
                    del self.adapter
                    gc.collect()
                    torch.cuda.empty_cache()

                logger.info(f"Loading JinaToSDXL adapter from {adapter_path}")
                if attn_pooling == False:
                    self.adapter = JinaToSDXLAdapterV2(
                        llm_dim=1024,           
                        sdxl_seq_dim=2048,
                        sdxl_pooled_dim=1280,
                        n_attention_blocks=4,
                        num_heads=16,
                        dropout=0,
                        max_seq_len=539, # Nearest multiple of 77 to 512. Found out that the adapter doesn't have to output embeddings with a sequence length of a multiple of 77. I had sage-attention cuda enabled which was causing black images when it wasn't.
                        attn_pooling=False) 
                else:
                    # Use attn_pooling
                    self.adapter = JinaToSDXLAdapterV2(
                        llm_dim=1024,           
                        sdxl_seq_dim=2048,
                        sdxl_pooled_dim=1280,
                        n_attention_blocks=4,
                        num_heads=16,
                        dropout=0,
                        max_seq_len=539, # Nearest multiple of 77 to 512. Found out that the adapter doesn't have to output embeddings with a sequence length of a multiple of 77. I had sage-attention cuda enabled which was causing black images when it wasn't.
                        attn_pooling=True)

                if os.path.exists(adapter_path):
                    checkpoint = load_file(adapter_path)
                    self.adapter.load_state_dict(checkpoint, strict=True)
                    logger.info(f"Loaded adapter weights from: {adapter_path}")
                else:
                    logger.warning(f"Adapter file not found: {adapter_path}, using initialized weights")

                self.adapter.to(device)
                self.adapter.eval()
                self.current_attn_pooling = attn_pooling
                self.current_adapter_path = adapter_path
                logger.info("Jina Adapter loaded successfully")

            info = f"Adapter: {adapter_path}\nDevice: {device}"
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
                "Padding_Mode": (["none", "Nearest-77", "539"], {"default": "Nearest-77"}),
                "format_text": ("BOOLEAN", {"default": True}),
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

    def get_token_data(self, tokenizer, clean_text, char_weights, device, do_pad):
        import math

        inputs = tokenizer(
            clean_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )

        input_ids = inputs.input_ids[0].to(device)
        attention_mask = inputs.attention_mask[0].to(device)
        offset_mapping = inputs.offset_mapping[0]

        seq_len = input_ids.shape[0]
        token_weights = torch.ones(seq_len, dtype=torch.float32, device=device)

        for t_idx in range(seq_len):
            start, end = int(offset_mapping[t_idx][0].item()), int(offset_mapping[t_idx][1].item())
            if start == end: continue # for special tokens 

            if start < len(char_weights) and end <= len(char_weights):
                segment_weights = char_weights[start:end]
                if segment_weights:
                    token_weights[t_idx] = sum(segment_weights) / len(segment_weights)
        info_str = ""                

        if do_pad != "none":
            
            # Pad sequence to nearest multiple of 77 to prevent black images
            # (Sage attention cuda was causing the black images when not using multiple of 77. Not actually needed, hence "none" mode)
            # However does provide better performance still. So still enabled by default.

            target_len = math.ceil(seq_len / 77) * 77

            info_str = f"Original token length: {seq_len}, Padded to nearest mutliple of 77: {target_len} Tokens"

            if do_pad == "539":
                target_len = 539
                info_str = f"Original token length: {seq_len}, Padded to {target_len} Tokens"
            pad_len = target_len - seq_len

            if pad_len > 0:
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = tokenizer.eos_token_id or 0

                pad_ids = torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype, device=device)
                pad_mask = torch.zeros((pad_len,), dtype=attention_mask.dtype, device=device)
                pad_weights = torch.ones((pad_len,), dtype=token_weights.dtype, device=device)

                input_ids = torch.cat([input_ids, pad_ids])
                attention_mask = torch.cat([attention_mask, pad_mask])
                token_weights = torch.cat([token_weights, pad_weights])

        return input_ids.unsqueeze(0), attention_mask.unsqueeze(0), token_weights, info_str

    def encode(self, jina_model, jina_adapter, text, weighting_mode, custom_dtype, Padding_Mode="none",format_text=False):
        
        original_text = text

        if format_text:
            # Match @ tags explicitly occurring at start of string, after comma, or after newline 
            # (ignoring any leading spaces for the tag itself)
            pattern = r'(^|,|\n)\s*(@[^,\.\n]+)'
            extracted_tags = []
            
            def extract_tag(match):
                prefix = match.group(1)
                tag_content = match.group(2)
                
                # Strip excessive whitespace inside the tags and ensure exactly 1 space right after @
                tag_content = re.sub(r'\s+', ' ', tag_content.strip())
                formatted_tag = re.sub(r'^@\s*', '@ ', tag_content)
                extracted_tags.append(formatted_tag)
                
                # Keep newlines to prevent structural collapse, but mark commas/starts for removal
                return '\n' if prefix == '\n' else ''
            
            clean_string = text
            clean_string = re.sub(pattern, extract_tag, clean_string)
            
            # --- Cleanup lingering formatting fragments ---
            # Collapse duplicate commas left behind
            clean_string = re.sub(r'(,\s*){2,}', ', ', clean_string)
            # Remove isolated commas/spaces occurring at the start or immediately after a newline
            clean_string = re.sub(r'(^|\n)[ \t,]+', r'\1', clean_string)
            # Remove isolated commas/spaces occurring at the end or immediately before a newline
            clean_string = re.sub(r'[ \t,]+($|\n)', r'\1', clean_string)
            
            # Prepend matching queries
            if extracted_tags:
                prefix_str = ", ".join(extracted_tags)
                if clean_string:
                    text = prefix_str + ", " + clean_string
                else:
                    text = prefix_str


        if custom_dtype == "auto":
            custom_dtype = torch.float32
        elif custom_dtype == "fp16":
            custom_dtype=torch.float16
        elif custom_dtype == "bf16":
            custom_dtype=torch.bfloat16
        elif custom_dtype == "fp32":
            custom_dtype=torch.float32
        
        do_pad_77 = Padding_Mode

        try:
            device = jina_model.device
            clean_text, char_weights = self.parse_weights(text)

            input_ids, attention_mask, token_weights, info_str = self.get_token_data(
                jina_model.tokenizer, clean_text, char_weights, device, do_pad=do_pad_77
            )

            # Helper for Jina-clip-v2's forward pass
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

                    # use mean-pooling fallback check
                    if not isinstance(pooled, torch.Tensor):
                        pooled = jina_model.mean_pooling(jina_model.hidden_states_cache, mask)

                    # prevent Jina's hook from overwriting the references across runs
                    return jina_model.hidden_states_cache.clone().to(custom_dtype), pooled.clone().to(custom_dtype)

            with torch.no_grad():
                empty_prompt_embeds = None
                if weighting_mode == "comfy":
                    # Pre-calculate baseline
                    empty_inputs = jina_model.tokenizer("", return_tensors="pt")
                    e_ids = empty_inputs.input_ids.to(device)
                    e_mask = empty_inputs.attention_mask.to(device)

                    seq_len = input_ids.shape[1]
                    pad_len = seq_len - e_ids.shape[1]

                    # align empty sequence length to the prompt sequence length
                    if pad_len > 0:
                        pad_token_id = jina_model.tokenizer.pad_token_id or 0
                        pad_ids = torch.full((1, pad_len), pad_token_id, dtype=e_ids.dtype, device=device)
                        pad_mask = torch.zeros((1, pad_len), dtype=e_mask.dtype, device=device)
                        e_ids = torch.cat([e_ids, pad_ids], dim=1)
                        e_mask = torch.cat([e_mask, pad_mask], dim=1)
                    elif pad_len < 0:
                        e_ids = e_ids[:, :seq_len]
                        e_mask = e_mask[:, :seq_len]

                    # Process empty baseline string through the text encoder + adapter 
                    e_hidden, e_pooled = run_jina_states(e_ids, e_mask)
                    empty_prompt_embeds, _ = jina_adapter(
                        e_hidden.to(torch.float32), 
                        e_pooled.to(torch.float32), 
                        e_mask.to(torch.float32)
                    )

                # Process prompt through the text encoder + adapter 
                base_hidden, base_pooled = run_jina_states(input_ids, attention_mask)
                prompt_embeds, final_pooled = jina_adapter(
                    base_hidden.to(torch.float32), 
                    base_pooled.to(torch.float32), 
                    attention_mask.to(torch.float32)
                )

                # mapped to adapter output sequence length
                w_tensor = token_weights.unsqueeze(0).unsqueeze(-1).to(prompt_embeds.device)

                # Apply weighting to the adapter's output
                if weighting_mode == "A1111":
                    prompt_embeds = prompt_embeds * w_tensor

                elif weighting_mode == "comfy":
                    diff = prompt_embeds - empty_prompt_embeds
                    prompt_embeds = empty_prompt_embeds + (diff * w_tensor)

                elif weighting_mode == "comfy++":
                    w_list = token_weights.tolist()
                    spans =[] 

                    if len(w_list) > 0:
                        current_start = 0
                        current_w = w_list[0]
                        for i in range(1, len(w_list)):
                            if abs(w_list[i] - current_w) > 1e-5:
                                if abs(current_w - 1.0) > 1e-5:
                                    spans.append((current_start, i, current_w))
                                current_start = i
                                current_w = w_list[i]

                        if abs(current_w - 1.0) > 1e-5:
                            spans.append((current_start, len(w_list), current_w))

                    for start, end, weight in spans:
                        span_mask = attention_mask.clone()
                        span_mask[0, start:end] = 0

                        masked_hidden, masked_pooled = run_jina_states(input_ids, span_mask)

                        # Process masked hidden states through adapter
                        masked_embeds, _ = jina_adapter(
                            masked_hidden.to(torch.float32),
                            masked_pooled.to(torch.float32),
                            span_mask.to(torch.float32)
                        )

                        full_slice = prompt_embeds[:, start:end, :]
                        masked_slice = masked_embeds[:, start:end, :]

                        # Interpolate final embeddings
                        diff_slice = full_slice - masked_slice
                        prompt_embeds[:, start:end, :] = masked_slice + (diff_slice * weight)

                elif weighting_mode == "skip":
                    pass

            prompt_embeds = prompt_embeds.to(custom_dtype).cpu().contiguous()
            final_pooled = final_pooled.to(custom_dtype).cpu().contiguous()

            conditioning = [[prompt_embeds, {"pooled_output": final_pooled}]]

            info_str = info_str + f"\nprompt_embeds shape: {prompt_embeds.shape}"
            if original_text != text:
                info_str = info_str + f"\nclean text (input to adapter) is: {text}"
            return (conditioning, info_str)

        except Exception as e:
            logger.error(f"Jina Encoding error: {e}")
            raise e

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