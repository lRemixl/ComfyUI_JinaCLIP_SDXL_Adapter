import torch
import logging
import gc
import os
import sys
import re
import math
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file

from .utils import get_llm_checkpoints, get_llm_checkpoint_path, get_llm_adapters, get_llm_adapter_path
from .jina_to_sdxl_adapter_v2 import JinaToSDXLAdapterV2
from .jina_attention_mask import build_conditioning, install_cross_attention_mask_patch

logger = logging.getLogger("JinaCLIP-SDXL-Adapter")

install_cross_attention_mask_patch()

def ensure_flash_attn_bypass():
    """
    Kept getting this crash "TypeError: 'function' object is not iterable"#
    when Jina tries to import the rotary submodule
    Most likely from a mock version of flash-attn
    """
    try:
        from flash_attn.ops.triton.rotary import apply_rotary
        return  # Return if flash_attn works
    except Exception:
        pass
    logger.warning("Removing Flash-attention, can't import apply_rotary. Jina-clip-v2 will FAIL crash ComfyUI if this is not done.")
    # Delete flash-attn from sys.modules
    keys_to_delete = [k for k in sys.modules if k.startswith("flash_attn")]
    for k in keys_to_delete:
        del sys.modules[k]
    return

class JinaStates:
    """
    Get hidden states from jina-clip-v2 using hook
    """
    def __init__(self,
                 model_id: str,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.bfloat16,
                 max_length: int = 1024):

        self.max_length = max_length
        self.device = device
        self.dtype = dtype

        logger.info(f"Loading tokenizer from {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=False,
            fix_mistral_regex=True,
        )

        logger.info(f"Loading model from {model_id}")
        ensure_flash_attn_bypass()
        self.model = AutoModel.from_pretrained(
            model_id,
            low_cpu_mem_usage=False,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=False,
        )
        self.model.to(device)

        # Unload the vision tower to save VRAM
        if hasattr(self.model, "vision_model"):
            del self.model.vision_model
            torch.cuda.empty_cache()
            logger.info("Vision model unloaded: ")

        self.model.eval()

        # attach forward hook
        self.hidden_states_cache = None
        self.encoder_module = None

        for name, module in self.model.named_modules():
            if 'vision' in name.lower():
                continue


            has_layer = hasattr(module, 'layer') and isinstance(getattr(module, 'layer'), torch.nn.ModuleList)
            has_layers = hasattr(module, 'layers') and isinstance(getattr(module, 'layers'), torch.nn.ModuleList)
            has_block = hasattr(module, 'block') and isinstance(getattr(module, 'block'), torch.nn.ModuleList)
            has_blocks = hasattr(module, 'blocks') and isinstance(getattr(module, 'blocks'), torch.nn.ModuleList)

            if has_layer or has_layers or has_block or has_blocks:
                layer_list = (getattr(module, 'layer', None) or getattr(module, 'layers', None) or 
                              getattr(module, 'block', None) or getattr(module, 'blocks', None))
                if layer_list is not None and len(layer_list) > 1:
                    self.encoder_module = module
                    break

        if self.encoder_module is None:
            raise RuntimeError("Could not find the text encoder module to attach a hook. Check model")

        def forward_hook(module, args, output):
            if hasattr(output, 'last_hidden_state'):
                self.hidden_states_cache = output.last_hidden_state
            elif isinstance(output, tuple):
                self.hidden_states_cache = output[0]
            else:
                self.hidden_states_cache = output

        self.encoder_module.register_forward_hook(forward_hook)
        logger.info(f"Successfully attached hook to: {self.encoder_module.__class__.__name__}")

    def mean_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """fallback mean pooling, code from Jina-embeddings-v3. Also used in previous adapter version."""
        hidden_states_f32 = hidden_states.to(torch.float32)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states_f32.size()).float()

        sum_embeddings = torch.sum(hidden_states_f32 * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        pooled = sum_embeddings / sum_mask
        return pooled.to(self.dtype)

class JinaClipLoader:
    """ComfyUI node that loads the Jina-clip-v2 model + tokenizer."""

    def __init__(self):
        self.model = None
        self.current_model_path = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_llm_checkpoints(), {
                    "default": get_llm_checkpoints()[0] if get_llm_checkpoints() else None
                }),
            },
            "optional": {
                "device": (["auto", "cuda:0", "cuda:1", "cpu"], {"default": "auto"}),
                "force_reload": ("BOOLEAN", {"default": False}),
                "dtype": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("JINA_MODEL", "STRING")
    RETURN_NAMES = ("jina_model", "info")
    FUNCTION = "load_model"
    CATEGORY = "llm_sdxl/jina"

    def load_model(self, model_name, device="auto", force_reload=False, dtype="auto"):
        weight_dtype = torch.bfloat16
        if dtype == "fp16":
            weight_dtype = torch.float16
        elif dtype == "fp32":
            weight_dtype = torch.float32

        if device == "auto":
            device = self.device
                
        try:
            model_path = get_llm_checkpoint_path(model_name)

            if force_reload or self.model is None or self.current_model_path != model_path:
                if self.model is not None:
                    del self.model
                    gc.collect()
                    torch.cuda.empty_cache()
                
                self.model = JinaStates(
                    model_id=model_path,
                    device=device,
                    dtype=weight_dtype,
                    max_length=1024
                )

                self.current_model_path = model_path
                logger.info("Jina CLIP Model loaded successfully")

            info = f"Model: {model_path}\nDevice: {device}\nLoaded: {self.model is not None}"
            return (self.model, info)

        except Exception as e:
            logger.error(f"Failed to load Jina CLIP Model: {str(e)}")
            raise Exception(f"Model loading failed: {str(e)}")

class JinaAdapterLoader:
    """ComfyUI node that loads the Jina-clip-v2 adapter."""

    def __init__(self):
        self.adapter = None
        self.current_adapter_path = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.current_attn_pooling = None
        self.current_use_positional = None
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
            }
        }
    
    RETURN_TYPES = ("JINA_ADAPTER", "STRING")
    RETURN_NAMES = ("jina_adapter", "info")
    FUNCTION = "load_adapter"
    CATEGORY = "llm_sdxl/jina"
    
    def load_adapter(self, adapter_name, device="auto", force_reload=False):
        use_positional_embeddings=False
        if device == "auto":
            device = self.device
        adapter_path = get_llm_adapter_path(adapter_name)

        try:
            if force_reload or self.adapter is None or self.current_adapter_path != adapter_path or self.current_use_positional != use_positional_embeddings:
                if self.adapter is not None:
                    del self.adapter
                    gc.collect()
                    torch.cuda.empty_cache()

                logger.info(f"Loading JinaToSDXL adapter from {adapter_path}")
                # Use attn_pooling
                self.adapter = JinaToSDXLAdapterV2(
                    llm_dim=1024,           
                    sdxl_seq_dim=2048,
                    sdxl_pooled_dim=1280,
                    n_attention_blocks=4,
                    num_heads=16,
                    dropout=0,
                    max_seq_len=1078,
                    attn_pooling=True,
                    use_positional=use_positional_embeddings)

                if os.path.exists(adapter_path):
                    checkpoint = load_file(adapter_path)
                    self.adapter.load_state_dict(checkpoint, strict=True)
                    logger.info(f"Loaded adapter weights from: {adapter_path}")
                else:
                    logger.warning(f"Adapter file not found: {adapter_path}? Check model path")

                self.adapter.to(device)
                self.adapter.eval()
                self.current_adapter_path = adapter_path
                self.current_attn_pooling = True
                self.current_use_positional = use_positional_embeddings
                logger.info("Jina Adapter loaded successfully")

            info = f"Adapter: {adapter_path}\nDevice: {device}\nPositional embeddings: {use_positional_embeddings}"
            return (self.adapter, info)

        except Exception as e:
            logger.error(f"Failed to load Jina adapter: {str(e)}")
            raise Exception(f"Adapter loading failed: {str(e)}")

class JinaTextEncoder:
    """
    Encodes text using Jina-Clip-V2 + adapter and outputs prompt + pooled embedding for SDXL
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "jina_model": ("JINA_MODEL",),
                "jina_adapter": ("JINA_ADAPTER",),
                "text": ("STRING", {"multiline": True, "default": "masterpiece, (best quality:1.2)"}),
                "Padding_Mode": (["none", "Nearest-77", "539", "1078"], {"default": "Nearest-77"}),
                "cross_attention_mask": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("CONDITIONING", "info")
    FUNCTION = "encode"
    CATEGORY = "llm_sdxl/jina"

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

    def get_token_data(self, tokenizer, clean_text, char_weights, device, do_pad, max_length):
        inputs = tokenizer(
            clean_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_length,
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

    def encode(self, jina_model, jina_adapter, text, Padding_Mode="Nearest-77", cross_attention_mask=True):
        weighting_mode = "comfy"
        custom_dtype = torch.float32
        format_text = False

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

        try:
            device = jina_model.device
            clean_text, char_weights = self.parse_weights(text)
            max_length = getattr(jina_model, "max_length", 1024)
            input_ids, attention_mask, token_weights, target_len, info_str = self.get_token_data(
                jina_model.tokenizer,
                clean_text,
                char_weights,
                device,
                do_pad=Padding_Mode,
                max_length=max_length,
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
                if len(char_weights) > 0:
                    diff = prompt_embeds - empty_prompt_embeds
                    prompt_embeds = empty_prompt_embeds + (diff * w_tensor)

            prompt_embeds = prompt_embeds.to(custom_dtype).cpu().contiguous()
            final_pooled = final_pooled.to(custom_dtype).cpu().contiguous()

            conditioning, mask_info = build_conditioning(
                prompt_embeds,
                final_pooled,
                adapter_mask,
                cross_attention_mask,
            )
            info_str = info_str + f"\nJina model sequence length: {input_ids.shape[1]}"
            info_str = info_str + f"\nprompt_embeds shape: {prompt_embeds.shape}{mask_info}"

            return (conditioning, info_str)

        except Exception as e:
            logger.error(f"Jina Encoding error: {e}")
            raise e

# ComfyUI Registration
NODE_CLASS_MAPPINGS = {
    "JinaClipLoader": JinaClipLoader,
    "JinaAdapterLoader": JinaAdapterLoader,
    "JinaTextEncoder": JinaTextEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JinaClipLoader": "Jina CLIP v2 Loader",
    "JinaAdapterLoader": "Jina Adapter Loader",
    "JinaTextEncoder": "Jina Text Encode (SDXL)",
}
