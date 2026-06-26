# ComfyUI_JinaCLIP_SDXL_Adapter
ComfyUI nodes for use with [Jina-clip-v2 adapter](https://huggingface.co/TheRemixer/jina-clip-v2-adapter/) 

### **Jina CLIP v2 Loader**
* **Path:** `llm_sdxl/jina/Jina CLIP v2 Loader`
* **Description:** Loads the base Jina-clip-v2 model.

### **Jina Adapter Loader**
* **Path:** `llm_sdxl/jina/Jina Adapter Loader`
* **Description:** Loads the Jina-clip-v2 adapter.

### **Jina Text Encode (SDXL)**
* **Path:** `llm_sdxl/jina/Jina Text Encode (SDXL)`
* **Description:** Text encode prompt for Jina-clip-v2 + adapter, returns SDXL conditioning.

**Advanced nodes**:
- Advanced nodes included for testing.
- Initial release on the adapter needs to be loaded with the advanced node. With settings:
  - Positional embeddings: true
  - max_seq_length: 539
  - attn_pooling: true 
Credits:
NeuroSenko: Uses their [custom nodes](https://github.com/NeuroSenko/ComfyUI_LLM_SDXL_Adapter/) as a base 
****
