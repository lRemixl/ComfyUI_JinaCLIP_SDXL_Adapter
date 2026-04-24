# ComfyUI_JinaCLIP_SDXL_Adapter
ComfyUI nodes for use with [Jina-clip-v2 adapter](https://huggingface.co/TheRemixer/jina-clip-v2-adapter/) 

**Node:** llm_sdxl/jina/Jina CLIP v2 Loader

**Purpose:** Load Jina-clip-v2.

**Node:** llm_sdxl/jina/Jina Adapter Loader

**Purpose:** Load the Jina-clip-v2 adapter.  

**Node:** llm_sdxl/jina/Jina Text Encode (SDXL)

**Purpose:** Text encode prompt for Jina-clip-v2 + adapter, returns SDXL conditioning.

**Advanced nodes:** 
  All the nodes under llm_sdxl/jina/advanced are there for more options. If you're getting black images when generating images. Change sage attention to sage attention triton, or use the Jina Text Encode (SDXL) node under llm_sdxl/jina/advanced and set it to Nearest-77 (<-- different Nearest-77 then the non-advanced version), for both the positive and negative prompt. 

Credits:
NeuroSenko: Uses their [custom nodes](https://github.com/NeuroSenko/ComfyUI_LLM_SDXL_Adapter/) as a base 
