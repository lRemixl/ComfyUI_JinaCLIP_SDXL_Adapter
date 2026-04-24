import logging
import sys
import os
from typing import Dict, Tuple
# Copied and modified from NeuroSenko/ComfyUI_LLM_SDXL_Adapter
# Setup logging
logger = logging.getLogger("JinaCLIP-SDXL-Adapter")
logger.setLevel(logging.WARN)

# Add custom formatter with module prefix
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[JinaCLIP-SDXL-Adapter] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False  # Prevent duplicate logs from parent loggers

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Check dependencies
try:
    import torch
    import transformers
    import safetensors
    import einops
    logger.info("All required dependencies found")
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    logger
# Import all node modules from separate files
try:
    from .jina_clip_v2_nodes import NODE_CLASS_MAPPINGS as JINA_CLIP_V2_MAPPINGS
    from .jina_clip_v2_nodes import NODE_DISPLAY_NAME_MAPPINGS as JINA_CLIP_V2_DISPLAY_MAPPINGS

    from .jina_clip_v2_advanced_nodes import NODE_CLASS_MAPPINGS as JINA_CLIP_V2_ADVANCED_MAPPINGS
    from .jina_clip_v2_advanced_nodes import NODE_DISPLAY_NAME_MAPPINGS as JINA_CLIP_V2_ADVANCED_DISPLAY_MAPPINGS

    logger.info("Successfully imported all node modules from separate files")
    
except Exception as e:
    logger.error(f"Failed to import node modules: {e}")
    raise

# Combine all node mappings
NODE_CLASS_MAPPINGS: Dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

# Add all mappings from separate files
all_class_mappings = [
    JINA_CLIP_V2_MAPPINGS,
    JINA_CLIP_V2_ADVANCED_MAPPINGS,
]
all_display_mappings = [
    JINA_CLIP_V2_DISPLAY_MAPPINGS,
    JINA_CLIP_V2_ADVANCED_DISPLAY_MAPPINGS,
]

for mapping in all_class_mappings:
    NODE_CLASS_MAPPINGS.update(mapping)

for mapping in all_display_mappings:
    NODE_DISPLAY_NAME_MAPPINGS.update(mapping)

# Version and metadata
__version__ = "1.0.0"
__author__ = "Remix"
__description__ = " ComfyUI nodes for Jina-Clip-v2 to SDXL adapter workflow"

# Export information for ComfyUI
WEB_DIRECTORY = "./web"  # For any web UI components (if needed)

# Log successful initialization
logger.info(f"Jina-clip-v2 SDXL Adapter v{__version__} initialized successfully")
logger.info(f"Registered {len(NODE_CLASS_MAPPINGS)} nodes from separate files:")
for node_name in sorted(NODE_CLASS_MAPPINGS.keys()):
    display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_name, node_name)
    logger.info(f"  - {node_name} ({display_name})")

# Custom type definitions for ComfyUI
CUSTOM_TYPES = {
    "JINA_MODEL": "JinaStates instance (holding model+tokenizer)",
    "JINA_ADAPTER": "Jina Adapter model instance",
}

def get_node_info() -> Dict[str, any]:
    """
    Return information about available nodes for debugging/documentation
    """
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "nodes": {
            name: {
                "display_name": NODE_DISPLAY_NAME_MAPPINGS.get(name, name),
                "class": cls.__name__,
                "category": getattr(cls, "CATEGORY", "unknown") if hasattr(cls, "CATEGORY") else "unknown",
                "function": getattr(cls, "FUNCTION", "unknown") if hasattr(cls, "FUNCTION") else "unknown"
            }
            for name, cls in NODE_CLASS_MAPPINGS.items()
        },
        "custom_types": CUSTOM_TYPES
    }

# Optional: Setup hook for ComfyUI initialization
def setup_js():
    """
    Setup any JavaScript/web components if needed
    """
    pass

# Export what ComfyUI expects
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS", 
    "WEB_DIRECTORY",
    "get_node_info"
]
# Above from NeuroSenko/ComfyUI_LLM_SDXL_Adapter