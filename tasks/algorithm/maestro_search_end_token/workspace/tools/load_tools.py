from dataclasses import dataclass
from typing import List

from loguru import logger

# Try to import config_loader, with fallback
try:
    from tools.config_loader import load_config
except ImportError:
    try:
        import sys
        import os
        # Try to find config_loader in parent directories
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from tools.config_loader import load_config
    except ImportError:
        # Fallback: create a simple load_config function
        def load_config(path):
            import yaml
            try:
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception:
                return {"tools": []}
import json, hashlib
from typing import Optional

@dataclass
class Tool:
    tool_name: str
    tool_config: dict
    


def load_tools(config: dict) -> List[Tool]:
        
    tools = []
    for tool_config in config["tools"]:
        tools.append(Tool(tool_name=tool_config['name'], tool_config=tool_config))
    tool_names = [t.tool_name for t in tools]
    logger.info(
        f"Load {len(tools)} tools: {tool_names}"
    )
           
    return tools
# test
