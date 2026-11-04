import asyncio
from dataclasses import dataclass
from loguru import logger
from awm.tools import check_mcp_server


@dataclass
class Config:
    # MCP server URL to check
    url: str = "http://localhost:8001/mcp"
    # connection timeout in seconds
    timeout: float = 10.0


def run(config: Config):
    running, tools_count, tools, error = asyncio.run(
        check_mcp_server(url=config.url, timeout=config.timeout)
    )

    if running and tools_count > 0:
        for tool in tools:
            logger.info(f"  - {tool['name']}: {tool.get('description', '')}")
        
        logger.success(f"Server is running at {config.url}")
        logger.info(f"Tools count: {tools_count}")
    else:
        logger.error(f"Server check failed at {config.url}: {error}")
        exit(1)
