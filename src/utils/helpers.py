"""
Utility functions: config loading, logging setup, common helpers.
"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_project_root() -> Path:
    return PROJECT_ROOT


def load_config() -> dict:
    """Load strategy configuration from YAML."""
    config_path = PROJECT_ROOT / "config" / "strategy.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging():
    """Configure loguru for the project."""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "trading_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
    )
    return logger


def get_env(key: str, default: str = None) -> str:
    """Get environment variable with optional default."""
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} is not set. Check your .env file.")
    return value
