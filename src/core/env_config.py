"""
Environment configuration management.

Loads environment variables from .env file and provides
secure access to API keys and configuration.
"""

import os
from pathlib import Path
from typing import Optional

from core.utils import get_logger

logger = get_logger(__name__)


def load_env_file(env_path: Optional[str | Path] = None) -> None:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Optional path to .env file. If None, looks for .env in project root.
    """
    try:
        from dotenv import load_dotenv
        
        if env_path is None:
            # Look for .env in project root
            project_root = Path(__file__).parent.parent.parent
            env_path = project_root / ".env"
        else:
            env_path = Path(env_path)
        
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from {env_path}")
        else:
            logger.warning(f".env file not found at {env_path}")
            logger.warning("Create .env file from .env.example template")
            
    except ImportError:
        logger.warning("python-dotenv not installed. Install with: pip install python-dotenv")


def get_roboflow_config() -> dict:
    """
    Get Roboflow API configuration from environment.
    
    Returns:
        Dictionary with Roboflow configuration.
        
    Raises:
        ValueError: If required environment variables are missing.
    """
    api_key = os.getenv("ROBOFLOW_API_KEY")
    
    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY not found in environment. "
            "Set it in .env file or as environment variable."
        )
    
    # Get workspace - if not set, use None (Roboflow will use default)
    workspace = os.getenv("ROBOFLOW_WORKSPACE")
    
    return {
        "api_key": api_key,
        "workspace": workspace,
        "forklift_project": os.getenv("ROBOFLOW_FORKLIFT_PROJECT", "forklift-i3vog-vwafw"),
        "forklift_version": int(os.getenv("ROBOFLOW_FORKLIFT_VERSION", "1")),
        "pallet_project": os.getenv("ROBOFLOW_PALLET_PROJECT", "pallet-unicd-k2rg0"),
        "pallet_version": int(os.getenv("ROBOFLOW_PALLET_VERSION", "1")),
    }


def get_analytics_config() -> dict:
    """
    Get analytics configuration from environment.
    
    Returns:
        Dictionary with analytics settings.
    """
    return {
        "cost_per_idle_hour": float(os.getenv("COST_PER_IDLE_HOUR", "75.0")),
    }


# Auto-load .env on import
load_env_file()
