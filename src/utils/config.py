"""Configuration loader for SRAG."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class Config:
    """SRAG configuration manager."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        load_dotenv()
        self._config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config not found: {self._config_path}")
        with open(self._config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation. E.g., 'neo4j.uri'."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    @property
    def neo4j_uri(self) -> str:
        return self.get("neo4j.uri", "bolt://localhost:7687")

    @property
    def neo4j_user(self) -> str:
        return self.get("neo4j.username", "neo4j")

    @property
    def neo4j_password(self) -> str:
        password = os.getenv("NEO4J_PASSWORD")
        if password:
            return password
        return self.get("neo4j.password", "srag_password")

    @property
    def gemini_api_key(self) -> str:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        return key

    @property
    def mimo_api_key(self) -> str:
        key = os.getenv("MIMO_API_KEY")
        if not key:
            raise ValueError("MIMO_API_KEY not found in environment")
        return key

    @property
    def embedding_model(self) -> str:
        return self.get("embedding.model_name", "sanganaka/bge-m3-sanskritFT")

    @property
    def embedding_device(self) -> str:
        return self.get("embedding.device", "cpu")

    def __repr__(self) -> str:
        return f"Config({self._config_path})"
