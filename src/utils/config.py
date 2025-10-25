"""
Configuration management for the monitoring system.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class DataConfig(BaseModel):
    """Data-related configuration."""
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    synthetic_data_dir: Path = Field(default=Path("data/synthetic"))
    test_size: float = Field(default=0.2, ge=0.0, le=1.0)
    random_state: int = Field(default=42)


class ModelConfig(BaseModel):
    """Model training configuration."""
    model_type: str = Field(default="xgboost")
    max_depth: int = Field(default=6)
    n_estimators: int = Field(default=100)
    learning_rate: float = Field(default=0.1)
    random_state: int = Field(default=42)


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    drift_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    performance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    fairness_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    monitoring_interval: int = Field(default=3600)  # seconds
    alert_enabled: bool = Field(default=True)


class APIConfig(BaseModel):
    """API configuration."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=True)
    workers: int = Field(default=1)
    log_level: str = Field(default="info")


class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8501)
    theme: str = Field(default="light")


class Config(BaseModel):
    """Main configuration class."""
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    
    project_name: str = Field(default="Trustworthy AI Monitor")
    version: str = Field(default="0.1.0")
    environment: str = Field(default="development")


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file or use defaults.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config: Configuration object
    """
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return Config(**config_dict)
    
    # Use environment variables to override defaults
    config_dict = {}
    
    # API config from environment
    if os.getenv("API_HOST"):
        config_dict.setdefault("api", {})["host"] = os.getenv("API_HOST")
    if os.getenv("API_PORT"):
        config_dict.setdefault("api", {})["port"] = int(os.getenv("API_PORT"))
    
    # Environment
    if os.getenv("ENVIRONMENT"):
        config_dict["environment"] = os.getenv("ENVIRONMENT")
    
    return Config(**config_dict) if config_dict else Config()


def save_config(config: Config, config_path: Path) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object
        config_path: Path to save YAML file
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)


# Global configuration instance
config = load_config()


__all__ = ["Config", "config", "load_config", "save_config"]

