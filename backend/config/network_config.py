#!/usr/bin/env python3
"""
Network Implementation Configuration Loader
Loads configuration parameters from NETWORK_IMPLEMENTATION_CONFIG.md
"""

import yaml
import re
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceConfig:
    """Performance-based incentives configuration."""
    min_stake: int = 0
    entry_barrier: str = "none"
    reward_basis: str = "performance_improvement"
    punishment_method: str = "no_reward"

@dataclass
class QualityGateConfig:
    """Smart anti-spam configuration."""
    model: str = "tiny_moe_toxic_v1.onnx"
    max_dup_similarity: float = 0.95
    max_toxic_score: float = 0.10
    min_perplexity_improvement: float = 0.02
    processing_timeout: str = "1s"

@dataclass
class QuotaConfig:
    """Progressive trust system configuration."""
    newbie_daily: int = 20
    trusted_daily: int = 200
    promotion_threshold: int = 3
    demotion_threshold: int = 5
    quota_recovery: str = "performance_only"

@dataclass
class AnomalyConfig:
    """Behavioral anomaly detection configuration."""
    min_upload_interval: str = "5min"
    max_daily_burst: int = 10
    suspicious_patterns: list = None
    auto_quarantine: bool = True
    
    def __post_init__(self):
        if self.suspicious_patterns is None:
            self.suspicious_patterns = ["regular_intervals", "bulk_upload", "off_hours"]

@dataclass
class ResourceConfig:
    """Resource optimization configuration."""
    validation_as_training: bool = True
    failed_models_as_data: bool = True
    zero_waste_principle: str = "all_computation_contributes_to_ai_advancement"

@dataclass
class EconomicConfig:
    """Economic safeguards configuration."""
    transaction_fee: float = 0.01
    fee_distribution: str = "validator_rewards"
    no_staking_required: bool = True

@dataclass
class TechnicalConfig:
    """Technical parameters configuration."""
    max_block_size: int = 10485760  # 10MB
    difficulty_adjustment: int = 2016
    consensus_mechanism: str = "proof_of_learning"

@dataclass
class DataEthicsConfig:
    """Data ethics configuration."""
    allowed_licenses: list = None
    data_consent_required: bool = True
    privacy_protection: bool = True
    
    def __post_init__(self):
        if self.allowed_licenses is None:
            self.allowed_licenses = ["CC0", "CC-BY", "CC-BY-SA", "Apache-2.0", "MIT"]

@dataclass
class EvolutionConfig:
    """Evolution configuration."""
    auto_upgrade_threshold: float = 0.51
    backward_compatibility: int = 3
    deprecation_notice: int = 90
    emergency_patch_quorum: float = 0.33

class NetworkConfig:
    """
    Centralized configuration loader for network implementation parameters.
    Parses YAML blocks from NETWORK_IMPLEMENTATION_CONFIG.md
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize network configuration."""
        if config_file is None:
            config_file = Path(__file__).parent.parent.parent / "docs" / "NETWORK_IMPLEMENTATION_CONFIG.md"
        
        self.config_file = Path(config_file)
        self._load_config()
    
    def _load_config(self):
        """Load configuration from markdown file with YAML blocks."""
        if not self.config_file.exists():
            logger.warning(f"Config file not found: {self.config_file}. Using defaults.")
            self._load_defaults()
            return
        
        try:
            content = self.config_file.read_text()
            yaml_blocks = self._extract_yaml_blocks(content)
            
            # Load each configuration section
            self.performance = self._load_section(yaml_blocks, "performance", PerformanceConfig)
            self.quality_gate = self._load_section(yaml_blocks, "quality_gate", QualityGateConfig)
            self.quota = self._load_section(yaml_blocks, "quota_management", QuotaConfig)
            self.anomaly = self._load_section(yaml_blocks, "anomaly_detection", AnomalyConfig)
            self.resource = self._load_section(yaml_blocks, "resource_efficiency", ResourceConfig)
            self.economic = self._load_section(yaml_blocks, "economic_safeguards", EconomicConfig)
            self.technical = self._load_section(yaml_blocks, "technical", TechnicalConfig)
            self.data_ethics = self._load_section(yaml_blocks, "data_ethics", DataEthicsConfig)
            self.evolution = self._load_section(yaml_blocks, "evolution", EvolutionConfig)
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}. Using defaults.")
            self._load_defaults()
    
    def _extract_yaml_blocks(self, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract YAML blocks from markdown content."""
        yaml_blocks = {}
        
        # Find all YAML code blocks
        yaml_pattern = r'```yaml\n(.*?)\n```'
        matches = re.findall(yaml_pattern, content, re.DOTALL)
        
        for yaml_content in matches:
            try:
                data = yaml.safe_load(yaml_content)
                if isinstance(data, dict):
                    yaml_blocks.update(data)
            except yaml.YAMLError as e:
                logger.warning(f"Failed to parse YAML block: {e}")
        
        return yaml_blocks
    
    def _load_section(self, yaml_blocks: Dict[str, Any], section_name: str, config_class):
        """Load a specific configuration section."""
        section_data = yaml_blocks.get(section_name, {})
        
        try:
            return config_class(**section_data)
        except Exception as e:
            logger.warning(f"Failed to load {section_name} config: {e}. Using defaults.")
            return config_class()
    
    def _load_defaults(self):
        """Load default configuration values."""
        self.performance = PerformanceConfig()
        self.quality_gate = QualityGateConfig()
        self.quota = QuotaConfig()
        self.anomaly = AnomalyConfig()
        self.resource = ResourceConfig()
        self.economic = EconomicConfig()
        self.technical = TechnicalConfig()
        self.data_ethics = DataEthicsConfig()
        self.evolution = EvolutionConfig()
    
    def get_upload_quota(self, user_trust_level: str = "newbie") -> int:
        """Get daily upload quota based on trust level."""
        if user_trust_level == "trusted":
            return self.quota.trusted_daily
        return self.quota.newbie_daily
    
    def should_auto_quarantine(self) -> bool:
        """Check if auto-quarantine is enabled."""
        return self.anomaly.auto_quarantine
    
    def get_max_block_size(self) -> int:
        """Get maximum block size in bytes."""
        return self.technical.max_block_size
    
    def get_transaction_fee(self) -> float:
        """Get network transaction fee."""
        return self.economic.transaction_fee
    
    def is_license_allowed(self, license_name: str) -> bool:
        """Check if a license is allowed for data."""
        return license_name in self.data_ethics.allowed_licenses


# Global instance
_network_config = None

def get_network_config() -> NetworkConfig:
    """Get the global network configuration instance."""
    global _network_config
    if _network_config is None:
        _network_config = NetworkConfig()
    return _network_config

def reload_network_config():
    """Reload the network configuration from file."""
    global _network_config
    _network_config = None
    return get_network_config()


if __name__ == "__main__":
    # Test the configuration loader
    config = NetworkConfig()
    
    print("🔧 Network Configuration Loaded:")
    print(f"   Newbie daily quota: {config.quota.newbie_daily}")
    print(f"   Trusted daily quota: {config.quota.trusted_daily}")
    print(f"   Max block size: {config.technical.max_block_size:,} bytes")
    print(f"   Transaction fee: {config.economic.transaction_fee}")
    print(f"   Auto quarantine: {config.anomaly.auto_quarantine}")
    print(f"   Allowed licenses: {config.data_ethics.allowed_licenses}")