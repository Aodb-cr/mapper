#!/usr/bin/env python3
"""
core_config.py - Centralized Configuration System
แก้ไข: Configuration กระจัดกระจาย, ไม่มี validation, ยากต่อการ maintain

Features:
- Single source of truth สำหรับ configuration
- Built-in validation และ defaults
- Environment-aware configuration
- Type safety และ schema validation
- Hot reload capability
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ProcessingMode(Enum):
    SINGLE = "single"
    BATCH = "batch"
    STREAMING = "streaming"

@dataclass
class AIConfig:
    """AI/ML Configuration"""
    enabled: bool = True
    model_name: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.65 #threshold 
    max_cache_size: int = 1000 #embeddings
    timeout_seconds: int = 30
    
    #validate
    def validate(self) -> List[str]:
        errors = []
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            errors.append("similarity_threshold must be between 0 and 1")
        if self.max_cache_size < 100:
            errors.append("max_cache_size must be at least 100")
        if self.timeout_seconds < 5:
            errors.append("timeout_seconds must be at least 5")
        return errors

@dataclass
class SchemaConfig:
    """Schema Detection Configuration"""
    confidence_threshold: float = 40.0
    min_confidence_threshold: float = 30.0
    max_confidence_threshold: float = 95.0
    enable_fallback: bool = True # confidence ต่ำ
    max_recursion_depth: int = 5 # protect infinite loop
    
    # Weights for scoring
    required_paths_weight: float = 40.0
    optional_paths_weight: float = 30.0
    distinctive_fields_weight: float = 30.0 # fieldsไม่ซ้ำ
    
    def validate(self) -> List[str]:
        """Validate schema configuration"""
        errors = []
        if not (0 <= self.confidence_threshold <= 100):
            errors.append("confidence_threshold must be between 0 and 100")
        if self.min_confidence_threshold >= self.max_confidence_threshold:
            errors.append("min_confidence_threshold must be less than max_confidence_threshold")
        if self.max_recursion_depth < 1:
            errors.append("max_recursion_depth must be at least 1")
        
        total_weight = (self.required_paths_weight + 
                       self.optional_paths_weight + 
                       self.distinctive_fields_weight)
        if abs(total_weight - 100.0) > 0.1:
            errors.append("Sum of weights must equal 100.0")
        
        return errors

@dataclass
class PerformanceConfig:
    """Performance and Resource Configuration"""
    max_file_size_mb: int = 100
    max_processing_time_seconds: int = 300
    max_memory_usage_mb: int = 2048
    enable_parallel_processing: bool = True
    max_worker_threads: int = 4
    cache_cleanup_interval_seconds: int = 3600
    
    def validate(self) -> List[str]:
        """Validate performance configuration"""
        errors = []
        if self.max_file_size_mb < 1:
            errors.append("max_file_size_mb must be at least 1")
        if self.max_processing_time_seconds < 10:
            errors.append("max_processing_time_seconds must be at least 10")
        if self.max_worker_threads < 1:
            errors.append("max_worker_threads must be at least 1")
        return errors

@dataclass
class QualityConfig:
    """Quality Analysis Configuration"""
    minimum_quality_score: float = 40.0
    critical_fields_weight: float = 40.0
    field_coverage_weight: float = 25.0
    data_completeness_weight: float = 20.0
    confidence_weight: float = 15.0
    
    # Critical fields for all schemas
    critical_fields: List[str] = field(default_factory=lambda: [
        'alert_name',
        'severity', 
        'detected_time',
        'log_source',
        'contexts.hostname',
        'incident_type'
    ])
    
    def validate(self) -> List[str]:
        """Validate quality configuration"""
        errors = []
        if not (0 <= self.minimum_quality_score <= 100):
            errors.append("minimum_quality_score must be between 0 and 100")
        
        total_weight = (self.critical_fields_weight + 
                       self.field_coverage_weight + 
                       self.data_completeness_weight + 
                       self.confidence_weight)
        if abs(total_weight - 100.0) > 0.1:
            errors.append("Sum of quality weights must equal 100.0")
        
        if not self.critical_fields:
            errors.append("critical_fields cannot be empty")
        
        return errors

@dataclass
class LoggingConfig:
    """Logging Configuration"""
    level: LogLevel = LogLevel.INFO
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    log_file_path: str = "smart_mapper.log"
    max_log_file_size_mb: int = 10
    backup_count: int = 5
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def validate(self) -> List[str]:
        """Validate logging configuration"""
        errors = []
        if self.max_log_file_size_mb < 1:
            errors.append("max_log_file_size_mb must be at least 1")
        if self.backup_count < 0:
            errors.append("backup_count must be non-negative")
        if not (self.enable_file_logging or self.enable_console_logging):
            errors.append("At least one logging output must be enabled")
        return errors

@dataclass
class FeedbackConfig:
    """Feedback and Learning Configuration"""
    enabled: bool = True
    storage_dir: str = "feedback_data"
    auto_apply_threshold: float = 80.0
    min_correction_frequency: int = 2
    max_feedback_history_days: int = 90
    enable_auto_learning: bool = True
    learning_cycle_interval_days: int = 7
    
    def validate(self) -> List[str]:
        """Validate feedback configuration"""
        errors = []
        if not (0 <= self.auto_apply_threshold <= 100):
            errors.append("auto_apply_threshold must be between 0 and 100")
        if self.min_correction_frequency < 1:
            errors.append("min_correction_frequency must be at least 1")
        if self.max_feedback_history_days < 1:
            errors.append("max_feedback_history_days must be at least 1")
        return errors

@dataclass
class SmartMapperConfig:
    """Main Configuration Class"""
    # Core configurations
    ai: AIConfig = field(default_factory=AIConfig)
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    
    # Global settings
    environment: str = "development"
    debug_mode: bool = False
    data_dir: str = "data"
    cache_dir: str = "cache"
    backup_dir: str = "backups"
    
    # Processing defaults
    default_processing_mode: ProcessingMode = ProcessingMode.SINGLE
    enable_interactive_mode: bool = True
    auto_detect_multiple_events: bool = True
    
    def validate(self) -> Dict[str, List[str]]:
        """Validate entire configuration"""
        all_errors = {}
        
        # Validate each section
        sections = {
            'ai': self.ai,
            'schema': self.schema,
            'performance': self.performance,
            'quality': self.quality,
            'logging': self.logging,
            'feedback': self.feedback
        }
        
        for section_name, section_config in sections.items():
            if hasattr(section_config, 'validate'):
                errors = section_config.validate()
                if errors:
                    all_errors[section_name] = errors
        
        # Global validation
        global_errors = []
        if not self.environment:
            global_errors.append("environment cannot be empty")
        if not self.data_dir:
            global_errors.append("data_dir cannot be empty")
        
        if global_errors:
            all_errors['global'] = global_errors
        
        return all_errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return len(self.validate()) == 0

class ConfigurationManager:
    """Central Configuration Manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._find_config_file()
        self.config = SmartMapperConfig()
        self._load_config()
        self._setup_logging()
    
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in standard locations"""
        possible_locations = [
            "config.yaml",
            "config.yml", 
            "config.json",
            "smart_mapper_config.yaml",
            os.path.expanduser("~/.smart_mapper/config.yaml"),
            "/etc/smart_mapper/config.yaml"
        ]
        
        for location in possible_locations:
            if Path(location).exists():
                logger.info(f"Found config file: {location}")
                return location
        
        logger.info("No config file found, using defaults")
        return None
    
    def _load_config(self):
        """Load configuration from file"""
        if not self.config_file or not Path(self.config_file).exists():
            logger.info("Using default configuration")
            self._apply_environment_overrides()
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.endswith(('.yaml', '.yml')):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Merge with defaults
            self._merge_config(data)
            self._apply_environment_overrides()
            
            # Validate
            errors = self.config.validate()
            if errors:
                logger.error(f"Configuration validation errors: {errors}")
                raise ValueError(f"Invalid configuration: {errors}")
            
            logger.info(f"Configuration loaded successfully from {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_file}: {e}")
            logger.info("Falling back to default configuration")
            self.config = SmartMapperConfig()
            self._apply_environment_overrides()
    
    def _merge_config(self, data: Dict[str, Any]):
            """Merge loaded data with default configuration"""
            # Handle nested config merging properly for dataclasses
            
            # AI config
            if 'ai' in data:
                ai_data = data['ai']
                self.config.ai = AIConfig(
                    enabled=ai_data.get('enabled', self.config.ai.enabled),
                    model_name=ai_data.get('model_name', self.config.ai.model_name),
                    similarity_threshold=ai_data.get('similarity_threshold', self.config.ai.similarity_threshold),
                    max_cache_size=ai_data.get('max_cache_size', self.config.ai.max_cache_size),
                    timeout_seconds=ai_data.get('timeout_seconds', self.config.ai.timeout_seconds)
                )
            
            # Schema config
            if 'schema' in data:
                schema_data = data['schema']
                self.config.schema = SchemaConfig(
                    confidence_threshold=schema_data.get('confidence_threshold', self.config.schema.confidence_threshold),
                    min_confidence_threshold=schema_data.get('min_confidence_threshold', self.config.schema.min_confidence_threshold),
                    max_confidence_threshold=schema_data.get('max_confidence_threshold', self.config.schema.max_confidence_threshold),
                    enable_fallback=schema_data.get('enable_fallback', self.config.schema.enable_fallback),
                    max_recursion_depth=schema_data.get('max_recursion_depth', self.config.schema.max_recursion_depth),
                    required_paths_weight=schema_data.get('required_paths_weight', self.config.schema.required_paths_weight),
                    optional_paths_weight=schema_data.get('optional_paths_weight', self.config.schema.optional_paths_weight),
                    distinctive_fields_weight=schema_data.get('distinctive_fields_weight', self.config.schema.distinctive_fields_weight)
                )
            
            # Performance config
            if 'performance' in data:
                perf_data = data['performance']
                self.config.performance = PerformanceConfig(
                    max_file_size_mb=perf_data.get('max_file_size_mb', self.config.performance.max_file_size_mb),
                    max_processing_time_seconds=perf_data.get('max_processing_time_seconds', self.config.performance.max_processing_time_seconds),
                    max_memory_usage_mb=perf_data.get('max_memory_usage_mb', self.config.performance.max_memory_usage_mb),
                    enable_parallel_processing=perf_data.get('enable_parallel_processing', self.config.performance.enable_parallel_processing),
                    max_worker_threads=perf_data.get('max_worker_threads', self.config.performance.max_worker_threads),
                    cache_cleanup_interval_seconds=perf_data.get('cache_cleanup_interval_seconds', self.config.performance.cache_cleanup_interval_seconds)
                )
            
            # Quality config
            if 'quality' in data:
                quality_data = data['quality']
                self.config.quality = QualityConfig(
                    minimum_quality_score=quality_data.get('minimum_quality_score', self.config.quality.minimum_quality_score),
                    critical_fields_weight=quality_data.get('critical_fields_weight', self.config.quality.critical_fields_weight),
                    field_coverage_weight=quality_data.get('field_coverage_weight', self.config.quality.field_coverage_weight),
                    data_completeness_weight=quality_data.get('data_completeness_weight', self.config.quality.data_completeness_weight),
                    confidence_weight=quality_data.get('confidence_weight', self.config.quality.confidence_weight),
                    critical_fields=quality_data.get('critical_fields', self.config.quality.critical_fields)
                )
            
            # Logging config
            if 'logging' in data:
                log_data = data['logging']
                # Handle LogLevel enum conversion
                log_level = log_data.get('level', self.config.logging.level)
                if isinstance(log_level, str):
                    try:
                        log_level = LogLevel(log_level.upper())
                    except ValueError:
                        log_level = LogLevel.INFO
                
                self.config.logging = LoggingConfig(
                    level=log_level,
                    enable_file_logging=log_data.get('enable_file_logging', self.config.logging.enable_file_logging),
                    enable_console_logging=log_data.get('enable_console_logging', self.config.logging.enable_console_logging),
                    log_file_path=log_data.get('log_file_path', self.config.logging.log_file_path),
                    max_log_file_size_mb=log_data.get('max_log_file_size_mb', self.config.logging.max_log_file_size_mb),
                    backup_count=log_data.get('backup_count', self.config.logging.backup_count),
                    log_format=log_data.get('log_format', self.config.logging.log_format)
                )
            
            # Feedback config
            if 'feedback' in data:
                feedback_data = data['feedback']
                self.config.feedback = FeedbackConfig(
                    enabled=feedback_data.get('enabled', self.config.feedback.enabled),
                    storage_dir=feedback_data.get('storage_dir', self.config.feedback.storage_dir),
                    auto_apply_threshold=feedback_data.get('auto_apply_threshold', self.config.feedback.auto_apply_threshold),
                    min_correction_frequency=feedback_data.get('min_correction_frequency', self.config.feedback.min_correction_frequency),
                    max_feedback_history_days=feedback_data.get('max_feedback_history_days', self.config.feedback.max_feedback_history_days),
                    enable_auto_learning=feedback_data.get('enable_auto_learning', self.config.feedback.enable_auto_learning),
                    learning_cycle_interval_days=feedback_data.get('learning_cycle_interval_days', self.config.feedback.learning_cycle_interval_days)
                )
            
            # Global settings
            self.config.environment = data.get('environment', self.config.environment)
            self.config.debug_mode = data.get('debug_mode', self.config.debug_mode)
            self.config.data_dir = data.get('data_dir', self.config.data_dir)
            self.config.cache_dir = data.get('cache_dir', self.config.cache_dir)
            self.config.backup_dir = data.get('backup_dir', self.config.backup_dir)
            
            # Processing defaults
            default_mode = data.get('default_processing_mode', self.config.default_processing_mode)
            if isinstance(default_mode, str):
                try:
                    default_mode = ProcessingMode(default_mode.lower())
                except ValueError:
                    default_mode = ProcessingMode.SINGLE
            self.config.default_processing_mode = default_mode
            
            self.config.enable_interactive_mode = data.get('enable_interactive_mode', self.config.enable_interactive_mode)
            self.config.auto_detect_multiple_events = data.get('auto_detect_multiple_events', self.config.auto_detect_multiple_events)
        
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            'SMART_MAPPER_DEBUG': ('debug_mode', bool),
            'SMART_MAPPER_ENV': ('environment', str),
            'SMART_MAPPER_AI_ENABLED': ('ai_enabled', bool),
            'SMART_MAPPER_CONFIDENCE_THRESHOLD': ('confidence_threshold', float),
            'SMART_MAPPER_MAX_FILE_SIZE': ('max_file_size_mb', int),
            'SMART_MAPPER_LOG_LEVEL': ('log_level', str),
        }
        
        for env_var, (config_attr, data_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if data_type == bool:
                        value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif data_type == str:
                        value = env_value
                    else:
                        value = data_type(env_value)
                    
                    # Apply environment overrides to specific config sections
                    if config_attr == 'debug_mode':
                        self.config.debug_mode = value
                    elif config_attr == 'environment':
                        self.config.environment = value
                    elif config_attr == 'ai_enabled':
                        self.config.ai.enabled = value
                    elif config_attr == 'confidence_threshold':
                        self.config.schema.confidence_threshold = value
                    elif config_attr == 'max_file_size_mb':
                        self.config.performance.max_file_size_mb = value
                    elif config_attr == 'log_level':
                        try:
                            self.config.logging.level = LogLevel(value.upper())
                        except ValueError:
                            pass
                    
                    logger.info(f"Applied environment override: {env_var} = {value}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment value for {env_var}: {env_value}, error: {e}")
    
    def _set_nested_value(self, path: str, value: Any):
        """Set nested configuration value"""
        parts = path.split('.')
        current = self.config
        
        for part in parts[:-1]:
            current = getattr(current, part)
        
        setattr(current, parts[-1], value)
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config.logging
        
        # Create logger
        root_logger = logging.getLogger()
        
        # Handle LogLevel properly
        if isinstance(log_config.level, LogLevel):
            log_level = log_config.level.value
        else:
            log_level = str(log_config.level).upper()
        
        root_logger.setLevel(getattr(logging, log_level))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        formatter = logging.Formatter(log_config.log_format)
        
        # Console handler
        if log_config.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if log_config.enable_file_logging:
            from logging.handlers import RotatingFileHandler
            
            log_dir = Path(log_config.log_file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_config.log_file_path,
                maxBytes=log_config.max_log_file_size_mb * 1024 * 1024,
                backupCount=log_config.backup_count
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        logger.info("Logging configured successfully")
    
    def get_config(self) -> SmartMapperConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values"""
        try:
            # Apply updates using the same merge logic
            self._merge_config(updates)
            
            # Validate
            errors = self.config.validate()
            
            if errors:
                logger.error(f"Configuration update validation failed: {errors}")
                return False
            
            logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    def save_config(self, file_path: Optional[str] = None) -> bool:
        """Save current configuration to file"""
        target_file = file_path or self.config_file or "config.yaml"
        
        try:
            config_dict = asdict(self.config)
            
            # Convert enums to strings
            def convert_enums(obj):
                if isinstance(obj, dict):
                    return {k: convert_enums(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_enums(item) for item in obj]
                elif isinstance(obj, Enum):
                    return obj.value
                return obj
            
            config_dict = convert_enums(config_dict)
            
            Path(target_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(target_file, 'w', encoding='utf-8') as f:
                if target_file.endswith(('.yaml', '.yml')):
                    yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {target_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {target_file}: {e}")
            return False
    
    def reload_config(self) -> bool:
        """Reload configuration from file"""
        try:
            self._load_config()
            self._setup_logging()
            logger.info("Configuration reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def get_schema_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get schema-specific configurations"""
        # This would be extended to load schema-specific configs
        # For now, return basic schema configuration
        return {
            'detection': asdict(self.config.schema),
            'quality': asdict(self.config.quality)
        }

# Global configuration instance
_config_manager: Optional[ConfigurationManager] = None

def get_config_manager(config_file: Optional[str] = None) -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_file)
    return _config_manager

def get_config() -> SmartMapperConfig:
    """Get current configuration"""
    return get_config_manager().get_config()

# Utility functions for common config access
def is_ai_enabled() -> bool:
    """Check if AI features are enabled"""
    return get_config().ai.enabled

def get_confidence_threshold() -> float:
    """Get schema detection confidence threshold"""
    return get_config().schema.confidence_threshold

def get_max_file_size_mb() -> int:
    """Get maximum file size limit"""
    return get_config().performance.max_file_size_mb

def is_debug_mode() -> bool:
    """Check if debug mode is enabled"""
    return get_config().debug_mode