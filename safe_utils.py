#!/usr/bin/env python3
"""
safe_utils.py - Safe Error Handling & Utility Functions
แก้ไข: ไม่มี error handling, infinite recursion, memory leaks, poor error messages

Features:
- Comprehensive error handling with graceful degradation
- Timeout protection และ resource limits
- Safe recursive operations with depth limits
- Memory-efficient operations
- Structured error reporting
- Performance monitoring utilities
"""

import time
import json

import os
import traceback
import functools
import threading
import psutil
import signal
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from enum import Enum
import logging

from core_config import get_config

logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
JSONType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class OperationType(Enum):
    SCHEMA_DETECTION = "schema_detection"
    FIELD_EXTRACTION = "field_extraction"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    FILE_IO = "file_io"
    AI_PROCESSING = "ai_processing"

@dataclass
class ErrorInfo:
    """Structured error information"""
    error_id: str
    operation: OperationType
    severity: ErrorSeverity
    message: str
    details: Optional[str] = None
    suggestions: List[str] = None
    timestamp: str = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.suggestions is None:
            self.suggestions = []
        if self.context is None:
            self.context = {}

@dataclass
class OperationResult:
    """Standard operation result"""
    success: bool
    data: Any = None
    error: Optional[ErrorInfo] = None
    warnings: List[str] = None
    metrics: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metrics is None:
            self.metrics = {}
        if self.metadata is None:
            self.metadata = {}

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

class ResourceLimitError(Exception):
    """Resource limit exceeded exception"""
    pass

class SafeOperationHandler:
    """Safe operation handler with timeout and resource limits"""
    
    def __init__(self):
        self.config = get_config()
        self._operation_count = 0
        self._start_time = time.time()
    
    def timeout_handler(self, signum, frame):
        """Signal handler for timeout"""
        raise TimeoutError("Operation timed out")
    
    @contextmanager
    def timeout_context(self, timeout_seconds: int):
        """Context manager for timeout protection"""
        if timeout_seconds <= 0:
            yield
            return
        
        # signal.SIGALRM is not available on Windows.
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, self.timeout_handler)
            signal.alarm(timeout_seconds)
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # On Windows, signal-based timeout is not possible.
            # We can just yield and proceed without a timeout.
            yield
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits"""
        try:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            limit_mb = self.config.performance.max_memory_usage_mb
            
            if memory_mb > limit_mb:
                logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds limit {limit_mb}MB")
                return False
            return True
        except Exception:
            return True  # If we can't check, assume it's okay
    
    def safe_execute(self, 
                    operation: Callable[[], T], 
                    operation_type: OperationType,
                    timeout_seconds: Optional[int] = None,
                    error_message: str = "Operation failed") -> OperationResult:
        """Safely execute an operation with full protection"""
        
        start_time = time.time()
        self._operation_count += 1
        
        # Use config timeout if not specified
        if timeout_seconds is None:
            timeout_seconds = self.config.performance.max_processing_time_seconds
        
        try:
            # Check memory before starting
            if not self.check_memory_usage():
                return OperationResult(
                    success=False,
                    error=ErrorInfo(
                        error_id=f"mem_{self._operation_count}",
                        operation=operation_type,
                        severity=ErrorSeverity.HIGH,
                        message="Memory limit exceeded",
                        suggestions=["Reduce input size", "Enable streaming mode"]
                    )
                )
            
            # Execute with timeout protection
            with self.timeout_context(timeout_seconds) as timeout_checker:
                result = operation()
                
                # Final memory check
                if not self.check_memory_usage():
                    logger.warning("Memory usage high after operation")
                
                execution_time = time.time() - start_time
                
                return OperationResult(
                    success=True,
                    data=result,
                    metrics={
                        'execution_time': execution_time,
                        'operation_count': self._operation_count
                    }
                )
        
        except TimeoutError:
            return OperationResult(
                success=False,
                error=ErrorInfo(
                    error_id=f"timeout_{self._operation_count}",
                    operation=operation_type,
                    severity=ErrorSeverity.HIGH,
                    message=f"Operation timed out after {timeout_seconds} seconds",
                    suggestions=["Increase timeout", "Optimize input data", "Use streaming mode"]
                )
            )
        
        except MemoryError:
            return OperationResult(
                success=False,
                error=ErrorInfo(
                    error_id=f"memory_{self._operation_count}",
                    operation=operation_type,
                    severity=ErrorSeverity.CRITICAL,
                    message="Out of memory",
                    suggestions=["Reduce input size", "Enable pagination", "Process in smaller chunks"]
                )
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_details = traceback.format_exc()
            
            return OperationResult(
                success=False,
                error=ErrorInfo(
                    error_id=f"error_{self._operation_count}",
                    operation=operation_type,
                    severity=self._classify_error_severity(e),
                    message=f"{error_message}: {str(e)}",
                    details=error_details,
                    suggestions=self._generate_error_suggestions(e, operation_type),
                    context={
                        'execution_time': execution_time,
                        'operation_count': self._operation_count
                    }
                )
            )
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on exception type"""
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (TimeoutError, ResourceLimitError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError, KeyError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _generate_error_suggestions(self, error: Exception, operation_type: OperationType) -> List[str]:
        """Generate helpful suggestions based on error and operation type"""
        suggestions = []
        
        error_type = type(error).__name__
        
        # General suggestions by error type
        if error_type == "KeyError":
            suggestions.extend([
                "Check if required fields exist in input data",
                "Verify schema mapping configuration",
                "Use safe field access methods"
            ])
        elif error_type == "TypeError":
            suggestions.extend([
                "Verify input data types",
                "Check for None values",
                "Validate data structure"
            ])
        elif error_type == "ValueError":
            suggestions.extend([
                "Check input value ranges",
                "Validate data format",
                "Use data conversion functions"
            ])
        elif error_type == "FileNotFoundError":
            suggestions.extend([
                "Verify file path exists",
                "Check file permissions",
                "Use absolute file paths"
            ])
        
        # Operation-specific suggestions
        if operation_type == OperationType.SCHEMA_DETECTION:
            suggestions.extend([
                "Enable fallback detection mode",
                "Lower confidence threshold",
                "Check input data structure"
            ])
        elif operation_type == OperationType.FIELD_EXTRACTION:
            suggestions.extend([
                "Use alternative field paths",
                "Enable semantic matching",
                "Check field name patterns"
            ])
        elif operation_type == OperationType.TRANSFORMATION:
            suggestions.extend([
                "Verify transformation mappings",
                "Enable graceful degradation",
                "Use safe conversion methods"
            ])
        elif operation_type == OperationType.AI_PROCESSING:
            suggestions.extend([
                "Check AI model availability",
                "Reduce input complexity",
                "Use traditional fallback methods"
            ])
        
        return suggestions[:5]  # Limit to 5 suggestions

class SafeDataProcessor:
    """Safe data processing utilities"""
    
    @staticmethod
    def safe_recursive_process(data: Any, 
                             processor: Callable[[Any], T],
                             max_depth: int = 10,
                             current_depth: int = 0) -> Optional[T]:
        """Safely process nested data with depth limits"""
        
        if current_depth >= max_depth:
            logger.warning(f"Maximum recursion depth {max_depth} reached")
            return None
        
        try:
            if data is None:
                return None
            
            return processor(data)
            
        except RecursionError:
            logger.error("Recursion limit exceeded")
            return None
        except Exception as e:
            logger.warning(f"Safe recursive processing failed at depth {current_depth}: {e}")
            return None
    
    @staticmethod
    def safe_json_parse(data: str, default: JSONType = None) -> JSONType:
        """Safely parse JSON with fallback"""
        if not data or not isinstance(data, str):
            return default
        
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return default
        except Exception as e:
            logger.warning(f"Unexpected JSON parse error: {e}")
            return default
    
    @staticmethod
    def safe_get_nested(data: Dict[str, Any], 
                       path: str, 
                       default: Any = None,
                       separator: str = '.') -> Any:
        """Safely get nested dictionary value"""
        if not isinstance(data, dict) or not path:
            return default
        
        try:
            current = data
            parts = path.split(separator)
            
            for part in parts:
                if not part:  # Skip empty parts
                    continue
                
                # Handle array notation
                if '[' in part and ']' in part:
                    try:
                        key = part.split('[')[0]
                        index_str = part.split('[')[1].split(']')[0]
                        index = int(index_str)
                        
                        if key in current and isinstance(current[key], list):
                            if 0 <= index < len(current[key]):
                                current = current[key][index]
                            else:
                                return default
                        else:
                            return default
                    except (ValueError, IndexError):
                        return default
                else:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        return default
            
            return current
            
        except Exception as e:
            logger.warning(f"Safe nested get failed for path '{path}': {e}")
            return default
    
    @staticmethod
    def safe_set_nested(data: Dict[str, Any], 
                       path: str, 
                       value: Any,
                       separator: str = '.') -> bool:
        """Safely set nested dictionary value"""
        if not isinstance(data, dict) or not path:
            return False
        
        try:
            parts = path.split(separator)
            current = data
            
            # Navigate to parent
            for part in parts[:-1]:
                if not part:
                    continue
                
                if part not in current:
                    current[part] = {}
                elif not isinstance(current[part], dict):
                    return False  # Can't navigate further
                
                current = current[part]
            
            # Set final value
            final_part = parts[-1]
            if final_part:
                current[final_part] = value
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Safe nested set failed for path '{path}': {e}")
            return False
    
    @staticmethod
    def safe_list_get(data: List[Any], index: int, default: Any = None) -> Any:
        """Safely get list item by index"""
        try:
            if isinstance(data, list) and 0 <= index < len(data):
                return data[index]
            return default
        except Exception:
            return default
    
    @staticmethod
    def clean_string(text: str, max_length: int = 1000) -> str:
        """Clean and truncate string safely"""
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Remove control characters
        cleaned = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Truncate if too long
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length - 3] + "..."
        
        return cleaned.strip()
    
    @staticmethod
    def safe_type_conversion(value: Any, target_type: type, default: Any = None) -> Any:
        """Safely convert value to target type"""
        if value is None:
            return default
        
        try:
            if target_type == bool:
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
            elif target_type == int:
                if isinstance(value, str):
                    # Remove non-numeric characters except minus
                    numeric_str = ''.join(c for c in value if c.isdigit() or c == '-')
                    return int(numeric_str) if numeric_str else default
                return int(value)
            elif target_type == float:
                return float(value)
            elif target_type == str:
                return str(value)
            elif target_type == list:
                if isinstance(value, list):
                    return value
                return [value]
            else:
                return target_type(value)
                
        except (ValueError, TypeError, OverflowError):
            return default

class PerformanceMonitor:
    """Performance monitoring utilities"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    @contextmanager
    def measure(self, operation_name: str):
        """Context manager for measuring operation time"""
        start_time = time.time()
        self.start_times[operation_name] = start_time
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            
            self.metrics[operation_name].append({
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'memory_mb': self._get_memory_usage()
            })
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def get_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get performance statistics for operation"""
        if operation_name not in self.metrics:
            return {}
        
        durations = [m['duration'] for m in self.metrics[operation_name]]
        
        return {
            'count': len(durations),
            'total_time': sum(durations),
            'avg_time': sum(durations) / len(durations),
            'min_time': min(durations),
            'max_time': max(durations),
            'last_run': self.metrics[operation_name][-1]['timestamp']
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all performance statistics"""
        return {op: self.get_stats(op) for op in self.metrics.keys()}

class SafeFileHandler:
    """Safe file handling utilities"""
    
    @staticmethod
    def safe_read_file(file_path: Union[str, Path], 
                      max_size_mb: Optional[int] = None,
                      encoding: str = 'utf-8') -> OperationResult:
        """Safely read file with size limits"""
        handler = SafeOperationHandler()
        config = get_config()
        
        if max_size_mb is None:
            max_size_mb = config.performance.max_file_size_mb
        
        def read_operation():
            file_path_obj = Path(file_path)
            
            # Check if file exists
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check file size
            file_size_mb = file_path_obj.stat().st_size / 1024 / 1024
            if file_size_mb > max_size_mb:
                raise ResourceLimitError(f"File size {file_size_mb:.1f}MB exceeds limit {max_size_mb}MB")
            
            # Read file
            with open(file_path_obj, 'r', encoding=encoding) as f:
                content = f.read()
            
            return content
        
        return handler.safe_execute(
            read_operation,
            OperationType.FILE_IO,
            error_message=f"Failed to read file {file_path}"
        )
    
    @staticmethod
    def safe_write_file(file_path: Union[str, Path], 
                    content: str,
                    encoding: str = 'utf-8',
                    create_dirs: bool = True) -> OperationResult:
        """Safely write file with enhanced error handling and logging"""
        handler = SafeOperationHandler()
        
        def write_operation():
            file_path_obj = Path(file_path).resolve()  # Convert to absolute path
            
            print(f"Writing to: {file_path_obj}")
            logger.info(f"Writing file to: {file_path_obj}")
            
            # Create directories if needed
            if create_dirs:
                file_path_obj.parent.mkdir(parents=True, exist_ok=True)
                print(f"Directory ensured: {file_path_obj.parent}")
            
            # Check if directory is writable
            if not os.access(file_path_obj.parent, os.W_OK):
                raise PermissionError(f"No write permission to directory: {file_path_obj.parent}")
            
            # Write to temporary file first (atomic operation)
            temp_file = file_path_obj.with_suffix(file_path_obj.suffix + '.tmp')
            
            try:
                with open(temp_file, 'w', encoding=encoding) as f:
                    f.write(content)
                    f.flush()  # Ensure data is written
                    os.fsync(f.fileno())  # Force write to disk
                
                # Atomic rename
                temp_file.rename(file_path_obj)
                
                # Verify file was created
                if not file_path_obj.exists():
                    raise FileNotFoundError(f"File was not created: {file_path_obj}")
                
                file_size = file_path_obj.stat().st_size
                print(f"File written successfully: {file_size:,} bytes")
                logger.info(f"File written successfully: {file_path_obj} ({file_size} bytes)")
                
                return str(file_path_obj)
                
            except Exception as e:
                # Clean up temp file if it exists
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass
                raise e
        
        return handler.safe_execute(
            write_operation,
            OperationType.FILE_IO,
            error_message=f"Failed to write file {file_path}"
        )

    @staticmethod
    def safe_write_json(file_path: Union[str, Path], 
                    data: JSONType,
                    indent: int = 2,
                    ensure_ascii: bool = False) -> OperationResult:
        """Safely write data as JSON file with validation"""
        handler = SafeOperationHandler()
        
        def write_operation():
            try:
                # Validate data can be serialized
                json_content = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, default=str)
            except Exception as e:
                raise ValueError(f"Data cannot be serialized to JSON: {e}")
            
            # Validate JSON content size
            content_size_mb = len(json_content.encode('utf-8')) / 1024 / 1024
            max_size_mb = get_config().performance.max_file_size_mb
            
            if content_size_mb > max_size_mb:
                logger.warning(f"Large JSON content: {content_size_mb:.1f}MB (limit: {max_size_mb}MB)")
            
            # Write using safe_write_file
            write_result = SafeFileHandler.safe_write_file(file_path, json_content)
            
            if not write_result.success:
                raise Exception(write_result.error.message if write_result.error else 'Unknown write error')
            
            return write_result.data
        
        return handler.safe_execute(
            write_operation,
            OperationType.FILE_IO,
            error_message=f"Failed to write JSON file {file_path}"
        )

    @staticmethod
    def get_safe_output_path(input_file: str, suffix: str = "_processed", 
                            output_dir: str = None) -> Path:
        """Generate safe output path with proper directory structure"""
        
        input_path = Path(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_dir:
            output_dir_path = Path(output_dir)
        else:
            # Create output directory next to input file
            output_dir_path = input_path.parent / "output"
        
        # Create output directory
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        output_filename = f"{input_path.stem}{suffix}_{timestamp}.json"
        output_path = output_dir_path / output_filename
        
        return output_path.resolve()

    @staticmethod
    def safe_read_json(file_path: Union[str, Path], 
                    max_size_mb: Optional[int] = None) -> OperationResult:
        """Safely read and parse JSON file"""
        # Read file first
        read_result = SafeFileHandler.safe_read_file(file_path, max_size_mb)
        
        if not read_result.success:
            return read_result
        
        # Parse JSON
        handler = SafeOperationHandler()
        
        def parse_operation():
            return SafeDataProcessor.safe_json_parse(read_result.data)
        
        return handler.safe_execute(
            parse_operation,
            OperationType.FILE_IO,
            error_message=f"Failed to parse JSON file {file_path}"
        )
        
# เพิ่มฟังก์ชัน utility
def check_disk_space(path: Union[str, Path], required_mb: float = 100) -> bool:
    """Check if there's enough disk space"""
    try:
        import shutil
        free_bytes = shutil.disk_usage(Path(path).parent).free
        free_mb = free_bytes / 1024 / 1024
        
        if free_mb < required_mb:
            logger.warning(f"Low disk space: {free_mb:.1f}MB available, {required_mb:.1f}MB required")
            return False
        
        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Assume OK if can't check

def create_output_directory_structure(base_dir: Union[str, Path]) -> Dict[str, Path]:
    """Create organized output directory structure"""
    
    base_path = Path(base_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory structure
    directories = {
        'main': base_path / f"smart_mapper_output_{timestamp}",
        'batch': base_path / f"smart_mapper_output_{timestamp}" / "batch_results",
        'individual': base_path / f"smart_mapper_output_{timestamp}" / "individual_events",
        'summaries': base_path / f"smart_mapper_output_{timestamp}" / "summaries",
        'logs': base_path / f"smart_mapper_output_{timestamp}" / "logs"
    }
    
    # Create all directories
    for dir_type, dir_path in directories.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created: {dir_type} -> {dir_path}")
    
    return directories

# Decorator utilities
def safe_operation(operation_type: OperationType, timeout: int = None):
    """Decorator for safe operation execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> OperationResult:
            handler = SafeOperationHandler()
            
            def operation():
                return func(*args, **kwargs)
            
            return handler.safe_execute(
                operation,
                operation_type,
                timeout_seconds=timeout,
                error_message=f"Function {func.__name__} failed"
            )
        
        return wrapper
    return decorator

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying failed operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        sleep_time = delay * (backoff ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {sleep_time:.1f}s")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator

# Global performance monitor instance
_performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor"""
    return _performance_monitor

# Utility functions for common operations
def safe_execute_with_fallback(primary_func: Callable[[], T], 
                              fallback_func: Callable[[], T],
                              operation_type: OperationType) -> T:
    """Execute primary function with fallback on failure"""
    handler = SafeOperationHandler()
    
    # Try primary function
    result = handler.safe_execute(primary_func, operation_type)
    
    if result.success:
        return result.data
    
    # Try fallback
    logger.warning(f"Primary operation failed, trying fallback: {result.error.message}")
    
    fallback_result = handler.safe_execute(fallback_func, operation_type)
    
    if fallback_result.success:
        logger.info("Fallback operation succeeded")
        return fallback_result.data
    
    # Both failed
    logger.error(f"Both primary and fallback operations failed")
    raise Exception(f"Operation failed: {result.error.message}")

def validate_input_data(data: Any, expected_type: type, operation_type: OperationType) -> OperationResult:
    """Validate input data type and structure"""
    if data is None:
        return OperationResult(
            success=False,
            error=ErrorInfo(
                error_id="validation_null",
                operation=operation_type,
                severity=ErrorSeverity.HIGH,
                message="Input data is None",
                suggestions=["Provide valid input data", "Check data source"]
            )
        )
    
    if not isinstance(data, expected_type):
        return OperationResult(
            success=False,
            error=ErrorInfo(
                error_id="validation_type",
                operation=operation_type,
                severity=ErrorSeverity.HIGH,
                message=f"Expected {expected_type.__name__}, got {type(data).__name__}",
                suggestions=["Check input data format", "Verify data transformation"]
            )
        )
    
    return OperationResult(success=True, data=data)