#!/usr/bin/env python3
"""
smart_mapper_main.py - Fixed Architecture Smart JSON Mapper
แก้ไข: Circular dependencies, Memory leaks, Performance issues

Changes:
1. Lazy imports แทน direct imports
2. Simplified event detection (O(n) แทน O(n³))
3. Memory management ใน AI components
4. Better error handling
"""

import sys
import json
import argparse
import os
import time
import gc
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging

# Force UTF-8 for stdout and stderr to handle non-ASCII paths in logs
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Core system imports - ไม่เปลี่ยน
from core_config import get_config_manager, get_config
from safe_utils import SafeOperationHandler, SafeFileHandler, SafeDataProcessor, OperationType, OperationResult, get_performance_monitor
from practical_auto_learning import AutoLearningIntegrator

# Lazy import variables
_semantic_engine = None
_semantic_collector = None
_feedback_manager = None
_field_extractor = None
_schema_detector = None

def get_semantic_engine():
    """Lazy import semantic engine"""
    global _semantic_engine
    if _semantic_engine is None:
        try:
            from semantic_enhancements import SemanticLearningEngine
            _semantic_engine = SemanticLearningEngine()
            logger.info(" Semantic Learning engine loaded")
        except ImportError as e:
            logger.warning(f" Semantic Learning unavailable: {e}")
            _semantic_engine = False
    return _semantic_engine if _semantic_engine is not False else None

def get_semantic_collector():
    """Lazy import semantic collector"""
    global _semantic_collector
    if _semantic_collector is None:
        engine = get_semantic_engine()
        if engine:
            try:
                from semantic_enhancements import SemanticFeedbackCollector
                _semantic_collector = SemanticFeedbackCollector(engine)
                logger.info(" Semantic Feedback collector loaded")
            except ImportError:
                _semantic_collector = False
        else:
            _semantic_collector = False
    return _semantic_collector if _semantic_collector is not False else None

def get_feedback_manager():
    """Lazy import feedback manager"""
    global _feedback_manager
    if _feedback_manager is None:
        try:
            from feedback_loop_system import FeedbackLoopManager
            _feedback_manager = FeedbackLoopManager()
            logger.info(" Feedback Loop manager loaded")
        except ImportError as e:
            logger.warning(f" Feedback system unavailable: {e}")
            _feedback_manager = False
    return _feedback_manager if _feedback_manager is not False else None

def get_field_extractor():
    """แก้ไข function นี้ในไฟล์ smart_mapper_main.py"""
    global _field_extractor
    if _field_extractor is None:
        try:
            from universal_field_extractor import SmartFieldExtractor
            _field_extractor = SmartFieldExtractor()  # จะมี learning layer อัตโนมัติ
            logger.info("Field extractor with learning loaded")
        except ImportError as e:
            logger.error(f"Field extractor unavailable: {e}")
            _field_extractor = False
    return _field_extractor if _field_extractor is not False else None

def get_schema_detector():
    """Lazy import schema detector"""
    global _schema_detector
    if _schema_detector is None:
        try:
            from enhanced_schema_detector import EnhancedSchemaDetector
            _schema_detector = EnhancedSchemaDetector()
            logger.info(" Schema detector loaded")
        except ImportError as e:
            logger.error(f" Schema detector unavailable: {e}")
            _schema_detector = False
    return _schema_detector if _schema_detector is not False else None

logger = logging.getLogger(__name__)

@dataclass
class ProcessingOptions:
    """Processing configuration options - ไม่เปลี่ยน"""
    interactive: bool = False
    enable_learning: bool = True
    enable_ai: bool = True
    force_schema: Optional[str] = None
    multiple_events: Optional[bool] = None
    quality_threshold: float = 60.0
    timeout_seconds: Optional[int] = None
    
    # Semantic options
    enable_semantic_learning: bool = False
    semantic_auto_apply_threshold: float = 0.8
    semantic_teaching_mode: bool = False

@dataclass
class ProcessingResult:
    """Comprehensive processing result - ไม่เปลี่ยน"""
    success: bool
    input_file: str
    output_files: List[str]
    processing_mode: str
    total_events: int
    successful_events: int
    failed_events: int
    overall_quality_score: float
    processing_time: float
    components_used: Dict[str, bool]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

class SmartMapperCore:
    """Core Smart Mapper with fixed architecture"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Initialize configuration
        self.config_manager = get_config_manager(config_file)
        self.config = get_config()
        
        # Initialize safe operation handler
        self.handler = SafeOperationHandler()
        self.performance_monitor = get_performance_monitor()
        
        # Components will be lazy loaded
        self._components_initialized = False

        self.auto_learning_integrator = AutoLearningIntegrator(self)

        logger.info("Smart Mapper Core initialized successfully")

    def _ensure_components_loaded(self):
        """Ensure components are loaded when needed"""
        if not self._components_initialized:
            # Force load essential components
            get_schema_detector()
            get_field_extractor()
            self._components_initialized = True

    def _simplified_count_events(self, data: Any) -> int:
        """Simplified O(n) event counting - แก้ performance issue"""
        
        # Quick type checks
        if isinstance(data, list) and len(data) > 1:
            # Validate first few items
            valid_items = 0
            for item in data[:3]:
                if isinstance(item, dict) and len(item) >= 2:
                    valid_items += 1
            
            if valid_items >= 2:
                logger.info(f" Found top-level array: {len(data)} events")
                return len(data)
            return 1
        
        if not isinstance(data, dict):
            return 1
        
        # Check common array patterns - O(1) lookups
        ARRAY_PATTERNS = [
            'incidents', 'resources', 'events', 'logs', 'alerts', 
            'items', 'results', 'records', 'entries', 'value', 'data'
        ]
        
        for pattern in ARRAY_PATTERNS:
            if pattern in data:
                value = data[pattern]
                if isinstance(value, list) and len(value) > 1:
                    # Quick validation
                    valid_count = sum(1 for item in value[:5] 
                                    if isinstance(item, dict) and len(item) >= 2)
                    if valid_count >= 2:
                        logger.info(f" Found events in '{pattern}': {len(value)} events")
                        return len(value)
        
        # Check nested patterns - limited depth
        NESTED_PATTERNS = [
            ('data', 'incidents', 'edges'),  # Trend Micro
            ('reply', 'alerts', 'data'),     # Cortex XDR
        ]
        
        for pattern_path in NESTED_PATTERNS:
            try:
                current = data
                for key in pattern_path:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        current = None
                        break
                
                if isinstance(current, list) and len(current) > 1:
                    logger.info(f" Found nested events: {len(current)} events")
                    return len(current)
            except Exception:
                continue
        
        return 1

    #-------------------------------------fix----------------------------------------------
    def process_file_with_learning(self, 
                                 input_file: str,
                                 output_file: Optional[str] = None,
                                 options: Optional[ProcessingOptions] = None) -> ProcessingResult:
        """Process file พร้อม Auto-Learning"""
        
        return self.auto_learning_integrator.enhanced_process_file(
            input_file, output_file, options
        )
    
    def get_learning_report(self) -> Dict[str, Any]:
        """ดูรายงานการเรียนรู้"""
        return self.auto_learning_integrator.get_intelligence_report()
    
    def teach_system(self, field_name: str, source_path: str, 
                    schema_type: str, example_data: Dict[str, Any]) -> bool:
        """สอนระบบด้วยตัวเอง"""
        return self.auto_learning_integrator.manual_teach_pattern(
            field_name, source_path, schema_type, example_data
        )
    #-------------------------------------fix----------------------------------------------

    def _memory_safe_process_multiple_events(self, 
                                           data: Dict[str, Any],
                                           events_info: Dict[str, Any],
                                           input_file: str,
                                           output_file: Optional[str],
                                           options: ProcessingOptions,
                                           start_time: float) -> ProcessingResult:
        """Memory-safe multiple events processing"""
        
        logger.info(f"Processing as multiple events: {events_info['total_events']} events")
        
        with self.performance_monitor.measure('multiple_events_processing'):
            # Extract individual events
            individual_events = self._extract_individual_events(data, events_info)
            
            if len(individual_events) <= 1:
                logger.warning("Expected multiple events but got ≤1, falling back to single event")
                return self._process_single_event(data, input_file, output_file, options, start_time)
            
            successful_events = []
            failed_events = []
            all_errors = []
            all_warnings = []
            
            # Process in batches to manage memory
            batch_size = min(50, len(individual_events))  # Process max 50 at a time
            
            for i in range(0, len(individual_events), batch_size):
                batch = individual_events[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} events)")
                
                for j, event_data in enumerate(batch, 1):
                    event_id = i + j
                    
                    try:
                        if not isinstance(event_data, dict):
                            failed_events.append(event_id)
                            all_errors.append(f"Event {event_id}: Invalid data type")
                            continue
                        
                        # Schema detection
                        schema_result = self._detect_schema(event_data, options)
                        
                        # Field extraction
                        extraction_result = self._extract_fields(event_data, schema_result['schema'], options)
                        
                        if extraction_result.success:
                            quality_score = self._analyze_quality(
                                extraction_result.data, 
                                schema_result['schema'], 
                                schema_result['confidence']
                            )
                            
                            if quality_score >= options.quality_threshold:
                                successful_events.append({
                                    'event_id': event_id,
                                    'data': extraction_result.data,
                                    'schema': schema_result['schema'],
                                    'confidence': schema_result['confidence'],
                                    'quality_score': quality_score
                                })
                            else:
                                failed_events.append(event_id)
                                all_warnings.append(f"Event {event_id}: Quality {quality_score:.1f}% below threshold")
                        else:
                            failed_events.append(event_id)
                            all_errors.append(f"Event {event_id}: {extraction_result.error.message}")
                            
                    except Exception as e:
                        failed_events.append(event_id)
                        all_errors.append(f"Event {event_id}: {str(e)}")
                
                # Force garbage collection after each batch
                if i > 0:  # Not first batch
                    gc.collect()
            
            logger.info(f"Processing complete: {len(successful_events)} successful, {len(failed_events)} failed")
            
            # Save outputs
            output_files = []
            if successful_events:
                save_results = self._save_multiple_outputs(
                    successful_events, input_file, output_file, events_info
                )
                output_files = save_results.get('files', [])
            
            # Calculate overall quality
            overall_quality = (sum(e['quality_score'] for e in successful_events) / len(successful_events)) if successful_events else 0.0
            
            processing_time = time.time() - start_time
            success = len(successful_events) > 0
            
            return ProcessingResult(
                success=success,
                input_file=input_file,
                output_files=output_files,
                processing_mode='multiple_events',
                total_events=len(individual_events),
                successful_events=len(successful_events),
                failed_events=len(failed_events),
                overall_quality_score=overall_quality,
                processing_time=processing_time,
                components_used=self._get_components_status(),
                errors=all_errors,
                warnings=all_warnings,
                metadata={
                    'events_info': events_info,
                    'successful_schemas': [e['schema'] for e in successful_events]
                }
            )

    def process_file(self, 
                    input_file: str,
                    output_file: Optional[str] = None,
                    options: Optional[ProcessingOptions] = None) -> ProcessingResult:
        """Process a single file with fixed architecture and integrated learning."""
        
        start_time = time.time()
        
        if options is None:
            options = ProcessingOptions()
        
        # Ensure components are loaded
        self._ensure_components_loaded()
        
        logger.info(f"Processing file: {input_file}")
        
        with self.performance_monitor.measure('total_processing'):
            # Step 1: Load and validate input file
            load_result = self._load_input_file(input_file)
            if not load_result.success:
                return ProcessingResult(
                    success=False,
                    input_file=input_file,
                    output_files=[],
                    processing_mode='failed',
                    total_events=0,
                    successful_events=0,
                    failed_events=0,
                    overall_quality_score=0.0,
                    processing_time=time.time() - start_time,
                    components_used=self._get_components_status(),
                    errors=[load_result.error.message],
                    warnings=[],
                    metadata={}
                )
            
            source_data = load_result.data
            
            # Step 2: Detect processing mode (simplified)
            events_count = self._simplified_count_events(source_data)
            
            if options.multiple_events is not None:
                # User override
                is_multiple = options.multiple_events
                events_count = events_count if is_multiple else 1
            else:
                # Auto detection
                is_multiple = events_count > 1
            
            events_info = {
                'multiple_events': is_multiple,
                'total_events': events_count,
                'detection_method': 'simplified_heuristic'
            }
            
            # Step 3: Process based on detected mode
            if is_multiple:
                return self._memory_safe_process_multiple_events(
                    source_data, events_info, input_file, output_file, options, start_time
                )
            else:
                return self._process_single_event(
                    source_data, input_file, output_file, options, start_time
                )

    def process_with_semantic_learning(self, 
                                     input_file: str,
                                     output_file: Optional[str] = None,
                                     options: Optional[ProcessingOptions] = None) -> ProcessingResult:
        """Process file with semantic learning - fixed lazy loading"""
        
        semantic_engine = get_semantic_engine()
        if not semantic_engine or not options.enable_semantic_learning:
            return self.process_file(input_file, output_file, options)
        
        start_time = time.time()
        logger.info(f"Processing with semantic learning: {input_file}")
        
        # Load input data
        load_result = self._load_input_file(input_file)
        if not load_result.success:
            return self._create_error_result(input_file, load_result.error.message)
        
        source_data = load_result.data
        
        # Enhanced schema detection with semantic context
        schema_result = self._detect_schema_with_semantic(source_data, options)
        schema_name = schema_result['schema']
        confidence = schema_result['confidence']
        
        # Enhanced field extraction with semantic predictions
        extraction_result = self._extract_fields_with_semantic(source_data, schema_name, options)
        
        if not extraction_result.success:
            return self._create_error_result(input_file, extraction_result.error.message)
        
        extracted_data = extraction_result.data
        
        # Interactive semantic learning session
        if options.interactive and options.semantic_teaching_mode:
            self._interactive_semantic_session(source_data, extracted_data, schema_name, input_file)
        
        # Quality analysis
        quality_score = self._analyze_quality(extracted_data, schema_name, confidence)
        
        # Save output
        if not output_file:
            output_file = self._generate_output_filename(input_file, 'semantic')
        
        save_result = self._save_output(extracted_data, output_file)
        
        processing_time = time.time() - start_time
        success = save_result.success and quality_score >= options.quality_threshold
        
        components_status = self._get_components_status()
        components_status['semantic_learning'] = semantic_engine is not None
        
        return ProcessingResult(
            success=success,
            input_file=input_file,
            output_files=[output_file] if save_result.success else [],
            processing_mode='semantic_learning',
            total_events=1,
            successful_events=1 if success else 0,
            failed_events=0 if success else 1,
            overall_quality_score=quality_score,
            processing_time=processing_time,
            components_used=components_status,
            errors=[] if success else ["Quality below threshold or save failed"],
            warnings=[],
            metadata={
                'schema': schema_name,
                'confidence': confidence,
                'semantic_predictions': extraction_result.metadata.get('semantic_predictions', 0)
            }
        )

    def get_semantic_statistics(self) -> Dict[str, Any]:
        """Get semantic learning statistics - fixed lazy loading"""
        
        semantic_engine = get_semantic_engine()
        if not semantic_engine:
            return {'error': 'Semantic learning not available'}
        
        try:
            stats = semantic_engine.get_learning_statistics()
            return {
                'semantic_available': True,
                'statistics': stats,
                'knowledge_base_size': len(semantic_engine.concepts),
                'total_learned_relationships': len(semantic_engine.field_relationships)
            }
        except Exception as e:
            return {'error': f'Could not get semantic statistics: {e}'}

    # All other methods remain the same as original...
    # [Include all original methods but with lazy loading fixes]
    
    def _detect_schema_with_semantic(self, data: Dict[str, Any], options: ProcessingOptions) -> Dict[str, Any]:
        """Enhanced schema detection with semantic context"""
        
        # Use existing schema detection
        basic_result = self._detect_schema(data, options)
        
        semantic_engine = get_semantic_engine()
        if not semantic_engine:
            return basic_result
        
        # Add semantic context analysis
        try:
            semantic_hints = semantic_engine.analyze_schema_context(data)
            
            if semantic_hints and semantic_hints.get('confidence', 0) > basic_result['confidence']:
                logger.info(f"Semantic hints improved schema detection: {semantic_hints}")
                return {
                    'schema': semantic_hints.get('schema', basic_result['schema']),
                    'confidence': semantic_hints.get('confidence', basic_result['confidence']),
                    'method': 'semantic_enhanced'
                }
        except Exception as e:
            logger.debug(f"Semantic schema analysis failed: {e}")
        
        return basic_result

    def _extract_fields_with_semantic(self, 
                                    data: Dict[str, Any], 
                                    schema: str, 
                                    options: ProcessingOptions) -> OperationResult:
        """Enhanced field extraction with semantic predictions"""
        
        # Use existing field extraction
        basic_result = self._extract_fields(data, schema, options)
        
        semantic_engine = get_semantic_engine()
        if not basic_result.success or not semantic_engine:
            return basic_result
        
        extracted_data = basic_result.data
        semantic_predictions = 0
        
        try:
            # Find missing critical fields
            missing_fields = self._find_missing_critical_fields(extracted_data)
            
            for field_name in missing_fields:
                # Use semantic engine to predict mapping
                predictions = semantic_engine.predict_field_mapping(
                    "", None, schema
                )
                
                for source_path, confidence in predictions:
                    if confidence > options.semantic_auto_apply_threshold:
                        value = SafeDataProcessor.safe_get_nested(data, source_path)
                        if value is not None:
                            extracted_data[field_name] = value
                            semantic_predictions += 1
                            logger.info(f"Semantic prediction: {field_name} → {source_path} ({confidence:.1%})")
                            break
            
            return OperationResult(
                success=True,
                data=extracted_data,
                metadata={'semantic_predictions': semantic_predictions}
            )
            
        except Exception as e:
            logger.warning(f"Semantic field extraction failed: {e}")
            return basic_result

    def _interactive_semantic_session(self, 
                                    source_data: Dict[str, Any],
                                    extracted_data: Dict[str, Any], 
                                    schema_name: str,
                                    input_file: str):
        """Interactive semantic learning session"""
        
        semantic_engine = get_semantic_engine()
        if not semantic_engine:
            return
        
        print(f"\n Semantic Learning Session")
        print("=" * 50)
        
        # Show current semantic knowledge
        stats = semantic_engine.get_learning_statistics()
        print(f" Current Knowledge:")
        print(f"   Concepts: {stats['total_concepts']}")
        print(f"   Relationships: {stats['total_relationships']}")
        
        # Ask if user wants to teach semantic patterns
        choice = input(f"\n Teach semantic patterns for better future predictions? (Y/n): ").strip().lower()
        if choice in ['n', 'no']:
            return
        
        # Get available fields from source data
        available_fields = self._get_all_field_paths(source_data)
        
        predictions_taught = 0
        max_teachings = 10
        
        for field_path in available_fields[:max_teachings]:
            # Get semantic predictions for this field
            predictions = semantic_engine.predict_field_mapping(
                field_path, None, schema_name
            )
            
            if predictions and predictions[0][1] > 0.6:
                target_field, confidence = predictions[0]
                value = SafeDataProcessor.safe_get_nested(source_data, field_path)
                
                print(f"\n Field: {field_path}")
                print(f"   Value: {str(value)[:100]}...")
                print(f"   Prediction: → {target_field} ({confidence:.1%} confidence)")
                
                choice = input("   Actions: (Y)es (N)o (T)each other (S)kip all: ").strip().lower()
                
                if choice == 'y':
                    # User confirms prediction
                    self._record_semantic_feedback(field_path, target_field, source_data, schema_name, 'confirmed')
                    predictions_taught += 1
                    print("    Confirmed and learned!")
                    
                elif choice == 't':
                    # User teaches different mapping
                    correct_field = input("   What should this map to? ").strip()
                    if correct_field:
                        self._record_semantic_feedback(field_path, correct_field, source_data, schema_name, 'taught')
                        predictions_taught += 1
                        print(f"    Learned: {field_path} → {correct_field}")
                
                elif choice == 's':
                    break
        
        # Show learning progress
        if predictions_taught > 0:
            new_stats = semantic_engine.get_learning_statistics()
            print(f"\n Learning Progress:")
            print(f"   Taught {predictions_taught} patterns")
            print(f"   Total relationships: {new_stats['total_relationships']}")
            print("    System will be smarter for future files!")

    def _record_semantic_feedback(self, source_path: str, target_field: str, 
                                source_data: Dict[str, Any], schema_type: str, action: str):
        """Record semantic feedback for learning"""
        
        semantic_collector = get_semantic_collector()
        if not semantic_collector:
            return
        
        # Create feedback record - need to import here to avoid circular import
        try:
            from feedback_loop_system import FeedbackRecord
            
            feedback = FeedbackRecord(
                field_name=target_field,
                schema_type=schema_type,
                corrected_source_path=source_path,
                file_source="semantic_session",
                user_action=f"semantic_{action}"
            )
            
            # Collect enhanced semantic feedback
            enhanced_feedback = semantic_collector.collect_semantic_feedback(feedback, source_data)
            
            # Save to feedback database if available
            feedback_manager = get_feedback_manager()
            if feedback_manager:
                feedback_manager.db.save_feedback(enhanced_feedback)
        except ImportError:
            logger.warning("Could not record semantic feedback - FeedbackRecord not available")

    def _find_missing_critical_fields(self, extracted_data: Dict[str, Any]) -> List[str]:
        """Find missing critical fields that need semantic prediction"""
        
        critical_fields = ['alert_name', 'severity', 'detected_time', 'contexts.hostname', 'log_source']
        missing_fields = []
        
        for field in critical_fields:
            value = SafeDataProcessor.safe_get_nested(extracted_data, field)
            if value is None or value == "":
                missing_fields.append(field)
        
        return missing_fields

    # Include all other original methods with appropriate fixes...
    # (All remaining methods from the original code with lazy loading)
    
    def _load_input_file(self, input_file: str) -> OperationResult:
        """Load and validate input file"""
        
        with self.performance_monitor.measure('file_loading'):
            # Read file safely
            read_result = SafeFileHandler.safe_read_json(
                input_file, 
                max_size_mb=self.config.performance.max_file_size_mb
            )
            
            if not read_result.success:
                return read_result
            
            # Validate data structure
            data = read_result.data
            if not isinstance(data, dict):
                return OperationResult(
                    success=False,
                    error=read_result.error
                )
            
            logger.info(f"File loaded successfully: {len(json.dumps(data))} bytes")
            return OperationResult(success=True, data=data)

    def _detect_schema(self, data: Dict[str, Any], options: ProcessingOptions) -> Dict[str, Any]:
        """Detect schema with fallback"""
        
        # Use forced schema if specified
        if options.force_schema:
            return {
                'schema': options.force_schema,
                'confidence': 100.0,
                'method': 'user_forced'
            }
        
        # Use schema detector if available
        schema_detector = get_schema_detector()
        if schema_detector:
            try:
                with self.performance_monitor.measure('schema_detection'):
                    schema, confidence, metadata = schema_detector.detect_schema_with_confidence(data)
                    
                    return {
                        'schema': schema,
                        'confidence': confidence,
                        'method': 'ai_detection',
                        'metadata': metadata
                    }
            except Exception as e:
                logger.warning(f"Schema detection failed: {e}")
        
        # Fallback to simple heuristic
        return {
            'schema': 'generic_security',
            'confidence': 50.0,
            'method': 'fallback_heuristic'
        }

    def _extract_fields(self, data: Dict[str, Any], schema: str, options: ProcessingOptions) -> OperationResult:
        """Extract fields using field extractor"""
        
        field_extractor = get_field_extractor()
        if not field_extractor:
            return OperationResult(
                success=False,
                error={
                    'message': 'Field extractor not available',
                    'severity': 'HIGH'
                }
            )
        
        try:
            with self.performance_monitor.measure('field_extraction'):
                extracted_fields = field_extractor.extract_all_fields(data, schema)
                
                # Convert extraction results to standard format
                result_data = self._convert_extraction_results(extracted_fields, data)
                
                # Validate minimum required fields
                required_fields = ['alert_name', 'severity', 'log_source']
                missing_required = [f for f in required_fields 
                                if f not in result_data or not result_data[f]]
                
                warnings = []
                if missing_required:
                    warnings.append(f"Missing required fields: {', '.join(missing_required)}")
                
                return OperationResult(
                    success=True,
                    data=result_data,
                    warnings=warnings,
                    metrics={
                        'fields_extracted': len([f for f in extracted_fields.values() if f.value is not None]),
                        'total_fields': len(extracted_fields),
                        'required_fields_missing': len(missing_required)
                    }
                )
                
        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return OperationResult(
                success=False,
                error={
                    'message': f'Field extraction failed: {str(e)}',
                    'severity': 'HIGH'
                }
            )

    def _process_single_event(self, 
                             data: Dict[str, Any],
                             input_file: str,
                             output_file: Optional[str],
                             options: ProcessingOptions,
                             start_time: float) -> ProcessingResult:
        """Process single event with full pipeline"""
        
        logger.info("Processing as single event")
        
        with self.performance_monitor.measure('single_event_processing'):
            # Step 1: Schema detection
            schema_result = self._detect_schema(data, options)
            schema_name = schema_result['schema']
            confidence = schema_result['confidence']
            
            # Step 2: Field extraction
            extraction_result = self._extract_fields(data, schema_name, options)
            
            if not extraction_result.success:
                return ProcessingResult(
                    success=False,
                    input_file=input_file,
                    output_files=[],
                    processing_mode='single_event',
                    total_events=1,
                    successful_events=0,
                    failed_events=1,
                    overall_quality_score=0.0,
                    processing_time=time.time() - start_time,
                    components_used=self._get_components_status(),
                    errors=[extraction_result.error.message],
                    warnings=[],
                    metadata={'schema': schema_name, 'confidence': confidence}
                )
            
            extracted_data = extraction_result.data
            
            # Step 3: Quality analysis
            quality_score = self._analyze_quality(extracted_data, schema_name, confidence)
            
            # Step 4: Save output
            if not output_file:
                output_file = self._generate_output_filename(input_file, 'single')
            
            save_result = self._save_output(extracted_data, output_file)
            
            # Step 5: Generate result
            processing_time = time.time() - start_time
            
            success = save_result.success and quality_score >= options.quality_threshold
            
            return ProcessingResult(
                success=success,
                input_file=input_file,
                output_files=[output_file] if save_result.success else [],
                processing_mode='single_event',
                total_events=1,
                successful_events=1 if success else 0,
                failed_events=0 if success else 1,
                overall_quality_score=quality_score,
                processing_time=processing_time,
                components_used=self._get_components_status(),
                errors=[] if success else ["Quality below threshold or save failed"],
                warnings=extraction_result.warnings if hasattr(extraction_result, 'warnings') else [],
                metadata={
                    'schema': schema_name,
                    'confidence': confidence,
                    'extraction_stats': self._get_extraction_stats(extracted_data)
                }
            )

    def _extract_individual_events(self, data: Dict[str, Any], events_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract individual events - simplified version"""
        
        logger.info(f"Extracting individual events from data structure")
        
        if not isinstance(data, dict) or not data:
            logger.warning("Input data is not a valid dictionary")
            return []
        
        # Standard Array Patterns - O(1) lookups
        STANDARD_PATTERNS = [
            'incidents', 'resources', 'events', 'logs', 'alerts', 
            'items', 'results', 'records', 'entries', 'data', 'entityType',
            'incident_id','agent_id','type','alert_type','schemaVersion',
            'data', # Added for trellix_epo_alerts.json
            'results' # Added for trellix_helix_events.json
        ]
        
        for pattern in STANDARD_PATTERNS:
            try:
                if pattern in data and isinstance(data[pattern], list) and len(data[pattern]) > 1:
                    events = []
                    
                    for i, item in enumerate(data[pattern]):
                        if isinstance(item, dict) and len(item) >= 2:
                            item_copy = item.copy()
                            item_copy['_event_index'] = i
                            events.append(item_copy)
                    
                    if len(events) > 1:
                        logger.info(f" Found {len(events)} events using standard pattern '{pattern}'")
                        return events
                        
            except Exception as e:
                logger.debug(f"Standard pattern '{pattern}' failed: {e}")
                continue
        
        # Nested Patterns - limited to important ones
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 1:
                # Quick validation
                valid_count = sum(1 for item in value[:5] 
                                if isinstance(item, dict) and len(item) >= 2)
                if valid_count >= 2:
                    logger.info(f" Found events in '{key}': {len(value)} events")
                    return value
        
        # Fallback: Single event
                logger.info("No multiple events found, treating as single event")
        
        # New logic for Cortex XDR
        if 'data' in data and 'incidents' in data['data'] and 'edges' in data['data']['incidents']:
            events = data['data']['incidents']['edges']
            if isinstance(events, list) and len(events) > 0:
                logger.info(f"Found {len(events)} events using Cortex XDR specific path")
                return [event['node'] for event in events if 'node' in event]

        if isinstance(data, dict) and len(data) >= 2:
            single_event = data.copy()
            single_event['_event_index'] = 0
            single_event['_source_type'] = 'single_event'
            return [single_event]
        else:
            logger.warning(" Data structure is not suitable for event processing")
            return []

    def _convert_extraction_results(self, extraction_results: Dict, original_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert extraction results to standard output format"""
        
        result = {
            'alert_name': None,
            'description': None,
            'incident_type': None,
            'device_type': None,
            'severity': None,
            'contexts': {
                'srp_ip': [],
                'hostname': None,
                'os': None,
                'user': None
            },
            'mitre': [
                {
                    'id': None,
                    'link': None,
                    'name_techique': None,
                    'name_subtechnique': None
                }
            ],
            'events': [
                {
                    'file_name': None,
                    'file_path': None,
                    'cmd_line': None,
                    'parent_type': None,
                    'event_type': None,
                    'hash': {
                        'sha256': None,
                        'sha1': None,
                        'md5': None
                    },
                    'src_ip': None,
                    'src_port': None,
                    'dst_ip': None,
                    'dst_port': None,
                    'domain': None
                }
            ],
            'mitigationStatus': {
                'info': {
                    'state': None,
                    'description': None
                },
                'status': None
            },
            'event_type': None,
            'detected_time': None,
            'log_source': None,
            'process_id': None,
            'rule_name': None,
            'url': None,
            'source_ip': None,
            'source_port': None,
            'src_geolocation': None,
            'src_interface_name': None,
            'destination_ip': None,
            'destination_port': None,
            'dst_geolocation': None,
            'dst_interface_name': None,
            'nat_ip': None,
            'protocols': None,
            'http_status': None,
            'session_id': None
        }
        
        for field_name, extraction_result in extraction_results.items():
            if extraction_result.value is not None:
                self._set_nested_result_value(result, field_name, extraction_result.value)
        
        # Ensure rawAlert is always present
        if 'rawAlert' not in result or not result['rawAlert']:
            result['rawAlert'] = json.dumps(original_data, ensure_ascii=False, indent=2)
        
        return result

    def _set_nested_result_value(self, result: Dict[str, Any], field_path: str, value: Any):
        """Set nested value in result dictionary safely"""
        
        try:
            keys = field_path.split('.')
            current_level = result
            for i, key in enumerate(keys):
                if i == len(keys) - 1:
                    current_level[key] = value
                else:
                    if key not in current_level or not isinstance(current_level[key], dict):
                        current_level[key] = {}
                    current_level = current_level[key]
            
        except Exception as e:
            logger.debug(f"Failed to set nested value for {field_path}: {e}")

    def _analyze_quality(self, extracted_data: Dict[str, Any], schema: str, confidence: float) -> float:
        """Analyze extraction quality"""
        
        # Simple quality scoring
        quality_factors = []
        
        # Schema confidence factor
        quality_factors.append(min(confidence, 100.0))
        
        # Field completeness factor
        required_fields = ['alert_name', 'severity', 'log_source']
        present_required = sum(1 for field in required_fields if extracted_data.get(field))
        completeness_score = (present_required / len(required_fields)) * 100
        quality_factors.append(completeness_score)
        
        # Data richness factor
        total_fields = len(self._flatten_dict(extracted_data))
        populated_fields = len([v for v in self._flatten_dict(extracted_data).values() if v is not None])
        richness_score = (populated_fields / total_fields) * 100 if total_fields > 0 else 0
        quality_factors.append(richness_score)
        
        # Calculate weighted average
        weights = [0.1, 0.5, 0.4]  # confidence, completeness, richness
        overall_quality = sum(score * weight for score, weight in zip(quality_factors, weights))
        
        return min(100.0, max(0.0, overall_quality))

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _save_output(self, data: Dict[str, Any], output_file: str) -> OperationResult:
        """Save single output file with clear path logging"""
        
        try:
            # Convert to absolute path
            output_path = Path(output_file).resolve()
            
            # Create output directory if not exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Log the exact path where file will be saved
            logger.info(f"Saving output to: {output_path}")
            print(f"Saving file to: {output_path}")
            
            # Save using SafeFileHandler
            result = SafeFileHandler.safe_write_json(str(output_path), data, indent=2, ensure_ascii=False)
            
            if result.success:
                # Verify file was actually created
                if output_path.exists():
                    file_size = output_path.stat().st_size
                    logger.info(f" File saved successfully: {output_path} ({file_size} bytes)")
                    print(f" File saved: {output_path.name} ({file_size:,} bytes)")
                    print(f" Full path: {output_path}")
                else:
                    logger.error(f" File was not created: {output_path}")
                    return OperationResult(success=False, error={'message': 'File was not created'})
            else:
                logger.error(f" Failed to save file: {result.error}")
                
            return result
            
        except Exception as e:
            logger.error(f" Save operation failed: {e}")
            return OperationResult(success=False, error={'message': f'Save failed: {str(e)}'})

    def _save_multiple_outputs(self, 
                            successful_events: List[Dict[str, Any]],
                            input_file: str,
                            output_file: Optional[str],
                            events_info: Dict[str, Any]) -> Dict[str, Any]:
        """Save multiple output files with clear path tracking"""
        
        try:
            # Determine output directory and file names
            input_path = Path(input_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if output_file:
                # Use specified output file
                batch_file = Path(output_file).resolve()
            else:
                # Create output in same directory as input
                output_dir = input_path.parent / "output"
                output_dir.mkdir(exist_ok=True)
                batch_file = output_dir / f"{input_path.stem}_batch_{timestamp}.json"
            
            print(f"\n Output Directory: {batch_file.parent}")
            print(f" Batch File: {batch_file.name}")
            
            output_files = []
            
            # Save batch file (all events combined)
            batch_data = {
                'metadata': {
                    'source_file': str(input_path),
                    'total_events': len(successful_events),
                    'processing_timestamp': datetime.now().isoformat(),
                    'events_info': events_info,
                    'output_location': str(batch_file)
                },
                'events': [event['data'] for event in successful_events]
            }
            
            logger.info(f"Saving batch file to: {batch_file}")
            batch_result = self._save_output(batch_data, str(batch_file))
            
            if batch_result.success:
                output_files.append(str(batch_file))
                print(f" Batch file saved: {batch_file}")
            else:
                print(f" Failed to save batch file: {batch_result.error}")
            
            # Save individual files (optional, for small batches)
            if len(successful_events) <= 5:
                individual_dir = batch_file.parent / f"{input_path.stem}_individual_{timestamp}"
                individual_dir.mkdir(exist_ok=True)
                
                print(f" Individual files directory: {individual_dir}")
                
                for event in successful_events:
                    event_file = individual_dir / f"event_{event['event_id']:03d}.json"
                    
                    # Add metadata to individual event
                    individual_data = {
                        'metadata': {
                            'source_file': str(input_path),
                            'event_id': event['event_id'],
                            'schema': event['schema'],
                            'confidence': event['confidence'],
                            'quality_score': event['quality_score'],
                            'processing_timestamp': datetime.now().isoformat()
                        },
                        'event': event['data']
                    }
                    
                    event_result = self._save_output(individual_data, str(event_file))
                    if event_result.success:
                        output_files.append(str(event_file))
                
                print(f" {len([f for f in output_files if 'individual' in f])} individual files saved")
            
            # Create summary file
            summary_file = batch_file.parent / f"{input_path.stem}_summary_{timestamp}.json"
            summary_data = {
                'processing_summary': {
                    'input_file': str(input_path),
                    'total_events_found': events_info.get('total_events', 0),
                    'successful_events': len(successful_events),
                    'failed_events': events_info.get('total_events', 0) - len(successful_events),
                    'processing_timestamp': datetime.now().isoformat(),
                    'schemas_detected': list(set(event['schema'] for event in successful_events)),
                    'average_confidence': sum(event['confidence'] for event in successful_events) / len(successful_events) if successful_events else 0,
                    'average_quality': sum(event['quality_score'] for event in successful_events) / len(successful_events) if successful_events else 0
                },
                'output_files': {
                    'batch_file': str(batch_file),
                    'individual_files': [f for f in output_files if 'individual' in f],
                    'total_files_created': len(output_files)
                }
            }
            
            summary_result = self._save_output(summary_data, str(summary_file))
            if summary_result.success:
                output_files.append(str(summary_file))
                print(f" Summary file saved: {summary_file}")
            
            # Print final summary
            print(f"\n Output Summary:")
            print(f"    Output location: {batch_file.parent}")
            print(f"    Files created: {len(output_files)}")
            print(f"    Main file: {batch_file.name}")
            
            return {
                'files': output_files, 
                'success': True,
                'output_directory': str(batch_file.parent),
                'main_file': str(batch_file)
            }
            
        except Exception as e:
            logger.error(f" Failed to save multiple outputs: {e}")
            print(f" Save error: {e}")
            return {
                'files': [], 
                'success': False, 
                'error': str(e),
                'output_directory': None
            }

    def _generate_output_filename(self, input_file: str, mode: str) -> str:
        """Generate output filename with clear directory structure"""
        
        input_path = Path(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_dir = input_path.parent / "Azurites"
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename
        output_file = output_dir / f"{input_path.stem}_{mode}_{timestamp}.json"
        
        logger.info(f"Generated output filename: {output_file}")
        return str(output_file)

    def _get_components_status(self) -> Dict[str, bool]:
        """Get status of all components"""
        return {
            'schema_detector': get_schema_detector() is not None,
            'field_extractor': get_field_extractor() is not None,
            'semantic_engine': get_semantic_engine() is not None,
            'feedback_manager': get_feedback_manager() is not None,
            'ai_enabled': self.config.ai.enabled,
            'config_loaded': True
        }

    def _create_error_result(self, input_file: str, error_message: str) -> ProcessingResult:
        """Create error result for failed processing"""
        return ProcessingResult(
            success=False,
            input_file=input_file,
            output_files=[],
            processing_mode='error',
            total_events=0,
            successful_events=0,
            failed_events=1,
            overall_quality_score=0.0,
            processing_time=0.0,
            components_used=self._get_components_status(),
            errors=[error_message],
            warnings=[],
            metadata={}
        )

    def _get_extraction_stats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flattened = self._flatten_dict(data)
        
        try:
            # Safe JSON serialization with fallback
            data_size_bytes = len(json.dumps(data, ensure_ascii=False, default=str))
        except Exception as e:
            logger.warning(f"Could not calculate data size: {e}")
            data_size_bytes = 0
        
        return {
            'total_fields': len(flattened),
            'populated_fields': len([v for v in flattened.values() if v is not None]),
            'empty_fields': len([v for v in flattened.values() if v is None]),
            'data_size_bytes': data_size_bytes
        }

    def _get_all_field_paths(self, data: Any, prefix: str = '', max_depth: int = 8) -> List[str]:
        """Get all possible field paths from nested data"""
        paths = []
        
        def extract_paths_recursive(obj: Any, current_prefix: str, depth: int = 0):
            if depth > max_depth:
                return
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    path = f"{current_prefix}.{key}" if current_prefix else key
                    paths.append(path)
                    
                    if isinstance(value, (dict, list)):
                        extract_paths_recursive(value, path, depth + 1)
            
            elif isinstance(obj, list) and obj:
                # Add array notation
                for i, item in enumerate(obj[:3]):  # Check first 3 items
                    array_path = f"{current_prefix}[{i}]"
                    paths.append(array_path)
                    
                    if isinstance(item, (dict, list)):
                        extract_paths_recursive(item, array_path, depth + 1)
        
        extract_paths_recursive(data, prefix)
        return paths

    def batch_process_with_learning(self, input_dir: str, pattern: str = "*.json") -> List[ProcessingResult]:
        """Batch process with clear output tracking and memory management"""
        
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return []
        
        # Create main output directory
        main_output_dir = input_path / "batch_output"
        main_output_dir.mkdir(exist_ok=True)
        
        print(f"\n Batch Output Directory: {main_output_dir.resolve()}")
        
        json_files = list(input_path.glob(pattern))
        if not json_files:
            logger.warning(f"No files found matching pattern: {pattern}")
            return []
        
        logger.info(f"Batch processing {len(json_files)} files")
        print(f" Processing {len(json_files)} files...")
        
        results = []
        options = ProcessingOptions(interactive=False)
        
        for i, json_file in enumerate(json_files, 1):
            print(f"\n Processing {i}/{len(json_files)}: {json_file.name}")
            
            try:
                # Set output file in batch directory
                output_file = main_output_dir / f"{json_file.stem}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                result = self.process_file(str(json_file), str(output_file), options)
                results.append(result)
                
                if result.success:
                    print(f" Success: {result.successful_events} events processed")
                    if result.output_files:
                        print(f"    Output: {Path(result.output_files[0]).name}")
                else:
                    print(f" Failed: {'; '.join(result.errors[:2])}")
                
                # Force cleanup after each file
                if i % 10 == 0:  # Every 10 files
                    gc.collect()
                    
            except Exception as e:
                print(f" Error: {e}")
                results.append(self._create_error_result(str(json_file), str(e)))
        
        # Create batch summary
        batch_summary_file = main_output_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        successful_results = [r for r in results if r.success]
        total_events = sum(r.total_events for r in results)
        successful_events = sum(r.successful_events for r in results)
        
        batch_summary = {
            'batch_processing_summary': {
                'input_directory': str(input_dir),
                'output_directory': str(main_output_dir),
                'processing_timestamp': datetime.now().isoformat(),
                'files_processed': len(results),
                'files_successful': len(successful_results),
                'files_failed': len(results) - len(successful_results),
                'total_events': total_events,
                'successful_events': successful_events,
                'overall_success_rate': f"{(successful_events/total_events*100):.1f}%" if total_events > 0 else "0%"
            },
            'file_results': [
                {
                    'input_file': result.input_file,
                    'success': result.success,
                    'events_processed': result.successful_events,
                    'quality_score': result.overall_quality_score,
                    'output_files': result.output_files
                }
                for result in results
            ]
        }

        # Save batch summary
        try:
            with open(batch_summary_file, 'w', encoding='utf-8') as f:
                json.dump(batch_summary, f, indent=2, ensure_ascii=False)
            print(f"\n Batch summary saved: {batch_summary_file}")
        except Exception as e:
            print(f" Failed to save batch summary: {e}")
        
        # Final summary
        print(f"\n Batch Processing Complete!")
        print(f"    Output directory: {main_output_dir}")
        print(f"    Files processed: {len(results)}")
        print(f"    Successful: {len(successful_results)}")
        print(f"    Total events: {total_events}")
        print(f"    Successful events: {successful_events}")
        
        return results

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary"""
        return self.performance_monitor.get_all_stats()

def main():
    """Main entry point with improved argument handling and lazy loading"""
    
    parser = argparse.ArgumentParser(
        description="Smart JSON Mapper - AI-Powered Security Data Transformation with Fixed Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # All original arguments remain the same...
    parser.add_argument('--input', help='Input JSON file')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--batch', '-b', help='Batch process all *.json files in a directory.')
    parser.add_argument('--interactive', '-i', action='store_true', help='Enable interactive mode')
    parser.add_argument('--multiple', '-m', action='store_true', help='Force multiple events mode')
    parser.add_argument('--single', '-s', action='store_true', help='Force single event mode')
    parser.add_argument('--force-schema', choices=['cortex_xdr', 'crowdstrike', 'trend_micro', 'fortigate', 'trellix_epo'], help='Force specific schema')
    parser.add_argument('--no-ai', action='store_true', help='Disable AI features')
    parser.add_argument('--quality-threshold', type=float, default=40.0, help='Minimum quality threshold (0-100)')
    parser.add_argument('--timeout', type=int, help='Processing timeout in seconds')
    parser.add_argument('--version', action='version', version='Smart Mapper 2.0 with Fixed Architecture')
    parser.add_argument('--performance', action='store_true', help='Show performance statistics')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--semantic-learning', action='store_true', help='Enable semantic learning for better field mapping')
    parser.add_argument('--semantic-teaching', action='store_true', help='Enable interactive semantic teaching mode')
    parser.add_argument('--semantic-auto-threshold', type=float, default=0.8, help='Confidence threshold for auto-applying semantic predictions')
    parser.add_argument('--semantic-stats', action = 'store_true', help = 'Show semantic learning statistics')
    parser.add_argument('--auto-learning', action='store_true', help='Enable auto-learning from successful extractions')
    parser.add_argument('--learning-report', action='store_true',help='Show learning statistics and report')
    parser.add_argument('--teach', nargs=3, metavar=('FIELD', 'PATH', 'SCHEMA'), help='Manually teach: field_name source_path schema_type')
    parser.add_argument('--show-learning', action='store_true', help='Show learning statistics')
    parser.add_argument('--reset-learning', action='store_true',help = 'Reset learning knowledge (careful!)')
                       
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize Smart Mapper with fixed architecture
        mapper = SmartMapperCore(config_file=args.config)
        
        # Show version and component status
        print("Smart JSON Mapper v2.0 - Fixed Architecture")
        print("=" * 80)
        
        components = mapper._get_components_status()
        for component, status in components.items():
            status_icon = "" if status else ""
            print(f"{status_icon} {component.replace('_', ' ').title()}")
        
        print()

        if args.show_learning:
            field_extractor = get_field_extractor()
            if field_extractor and hasattr(field_extractor, 'show_learning_stats'):
                field_extractor.show_learning_stats()
            else:
                print("Learning not available")
            return
        
        # รีเซ็ต learning
        if args.reset_learning:
            confirm = input("Reset all learning data? (yes/no): ")
            if confirm.lower() == 'yes':
                try:
                    os.remove("field_knowledge.pkl")
                    print("Learning data reset")
                except FileNotFoundError:
                    print("No learning data to reset")
            return

        # Show learning report
        if args.learning_report:
            report = mapper.get_learning_report()
            print(f"\n Auto-Learning Report:")
            print(f"=" * 40)
            print(f" Total patterns learned: {report['learning_statistics']['total_patterns']}")
            print(f" High confidence patterns: {report['learning_statistics']['high_confidence_patterns']}")
            print(f" Schemas learned: {report['learning_statistics']['schemas_learned']}")
            
            if report['next_steps']:
                print(f"\n Next Steps:")
                for step in report['next_steps']:
                    print(f"   • {step}")
            
            return

        # Manual teaching
        if args.teach:
            field_name, source_path, schema_type = args.teach
            if not args.input:
                print(" Need input file for teaching context")
                return
            
            # Load example data
            load_result = mapper._load_input_file(args.input)
            if load_result.success:
                success = mapper.teach_system(field_name, source_path, schema_type, load_result.data)
                if success:
                    print(f"✨ Taught system: {field_name} → {source_path} (schema: {schema_type})")
                else:
                    print(f" Failed to teach pattern")
            return

        # Prepare processing options
        options = ProcessingOptions(
            interactive=args.interactive,
            enable_learning=not args.no_ai,
            enable_ai=not args.no_ai,
            force_schema=args.force_schema,
            multiple_events=args.multiple if args.multiple else (False if args.single else None),
            quality_threshold=40.0 if args.semantic_learning else args.quality_threshold,
            timeout_seconds=args.timeout,
            enable_semantic_learning=args.semantic_learning,
            semantic_auto_apply_threshold=0.5 if args.semantic_learning else 0.8,
            semantic_teaching_mode=args.semantic_teaching if hasattr(args, 'semantic_teaching') else False
        )
        # Process with auto-learning
        if args.auto_learning:
            result = mapper.process_file_with_learning(args.input, args.output, options)

            # Show learning results
            if 'auto_learning' in result.metadata:
                learning = result.metadata['auto_learning']
                if learning['learned']:
                    print(f"\nAuto-Learning Results:")
                    print(f"  Learned {len(learning['patterns'])} new patterns")
                    for pattern in learning['patterns']:
                        print(f"      • {pattern['field']} → {pattern['source_path']} ({pattern['confidence']:.2f})")
                else:
                    print(f"\nAuto-Learning: No new patterns (quality: {learning.get('quality_score', 0):.1f}%)")
        else:
            # Normal processing
            result = mapper.process_file(args.input, args.output, options)

        # Provide feedback on low accuracy
        if result.overall_quality_score < 75.0:
            print("\n--- Feedback for Low Accuracy ---")
            print(f"Overall accuracy is {result.overall_quality_score:.2f}%, which is below the 75% threshold.")
            print("Suggestions for improvement:")
            print("- Check the schema detection confidence. If it's low, consider forcing a schema with --force-schema.")
            print("- Review the field mappings in the configuration to ensure they are correct for your data.")
            print("- Use the interactive mode (--interactive) to provide feedback and correct mappings.")
            print("---------------------------------")

        # Handle semantic statistics
        if args.semantic_stats:
            stats = mapper.get_semantic_statistics()
            print(f" Semantic Learning Statistics:")
            print(f"=" * 40)
            
            if 'error' in stats:
                print(f" {stats['error']}")
            else:
                print(f" Semantic learning available")
                s = stats['statistics']
                print(f" Concepts: {s['total_concepts']}")
                print(f" Relationships: {s['total_relationships']}")
                
                for concept, details in s['concept_details'].items():
                    print(f"\n   {concept}:")
                    print(f"     Patterns: {details['patterns_learned']}")
                    print(f"     Usage: {details['usage_count']}")
                    print(f"     Confidence: {details['confidence']:.1%}")
            return
        
        # Process based on arguments
        if args.batch:
            normalized_path = os.path.abspath(os.path.normpath(args.batch))
            results = mapper.batch_process_with_learning(normalized_path)
            
            # Summary
            successful = [r for r in results if r.success]
            print("Batch Processing Summary:")
            print(f"   Files processed: {len(results)}")
            print(f"   Successful: {len(successful)}")
            print(f"   Failed: {len(results) - len(successful)}")
            
        elif args.input:
            # Single file processing
            normalized_input_path = os.path.abspath(os.path.normpath(args.input))
            if options.enable_semantic_learning:
                result = mapper.process_with_semantic_learning(normalized_input_path, args.output, options)
                print(f" Semantic predictions: {result.metadata.get('semantic_predictions', 0)}")
            else:
                result = mapper.process_file(normalized_input_path, args.output, options)
            
            # Display result
            print(f" File: {Path(result.input_file).name}")
            print(f" Mode: {result.processing_mode}")
            print(f" Time: {result.processing_time:.2f}s")
            print(f" Quality: {result.overall_quality_score:.1f}%")
            print(f" Success: {result.success}")
            
            if result.success:
                print(f" Output files: {len(result.output_files)}")
                for output_file in result.output_files:
                    print(f"   • {Path(output_file).name}")
            
            if result.errors:
                print(f"\n Errors:")
                for error in result.errors:
                    print(f"   • {error}")
            
            if result.warnings:
                print(f"\n Warnings:")
                for warning in result.warnings:
                    print(f"   • {warning}")
        else:
            parser.print_help()
            return
        
        # Show performance statistics if requested
        if args.performance:
            print(f"\n Performance Statistics:")
            stats = mapper.get_performance_summary()
            for operation, metrics in stats.items():
                if metrics.get('count', 0) > 0:
                    print(f"   {operation}:")
                    print(f"     Count: {metrics['count']}")
                    print(f"     Avg Time: {metrics['avg_time']:.3f}s")
                    print(f"     Total Time: {metrics['total_time']:.3f}s")
        
        print("\n Processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()