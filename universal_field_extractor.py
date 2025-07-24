#!/usr/bin/env python3
"""
universal_field_extractor.py - Fixed AI-Powered Universal Field Extraction
แก้ไข: Memory management, over-complex logic, AI loading issues

Changes:
1. Fixed memory management in AI components
2. Simplified extraction strategies 
3. Better error handling and fallbacks
4. Removed circular import dependencies
"""

import re
import json
import time
import gc
import weakref
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging

from core_config import get_config
from safe_utils import (
    SafeOperationHandler, SafeDataProcessor, OperationType,
    safe_operation, OperationResult, PerformanceMonitor, get_performance_monitor
)
from field_learner import SmartFieldLearner
from learning_database import LearningDatabase

logger = logging.getLogger(__name__)

@dataclass
class FieldMapping:
    """Enhanced field mapping with learning capabilities - ไม่เปลี่ยน"""
    target_field: str
    source_patterns: List[str]
    alternative_patterns: List[str]
    semantic_keywords: List[str]
    data_type: str
    converter: Optional[str] = None
    required: bool = False
    confidence_score: float = 0.0
    usage_count: int = 0
    success_rate: float = 0.0
    learned_patterns: List[str] = field(default_factory=list)

@dataclass
class ExtractionResult:
    """Field extraction result with metadata - ไม่เปลี่ยน"""
    field_name: str
    value: Any
    source_path: str
    confidence: float
    method: str
    conversion_applied: bool = False
    issues: List[str] = field(default_factory=list)

class AISemanticMatcher:
    """AI-powered semantic field matching with fixed memory management"""
    
    def __init__(self):
        self.config = get_config()
        self.ai_enabled = self.config.ai.enabled
        self.semantic_model = None
        
        # Fixed caching system with limits
        self.embedding_cache = {}
        self.similarity_cache = {}
        self.preprocessed_cache = {}
        
        # Cache management
        self._cache_limit = 500  # Reduced from unlimited
        self._last_cleanup = datetime.now()
        self._cleanup_interval = timedelta(minutes=20)
        
        # Weighted similarity components
        self.field_weights = {
            'exact_match': 1.0,
            'semantic_match': 0.8,
            'syntactic_match': 0.6,
            'fuzzy_match': 0.4
        }
        
        if self.ai_enabled:
            self._initialize_ai_model()
    
    def _initialize_ai_model(self):
        """Initialize AI semantic model safely with better error handling"""
        try:
            # Check dependencies first
            missing_deps = []
            
            try:
                import sentence_transformers
            except ImportError:
                missing_deps.append("sentence-transformers")
            
            try:
                import sklearn
            except ImportError:
                missing_deps.append("scikit-learn")
            
            try:
                import numpy
            except ImportError:
                missing_deps.append("numpy")
            
            if missing_deps:
                logger.info(f"AI dependencies missing: {', '.join(missing_deps)}")
                self.ai_enabled = False
                return
            
            # Try to load model
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            
            try:
                self.semantic_model = SentenceTransformer('paraphrase-mpnet-base-v2')
                logger.info("AI semantic model initialized: paraphrase-mpnet-base-v2")
            except Exception as e:
                logger.warning(f"Failed to load AI model: {e}")
                self.ai_enabled = False
                return
            
            self.cosine_similarity = cosine_similarity
            
            # Test the model
            test_embedding = self.semantic_model.encode("test")
            if test_embedding is None:
                raise Exception("Model test failed")
            
            logger.info("AI semantic model initialized successfully")
            
        except ImportError as e:
            logger.warning(f"AI dependencies not available: {e}")
            self.ai_enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize AI model: {e}")
            self.ai_enabled = False

    def _cleanup_caches(self):
        """Clean up caches to prevent memory bloat"""
        current_time = datetime.now()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        # Clean embedding cache
        if len(self.embedding_cache) > self._cache_limit:
            keep_count = int(self._cache_limit * 0.7)
            cache_items = list(self.embedding_cache.items())
            self.embedding_cache = dict(cache_items[-keep_count:])
            logger.debug(f"Cleaned embedding cache: kept {keep_count} entries")
        
        # Clean similarity cache
        if len(self.similarity_cache) > self._cache_limit:
            keep_count = int(self._cache_limit * 0.7)
            cache_items = list(self.similarity_cache.items())
            self.similarity_cache = dict(cache_items[-keep_count:])
        
        # Clean preprocessed cache
        if len(self.preprocessed_cache) > self._cache_limit:
            keep_count = int(self._cache_limit * 0.7)
            cache_items = list(self.preprocessed_cache.items())
            self.preprocessed_cache = dict(cache_items[-keep_count:])
        
        self._last_cleanup = current_time
        
        # Force garbage collection
        gc.collect()
    
    def find_semantic_matches(self, target_field: str, available_fields: List[str]) -> List[Tuple[str, float]]:
        """Find semantically similar fields using AI with memory management"""
        
        # Cleanup caches periodically
        self._cleanup_caches()
        
        if not self.ai_enabled or not self.semantic_model:
            return self._fallback_semantic_matching(target_field, available_fields)
        
        # Limit available fields for performance
        limited_fields = available_fields[:50] if len(available_fields) > 50 else available_fields
        
        try:
            # Prepare target field for embedding
            target_context = self._enhance_field_context(target_field)
            
            # Get or compute embeddings
            target_embedding = self._get_embedding_safe(target_context)
            if target_embedding is None:
                return self._fallback_semantic_matching(target_field, limited_fields)
            
            field_embeddings = []
            valid_fields = []
            
            for field in limited_fields:
                field_context = self._enhance_field_context(field)
                embedding = self._get_embedding_safe(field_context)
                if embedding is not None:
                    field_embeddings.append(embedding)
                    valid_fields.append(field)
            
            if not field_embeddings:
                return self._fallback_semantic_matching(target_field, limited_fields)
            
            # Calculate similarities
            similarities = self.cosine_similarity([target_embedding], field_embeddings)[0]
            
            # Create results with confidence scores
            results = []
            for field, similarity in zip(valid_fields, similarities):
                if similarity >= self.config.ai.similarity_threshold:
                    results.append((field, float(similarity)))
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:10]  # Return top 10 matches
            
        except Exception as e:
            logger.debug(f"AI semantic matching failed: {e}")
            return self._fallback_semantic_matching(target_field, limited_fields)
    
    def _get_embedding_safe(self, text: str) -> Optional[Any]:
        """Get embedding with caching and error handling"""
        if not text or len(text) > 256:  # Limit text length
            return None
        
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            embedding = self.semantic_model.encode(text, show_progress_bar=False)
            
            # Cache management with limits
            if len(self.embedding_cache) < self._cache_limit:
                self.embedding_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            logger.debug(f"Embedding generation failed for '{text}': {e}")
            return None
    
    def _enhance_field_context(self, field_name: str) -> str:
        """Enhance field name with context for better semantic matching"""
        if field_name in self.preprocessed_cache:
            return self.preprocessed_cache[field_name]
        
        # Clean field name
        cleaned = re.sub(r'[^a-zA-Z0-9_\s]', ' ', field_name)
        cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)  # Split camelCase
        cleaned = cleaned.replace('_', ' ').strip()
        
        # Add domain context
        context_words = []
        
        # Security domain keywords
        if any(word in cleaned.lower() for word in ['alert', 'threat', 'security', 'incident']):
            context_words.append('security')
        
        # Network domain keywords
        if any(word in cleaned.lower() for word in ['ip', 'port', 'network', 'address']):
            context_words.append('network')
        
        # Identity domain keywords
        if any(word in cleaned.lower() for word in ['user', 'account', 'login', 'identity']):
            context_words.append('identity')
        
        # Time domain keywords
        if any(word in cleaned.lower() for word in ['time', 'date', 'timestamp', 'when']):
            context_words.append('temporal')
        
        # Combine context with field name
        enhanced = ' '.join(context_words + [cleaned])
        result = enhanced.lower()
        
        # Cache result with limits
        if len(self.preprocessed_cache) < self._cache_limit:
            self.preprocessed_cache[field_name] = result
        
        return result
    
    def _fallback_semantic_matching(self, target_field: str, available_fields: List[str]) -> List[Tuple[str, float]]:
        """Fallback semantic matching using pattern matching"""
        results = []
        target_lower = target_field.lower()
        
        # Extract keywords from target field
        target_keywords = set(re.findall(r'\w+', target_lower))
        
        for field in available_fields:
            field_lower = field.lower()
            field_keywords = set(re.findall(r'\w+', field_lower))
            
            # Calculate similarity based on keyword overlap
            common_keywords = target_keywords & field_keywords
            if common_keywords:
                # Jaccard similarity
                similarity = len(common_keywords) / len(target_keywords | field_keywords)
                
                # Boost for exact substring matches
                if target_lower in field_lower or field_lower in target_lower:
                    similarity += 0.3
                
                # Boost for exact matches
                if target_lower == field_lower:
                    similarity = 1.0
                
                if similarity >= 0.3:  # Minimum threshold
                    results.append((field, min(similarity, 1.0)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:10]

class SmartFieldExtractor:
    """Smart field extraction with simplified strategies and memory management"""
    
    def __init__(self):
        """Smart field extraction with simplified strategies and memory management"""
        self.config = get_config()
        self.semantic_matcher = AISemanticMatcher()
        self.performance_monitor = get_performance_monitor()
        self.field_mappings = self._initialize_field_mappings()
        
        # Simplified extraction strategies (reduced from 5 to 3)
        self.extraction_strategies = [
            self._exact_path_extraction,
            self._semantic_extraction,
            self._pattern_based_extraction
        ]
        
        # Learning data with limits
        self.db = LearningDatabase()
        self.field_learner = SmartFieldLearner(self.db)
        self.feedback_patterns = defaultdict(list)
        self._max_feedback_patterns = 100  # Limit feedback storage

        # --- THE REAL FIX: Load and apply learned knowledge on initialization ---
        self._load_and_apply_knowledge()

    def _load_and_apply_knowledge(self):
        """
        Loads knowledge from the database and immediately applies it to the
        current field mappings. This ensures that every instance of the extractor
        uses the most up-to-date learned patterns.
        """
        try:
            # 1. Load all learned patterns from the database
            # This now correctly calls the new function in field_learner.py
            learned_patterns = self.field_learner.load_all_knowledge()
            if not learned_patterns:
                logger.info("No prior knowledge found in the database.")
                return

            enhanced_count = 0
            # 2. Inject these patterns into the active field mappings
            for target_field, patterns in learned_patterns.items():
                if target_field in self.field_mappings:
                    mapping = self.field_mappings[target_field]
                    # Patterns are tuples of (source_path, confidence)
                    # We sort by confidence to use the best ones first
                    sorted_patterns = sorted(patterns, key=lambda x: x[1], reverse=True)
                    
                    for source_path, confidence in sorted_patterns:
                        if source_path not in mapping.learned_patterns:
                            # Add the learned path to the front of the list for high priority
                            mapping.learned_patterns.insert(0, source_path)
                            enhanced_count += 1
            
            if enhanced_count > 0:
                logger.info(f"Successfully loaded and applied {enhanced_count} learned patterns to the extractor.")
        except Exception as e:
            logger.error(f"Failed to load and apply knowledge: {e}")
    
    def _initialize_field_mappings(self) -> Dict[str, FieldMapping]:
        """Initialize comprehensive field mappings - ไม่เปลี่ยน"""
        mappings = {}
        
        # Core security fields
        mappings['alert_name'] = FieldMapping(
            target_field='alert_name',
            source_patterns=['reply.alerts.data.name', 'name', 'alert_name', 'event_name', 'title'],
            alternative_patterns=['logdesc', 'description', 'summary', 'subject'],
            semantic_keywords=['alert', 'event', 'incident', 'notification', 'message', 'rule'],
            data_type='str',
            required=True
        )

        mappings['description'] = FieldMapping(
            target_field='description',
            source_patterns=['reply.alerts.data.description', 'description'],
            alternative_patterns=['description','alert','msg'],
            semantic_keywords=['alert','msg', 'details'],
            data_type='str',
            required=True    
        )

        mappings['device_type'] = FieldMapping(
            target_field='device_type',
            source_patterns=['reply.alerts.data.endpoint_type', 'device_type','agent_os_type','platform_name','Host_OS'],
            alternative_patterns=['device_type','Host_OS', 'os_type'],
            semantic_keywords=['agent_os_sub_type','device_type','agent_os_type','platform_name','OS', 'platform'],
            data_type='str',
        )
         
        mappings['severity'] = FieldMapping(
            target_field='severity',
            source_patterns=['reply.alerts.data.severity', 'severity', 'priority', 'level','crlevel'],
            alternative_patterns=['risk_level', 'criticality', 'impact'],
            semantic_keywords=['severity', 'priority', 'critical', 'risk', 'level','crlevel', 'threat_level'],
            data_type='str',
            converter='normalize_severity',
            required=True
        )
        
        mappings['detected_time'] = FieldMapping(
            target_field='detected_time',
            source_patterns=['reply.alerts.data.detection_timestamp', 'timestamp', 'time', 'detected_time', 'event_time'],
            alternative_patterns=['created_at', 'occurrence_time', 'eventtime', 'date'],
            semantic_keywords=['time', 'timestamp', 'when', 'detected', 'occurred', 'detection_time'],
            data_type='str',
            converter='normalize_timestamp',
            required=False
        )
        
        mappings['log_source'] = FieldMapping(
            target_field='log_source',
            source_patterns=['reply.alerts.data.source', 'source', 'log_source', 'data_source'],
            alternative_patterns=['product', 'vendor', 'system', 'devname'],
            semantic_keywords=['source', 'origin', 'system', 'product', 'vendor', 'log_source'],
            data_type='str',
            required=True
        )
        
        mappings['incident_type'] = FieldMapping(
            target_field='incident_type',
            source_patterns=['reply.alerts.data.category', 'incident_type', 'event_type', 'type'],
            alternative_patterns=['category', 'classification', 'subtype'],
            semantic_keywords=['type', 'category', 'class', 'kind', 'incident', 'alert_category'],
            data_type='str',
            required=True
        )
        
        # Context fields
        mappings['contexts.hostname'] = FieldMapping(
            target_field='contexts.hostname',
            source_patterns=['reply.alerts.data.host_name', 'hostname', 'host_name', 'host','guid'],
            alternative_patterns=['computer_name', 'machine_name', 'device_name'],
            semantic_keywords=['host', 'computer', 'machine', 'device', 'endpoint','guid', 'hostname'],
            data_type='str',
            required=True
        )
        
        mappings['contexts.src_ip'] = FieldMapping(
            target_field='contexts.src_ip',
            source_patterns=['reply.alerts.data.actor_process_image_path', 'src_ip', 'source_ip', 'srcip','ips'],
            alternative_patterns=['host_ip', 'client_ip', 'local_ip'],
            semantic_keywords=['ip', 'address', 'source', 'client', 'host','ips', 'source_address'],
            data_type='list',
            converter='to_array',
            required=False
        )
        
        mappings['contexts.user'] = FieldMapping(
            target_field='contexts.user',
            source_patterns=['reply.alerts.data.actor_effective_username', 'user', 'username', 'user_name','name'],
            alternative_patterns=['account', 'login', 'principal'],
            semantic_keywords=['user', 'account', 'login', 'person', 'identity','name', 'username'],
            data_type='str',
            required=False
        )
        
        # mappings['mitre.id'] = FieldMapping(
        #     target_field='mitre.id',
        #     source_patterns=[],
        #     alternative_patterns=[],
        #     semantic_keywords=[],
        #     data_type='str',
        #     required=False
        # )
        # mappings['mitre.link'] = FieldMapping(
        #     target_field='mitre.link',
        #     source_patterns=[],
        #     alternative_patterns=[],
        #     semantic_keywords=[],
        #     data_type='str',
        #     required=False
        # )
        # mappings['mitre.name_techique'] = FieldMapping(
        #     target_field='mitre.name_techique',
        #     source_patterns=[],
        #     alternative_patterns=[],
        #     semantic_keywords=[],
        #     data_type='str',
        #     required=False
        # )
        # mappings['mitre.name_subtechnique'] = FieldMapping(
        #     target_field='mitre.name_subtechnique',
        #     source_patterns=[],
        #     alternative_patterns=[],
        #     semantic_keywords=[],
        #     data_type='str',
        #     required=False
        # )
        
        # Network fields
        mappings['source_ip'] = FieldMapping(
            target_field='source_ip',
            source_patterns=['source_ip', 'src_ip', 'srcip', 'external_ip'],
            alternative_patterns=['client_ip', 'host_ip', 'local_ip'],
            semantic_keywords=['source', 'ip', 'address', 'client', 'origin'],
            data_type='str',
            required=False
        )
        
        mappings['destination_ip'] = FieldMapping(
            target_field='destination_ip',
            source_patterns=['destination_ip', 'dest_ip', 'dst_ip', 'dstip'],
            alternative_patterns=['target_ip', 'server_ip', 'remote_ip'],
            semantic_keywords=['destination', 'target', 'server', 'remote', 'ip'],
            data_type='str',
            required=False
        )
        
        mappings['source_port'] = FieldMapping(
            target_field='source_port',
            source_patterns=['source_port', 'src_port', 'srcport'],
            alternative_patterns=['client_port', 'local_port'],
            semantic_keywords=['source', 'port', 'client', 'local'],
            data_type='int',
            converter='to_int',
            required=False
        )
        
        mappings['destination_port'] = FieldMapping(
            target_field='destination_port',
            source_patterns=['destination_port', 'dest_port', 'dst_port', 'dstport'],
            alternative_patterns=['target_port', 'server_port', 'remote_port'],
            semantic_keywords=['destination', 'target', 'server', 'port'],
            data_type='int',
            converter='to_int',
            required=False
        )
        
        mappings['protocols'] = FieldMapping(
            target_field='protocols',
            source_patterns=['protocol', 'protocols', 'proto'],
            alternative_patterns=['transport', 'service', 'ip_protocol'],
            semantic_keywords=['protocol', 'transport', 'service', 'communication'],
            data_type='str',
            required=False
        )
        
        # File and process fields
        mappings['events.file_name'] = FieldMapping(
            target_field='events.file_name',
            source_patterns=['file_name', 'filename', 'executable','Parent_Process_Name'],
            alternative_patterns=['image_name', 'binary_name', 'process_name'],
            semantic_keywords=['file', 'executable', 'binary', 'program', 'process','Parent_Process_Name'],
            data_type='str',
            required=False
        )
        
        mappings['events.file_path'] = FieldMapping(
            target_field='events.file_path',
            source_patterns=['file_path', 'filepath', 'full_path', 'Process_Path', 'reply.alerts.data.causality_actor.process_image_path', 'causality_actor.process_image_path'],
            alternative_patterns=['image_path', 'executable_path', 'binary_path'],
            semantic_keywords=['path', 'location', 'directory', 'file', 'Process_Path'],
            data_type='str',
            converter='clean_file_path',
            required=False
        )
        
        mappings['events.cmd_line'] = FieldMapping(
            target_field='events.cmd_line',
            source_patterns=['cmd_line', 'command_line', 'cmdline','CommandLine'],
            alternative_patterns=['command', 'arguments', 'parameters','Parent_Process_CmdLine'],
            semantic_keywords=['command', 'cmd', 'arguments', 'parameters','CommandLine'],
            data_type='str',
            required=False
        )
        
        # Hash fields
        mappings['events.hash.md5'] = FieldMapping(
            target_field='events.hash.md5',
            source_patterns=['md5', 'file_md5', 'hash_md5','Process_Md5'],
            alternative_patterns=['hash'],
            semantic_keywords=['md5', 'hash', 'checksum', 'signature'],
            data_type='str',
            converter='normalize_hash',
            required=False
        )
        
        mappings['events.hash.sha256'] = FieldMapping(
            target_field='events.hash.sha256',
            source_patterns=['sha256', 'file_sha256', 'hash_sha256','Parent_Process_Sha256'],
            alternative_patterns=['hash'],
            semantic_keywords=['sha256', 'hash', 'checksum', 'signature'],
            data_type='str',
            converter='normalize_hash',
            required=False
        )
        
        return mappings
    
    def extract_all_fields(self, data: Dict[str, Any], schema_hint: Optional[str] = None) -> Dict[str, ExtractionResult]:
        """Extract fields พร้อม learning (แทนที่ method เดิม)"""
        
        # 1. Extract ปกติ (โค้ดเดิม)
        results = self._extract_fields_original(data, schema_hint)
        
        # 2. เรียนรู้จากข้อมูลที่พบ
        self._learn_from_extraction(data, results, schema_hint)
        
        # 3. ปรับปรุงผลลัพธ์ด้วย learning
        enhanced_results = self._enhance_with_learning(data, results)
        
        # 4. บันทึก knowledge (ทุก 10 ครั้ง)
        if hasattr(self, '_extraction_count'):
            self._extraction_count += 1
        else:
            self._extraction_count = 1
            
        if self._extraction_count % 10 == 0:
            self.field_learner.save_knowledge()
        
        return enhanced_results
    def _extract_fields_original(self, data: Dict[str, Any], schema_hint: Optional[str] = None) -> Dict[str, ExtractionResult]:
        """โค้ด extract เดิม (copy จาก method เดิม)"""
        try:
            with self.performance_monitor.measure('field_extraction_total'):
                results = {}
                available_fields = self._get_all_field_paths(data, max_paths=200)
                
                for field_name, mapping in self.field_mappings.items():
                    with self.performance_monitor.measure(f'extract_{field_name}'):
                        result = self._extract_single_field(data, mapping, available_fields, schema_hint)
                        if result:
                            results[field_name] = result
                
                results = self._post_process_results(results, data)
                return results
        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {}
    
    def _learn_from_extraction(self, data: Dict[str, Any], results: Dict[str, ExtractionResult], schema_hint: str):
        """เรียนรู้จากการ extract"""
        
        def learn_recursive(obj: Any, prefix: str = '', depth: int = 0):
            if depth > 4:  # จำกัดความลึก
                return
                
            if isinstance(obj, dict):
                for key, value in obj.items():
                    field_path = f"{prefix}.{key}" if prefix else key
                    
                    # เรียนรู้ field นี้
                    if value is not None:
                        self.field_learner.learn_field(field_path, value, schema_hint or 'unknown')
                    
                    # เรียนรู้ nested
                    if isinstance(value, (dict, list)) and depth < 3:
                        learn_recursive(value, field_path, depth + 1)
                        
            elif isinstance(obj, list) and obj:
                # เรียนรู้จากรายการแรก
                learn_recursive(obj[0], f"{prefix}[]", depth + 1)
        
        try:
            learn_recursive(data)
        except Exception as e:
            logger.debug(f"Learning failed: {e}")
    
    def _enhance_with_learning(self, data: Dict[str, Any], results: Dict[str, ExtractionResult]) -> Dict[str, ExtractionResult]:
        """ปรับปรุงผลลัพธ์ด้วย learning"""
        
        enhanced = results.copy()
        
        # หา fields ที่ยังหายไป
        target_fields = ['alert_name', 'severity', 'detected_time', 'contexts.hostname', 'source_ip']
        missing_fields = []
        
        for field in target_fields:
            if field not in results or not results[field].value:
                missing_fields.append(field)
        
        if not missing_fields:
            return enhanced
        
        # ใช้ learning แนะนำ mapping
        available_fields = self._get_all_field_paths(data, max_paths=100)
        
        for missing_field in missing_fields:
            best_suggestion = None
            best_confidence = 0.0
            
            for available_field in available_fields:
                suggestions = self.field_learner.suggest_mapping(available_field, [missing_field])
                
                for suggested_target, confidence in suggestions:
                    if suggested_target == missing_field and confidence > best_confidence and confidence > 0.6:
                        value = self._safe_get_nested(data, available_field)
                        if value is not None:
                            best_suggestion = (available_field, value, confidence)
                            best_confidence = confidence
            
            # ใช้ suggestion ที่ดีที่สุด
            if best_suggestion:
                field_path, value, confidence = best_suggestion
                
                enhanced[missing_field] = ExtractionResult(
                    field_name=missing_field,
                    value=value,
                    source_path=field_path,
                    confidence=confidence,
                    method='learning_suggestion'
                )
                
                logger.info(f"Learning แนะนำ: {missing_field} ← {field_path} ({confidence:.2f})")
        
        return enhanced

    def _safe_get_nested(self, data: Dict[str, Any], path: str) -> Any:
        """ดึงข้อมูล nested อย่างปลอดภัย"""
        try:
            current = data
            for part in path.split('.'):
                if '[]' in part:
                    key = part.replace('[]', '')
                    if key in current and isinstance(current[key], list) and current[key]:
                        current = current[key][0]
                    else:
                        return None
                else:
                    if part in current:
                        current = current[part]
                    else:
                        return None
            return current
        except:
            return None
    def show_learning_stats(self):
        """แสดงสถิติการเรียนรู้"""
        stats = self.field_learner.get_stats()
        print(f"\n Learning Statistics:")
        print(f"   Total fields learned: {stats['total_fields']}")
        print(f"   High confidence fields: {stats['high_confidence']}")
        print(f"   Semantic distribution: {stats['semantic_distribution']}")
            
    def _extract_single_field(self, 
                             data: Dict[str, Any],
                             mapping: FieldMapping,
                             available_fields: List[str],
                             schema_hint: Optional[str] = None) -> Optional[ExtractionResult]:
        """Extract single field using simplified strategies"""
        
        # Try extraction strategies in order
        for i, strategy in enumerate(self.extraction_strategies):
            try:
                result = strategy(data, mapping, available_fields, schema_hint)
                if result and result.value is not None:
                    result.method = f"strategy_{i+1}_{strategy.__name__}"
                    
                    # Apply data conversion if needed
                    if mapping.converter:
                        converted_value = self._apply_conversion(result.value, mapping.converter)
                        if converted_value != result.value:
                            result.value = converted_value
                            result.conversion_applied = True
                    
                    # Update mapping statistics
                    mapping.usage_count += 1
                    mapping.confidence_score = (mapping.confidence_score + result.confidence) / 2
                    
                    return result
                    
            except Exception as e:
                logger.debug(f"Strategy {strategy.__name__} failed for {mapping.target_field}: {e}")
                continue
        
        # No value found
        return ExtractionResult(
            field_name=mapping.target_field,
            value=None,
            source_path='not_found',
            confidence=0.0,
            method='no_strategy_succeeded',
            issues=['Field not found in source data']
        )
    
    def _exact_path_extraction(self, 
                              data: Dict[str, Any],
                              mapping: FieldMapping,
                              available_fields: List[str],
                              schema_hint: Optional[str] = None) -> Optional[ExtractionResult]:
        """Strategy 1: Exact path matching"""
        
        # Try learned patterns first (if any)
        for pattern in mapping.learned_patterns[:5]:  # Limit to top 5 learned patterns
            value = SafeDataProcessor.safe_get_nested(data, pattern)
            if value is not None:
                return ExtractionResult(
                    field_name=mapping.target_field,
                    value=value,
                    source_path=pattern,
                    confidence=0.98,  # High confidence for learned patterns
                    method='learned_pattern'
                )
        
        # Try source patterns
        for pattern in mapping.source_patterns:
            value = SafeDataProcessor.safe_get_nested(data, pattern)
            if value is not None:
                return ExtractionResult(
                    field_name=mapping.target_field,
                    value=value,
                    source_path=pattern,
                    confidence=0.95,
                    method='exact_path'
                )
        
        # Try alternative patterns
        for pattern in mapping.alternative_patterns:
            value = SafeDataProcessor.safe_get_nested(data, pattern)
            if value is not None:
                return ExtractionResult(
                    field_name=mapping.target_field,
                    value=value,
                    source_path=pattern,
                    confidence=0.85,
                    method='alternative_path'
                )
        
        return None
    
    def _semantic_extraction(self, 
                           data: Dict[str, Any],
                           mapping: FieldMapping,
                           available_fields: List[str],
                           schema_hint: Optional[str] = None) -> Optional[ExtractionResult]:
        """Strategy 2: AI semantic matching"""
        
        # Create target context for semantic matching
        target_context = f"{mapping.target_field} {' '.join(mapping.semantic_keywords)}"
        
        # Find semantic matches
        semantic_matches = self.semantic_matcher.find_semantic_matches(target_context, available_fields)
        
        for field_path, similarity in semantic_matches:
            value = SafeDataProcessor.safe_get_nested(data, field_path)
            if value is not None:
                return ExtractionResult(
                    field_name=mapping.target_field,
                    value=value,
                    source_path=field_path,
                    confidence=float(similarity),
                    method='semantic_matching'
                )
        
        return None
    
    def _pattern_based_extraction(self, 
                                 data: Dict[str, Any],
                                 mapping: FieldMapping,
                                 available_fields: List[str],
                                 schema_hint: Optional[str] = None) -> Optional[ExtractionResult]:
        """Strategy 3: Simplified pattern-based fuzzy matching"""
        
        target_keywords = set(mapping.semantic_keywords + [mapping.target_field.split('.')[-1]])
        
        best_match = None
        best_score = 0.0
        
        # Limit fields to check for performance
        limited_fields = available_fields[:100] if len(available_fields) > 100 else available_fields
        
        for field_path in limited_fields:
            field_name = field_path.split('.')[-1].lower()
            
            # Calculate simplified pattern match score
            score = 0.0
            
            for keyword in target_keywords:
                keyword_lower = keyword.lower()
                
                # Exact match
                if keyword_lower == field_name:
                    score += 1.0
                # Substring match
                elif keyword_lower in field_name or field_name in keyword_lower:
                    score += 0.6
            
            # Normalize score
            if target_keywords:
                score = score / len(target_keywords)
            
            if score > best_score and score >= 0.4:  # Minimum threshold
                value = SafeDataProcessor.safe_get_nested(data, field_path)
                if value is not None:
                    best_match = ExtractionResult(
                        field_name=mapping.target_field,
                        value=value,
                        source_path=field_path,
                        confidence=score,
                        method='pattern_matching'
                    )
                    best_score = score
        
        return best_match
    
    def _get_all_field_paths(self, data: Any, prefix: str = '', max_depth: int = 6, max_paths: int = 500) -> List[str]:
        """Get all possible field paths from nested data with limits"""
        paths = []
        
        def extract_paths_recursive(obj: Any, current_prefix: str, depth: int = 0):
            if depth > max_depth or len(paths) >= max_paths:
                return
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if len(paths) >= max_paths:
                        break
                    
                    path = f"{current_prefix}.{key}" if current_prefix else key
                    paths.append(path)
                    
                    if isinstance(value, (dict, list)) and depth < max_depth:
                        extract_paths_recursive(value, path, depth + 1)
            
            elif isinstance(obj, list) and obj and len(paths) < max_paths:
                # Add array notation for first few items only
                for i, item in enumerate(obj[:3]):  # Check first 3 items
                    if len(paths) >= max_paths:
                        break
                    
                    array_path = f"{current_prefix}[{i}]"
                    paths.append(array_path)
                    
                    if isinstance(item, (dict, list)) and depth < max_depth:
                        extract_paths_recursive(item, array_path, depth + 1)
        
        extract_paths_recursive(data, prefix)
        return paths[:max_paths]  # Ensure we don't exceed limit
    
    def _apply_conversion(self, value: Any, converter: str) -> Any:
        """Apply data conversion safely - ไม่เปลี่ยน"""
        converters = {
            'to_array': lambda x: [x] if not isinstance(x, list) else x,
            'to_int': lambda x: int(x) if str(x).isdigit() else None,
            'to_str': lambda x: str(x),
            'normalize_severity': self._normalize_severity,
            'normalize_timestamp': self._normalize_timestamp,
            'clean_file_path': self._clean_file_path,
            'normalize_hash': self._normalize_hash
        }
        
        if converter in converters:
            try:
                return converters[converter](value)
            except Exception as e:
                logger.debug(f"Conversion {converter} failed for value {value}: {e}")
                return value
        
        return value
    
    def _normalize_severity(self, value: Any) -> str:
        """Normalize severity values - ไม่เปลี่ยน"""
        if not value:
            return "Medium"
        
        severity_map = {
            'low': 'Low', 'info': 'Low', 'notice': 'Low',
            'medium': 'Medium', 'warning': 'Medium', 'warn': 'Medium',
            'high': 'High', 'error': 'High', 'severe': 'High',
            'critical': 'Critical', 'crit': 'Critical'
        }
        
        return severity_map.get(str(value).lower(), str(value).title())
    
    def _normalize_timestamp(self, value: Any) -> Optional[str]:
        """Normalize timestamp to ISO format - ไม่เปลี่ยน"""
        if not value:
            return None
        
        try:
            if isinstance(value, (int, float)):
                # Unix timestamp
                from datetime import datetime
                dt = datetime.fromtimestamp(value)
                return dt.isoformat() + 'Z'
            elif isinstance(value, str):
                # Try to parse as ISO
                if 'T' in value:
                    return value
                else:
                    return value  # Return as-is for now
            else:
                return str(value)
        except Exception:
            return str(value)
    
    def _clean_file_path(self, value: Any) -> Optional[str]:
        """Clean file path - ไม่เปลี่ยน"""
        if not value:
            return None
        
        path = str(value).strip()
        path = path.replace('\\', '\\')  # Fix double backslashes
        path = path.strip('"\'')  # Remove quotes
        
        return path if path else None
    
    def _normalize_hash(self, value: Any) -> Optional[str]:
        """Normalize hash values - ไม่เปลี่ยน"""
        if not value:
            return None
        
        hash_str = str(value).strip().lower()
        
        # Remove prefixes
        if ':' in hash_str:
            hash_str = hash_str.split(':')[-1]
        
        # Validate hash format
        if len(hash_str) in [32, 40, 64] and all(c in '0123456789abcdef' for c in hash_str):
            return hash_str
        
        return None
    
    def _post_process_results(self, results: Dict[str, ExtractionResult], original_data: Dict[str, Any]) -> Dict[str, ExtractionResult]:
        """Post-process extraction results"""
        
        # Apply business logic and validation
        processed = {}
        
        for field_name, result in results.items():
            if result.value is not None:
                # Validate result based on field type
                if self._validate_extraction_result(result):
                    processed[field_name] = result
                else:
                    result.issues.append("Failed validation")
                    processed[field_name] = result
        
        # Add synthetic fields if needed
        processed = self._add_synthetic_fields(processed, original_data)
        
        return processed
    
    def _validate_extraction_result(self, result: ExtractionResult) -> bool:
        """Validate extraction result"""
        if result.value is None:
            return False
        
        # Field-specific validation
        if 'ip' in result.field_name.lower():
            return self._is_valid_ip(str(result.value))
        elif 'port' in result.field_name.lower():
            try:
                port = int(result.value)
                return 1 <= port <= 65535
            except:
                return False
        elif 'hash' in result.field_name.lower():
            return self._is_valid_hash(str(result.value))
        
        return True
    
    def _is_valid_ip(self, ip_str: str) -> bool:
        """Validate IP address format"""
        import re
        ip_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
        return bool(re.match(ip_pattern, ip_str))
    
    def _is_valid_hash(self, hash_str: str) -> bool:
        """Validate hash format"""
        return (len(hash_str) in [32, 40, 64] and 
                all(c in '0123456789abcdef' for c in hash_str.lower()))
    
    def _add_synthetic_fields(self, results: Dict[str, ExtractionResult], original_data: Dict[str, Any]) -> Dict[str, ExtractionResult]:
        """Add synthetic/computed fields"""
        
        # Add rawAlert if not present
        if 'rawAlert' not in results:
            results['rawAlert'] = ExtractionResult(
                field_name='rawAlert',
                value=json.dumps(original_data, ensure_ascii=False, indent=2),
                source_path='synthetic',
                confidence=1.0,
                method='synthetic'
            )
        
        # Add default values for missing required fields
        required_fields = {
            'incident_type': 'Security Event',
            'severity': 'Medium',
            'log_source': 'Unknown'
        }
        
        for field_name, default_value in required_fields.items():
            if field_name not in results or results[field_name].value is None:
                results[field_name] = ExtractionResult(
                    field_name=field_name,
                    value=default_value,
                    source_path='default',
                    confidence=0.5,
                    method='default_value'
                )
        
        return results
    
    def learn_from_feedback(self, field_name: str, correct_source_path: str, schema_hint: Optional[str] = None):
        """Learn from user feedback with memory management"""
        if field_name in self.field_mappings:
            mapping = self.field_mappings[field_name]
            
            # Add to learned patterns if not already present
            if correct_source_path not in mapping.learned_patterns:
                mapping.learned_patterns.insert(0, correct_source_path)
                
                # Limit learned patterns to prevent bloat
                if len(mapping.learned_patterns) > 20:  # Increased limit slightly
                    mapping.learned_patterns = mapping.learned_patterns[:20]
                
                logger.info(f"Learned new pattern for {field_name}: {correct_source_path}")
        
        # Store in feedback patterns with limits
        if len(self.feedback_patterns[field_name]) >= self._max_feedback_patterns:
            # Remove oldest patterns
            self.feedback_patterns[field_name] = self.feedback_patterns[field_name][-50:]
        
        self.feedback_patterns[field_name].append({
            'source_path': correct_source_path,
            'timestamp': datetime.now().isoformat(),
            'schema_hint': schema_hint
        })
    
    def get_extraction_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get extraction performance statistics"""
        stats = {}
        
        for field_name, mapping in self.field_mappings.items():
            stats[field_name] = {
                'usage_count': mapping.usage_count,
                'confidence_score': mapping.confidence_score,
                'learned_patterns': len(mapping.learned_patterns),
                'required': mapping.required,
                'feedback_count': len(self.feedback_patterns.get(field_name, []))
            }
        
        return stats

# Lazy loading for components that might cause circular imports
def get_enhanced_field_extractor():
    """Get enhanced field extractor with semantic learning"""
    try:
        # Lazy import to avoid circular dependency
        from semantic_enhancements import SemanticLearningEngine
        
        class EnhancedSmartFieldExtractor(SmartFieldExtractor):
            """Field extractor ที่มี semantic learning"""
            
            def __init__(self):
                super().__init__()
                try:
                    self.semantic_engine = SemanticLearningEngine()
                    logger.info(" Enhanced field extractor with semantic learning")
                except Exception as e:
                    logger.warning(f" Semantic engine unavailable: {e}")
                    self.semantic_engine = None
            
            def extract_with_semantic_learning(self, data: Dict[str, Any], 
                                             schema_hint: Optional[str] = None) -> Dict[str, ExtractionResult]:
                """Extract fields พร้อม semantic learning"""
                
                # 1. ทำ extraction ปกติก่อน
                results = self.extract_all_fields(data, schema_hint)
                
                # 2. ใช้ semantic engine ช่วยทำนาย fields ที่หายไป
                if self.semantic_engine:
                    missing_fields = self._find_missing_critical_fields(results)
                    
                    for field_name in missing_fields:
                        predictions = self.semantic_engine.predict_field_mapping(
                            "", None, schema_hint or 'unknown'
                        )
                        
                        for source_path, confidence in predictions:
                            if confidence > 0.8:  # High confidence threshold
                                value = SafeDataProcessor.safe_get_nested(data, source_path)
                                if value is not None:
                                    results[field_name] = ExtractionResult(
                                        field_name=field_name,
                                        value=value,
                                        source_path=source_path,
                                        confidence=confidence,
                                        method='semantic_prediction'
                                    )
                                    break
                
                return results
            
            def learn_from_correction(self, field_name: str, correct_source_path: str,
                                    source_data: Dict[str, Any], schema_type: str):
                """เรียนรู้จากการแก้ไขของ user"""
                
                # อัปเดต field mappings เดิม
                super().learn_from_feedback(field_name, correct_source_path, schema_type)
                
                # เรียนรู้ semantic patterns
                if self.semantic_engine:
                    self.semantic_engine.learn_field_mapping(
                        correct_source_path, field_name, source_data, schema_type
                    )
                    
                    logging.info(f"Semantic learning updated for {field_name} → {correct_source_path}")
            
            def _find_missing_critical_fields(self, results: Dict[str, ExtractionResult]) -> List[str]:
                """Find missing critical fields that need semantic prediction"""
                
                critical_fields = ['alert_name', 'severity', 'detected_time', 'contexts.hostname', 'log_source']
                missing_fields = []
                
                for field in critical_fields:
                    if field not in results or results[field].value is None:
                        missing_fields.append(field)
                
                return missing_fields
        
        return EnhancedSmartFieldExtractor()
        
    except ImportError:
        # Fallback to regular field extractor
        logger.info("Using standard field extractor (semantic learning unavailable)")
        return SmartFieldExtractor()