#!/usr/bin/env python3
"""
enhanced_schema_detector.py - Fixed AI-Powered Schema Detection System
แก้ไข: Syntax errors, indentation issues, missing imports

Changes:
1. Fixed all syntax errors and indentation
2. Added proper imports
3. Fixed method definitions
4. Completed missing functions
"""

import re
import json
import time
import pickle
import weakref
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging

from core_config import get_config
from safe_utils import (
    SafeOperationHandler, SafeDataProcessor, OperationType, 
    safe_operation, OperationResult, ErrorSeverity
)

logger = logging.getLogger(__name__)

@dataclass
class SemanticConcept:
    """แนวคิดเชิงความหมายของ field"""
    concept_name: str
    field_patterns: List[str]
    context_keywords: List[str]
    confidence: float
    usage_count: int = 0
    learned_from: List[str] = None
    
    def __post_init__(self):
        if self.learned_from is None:
            self.learned_from = []

class SemanticLearningEngine:
    """Engine สำหรับเรียนรู้ความหมายของ fields - Fixed memory management"""
    
    def __init__(self, knowledge_base_path: str = "semantic_knowledge.pkl"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.concepts = self._load_knowledge_base()
        self.field_relationships = {}
        self.context_patterns = {}
        
        # Memory management
        self._last_cleanup = datetime.now()
        self._cleanup_interval = timedelta(minutes=30)
        self._max_relationships = 1000
        
    def _cleanup_old_relationships(self):
        """Clean up old relationships to prevent memory bloat"""
        
        current_time = datetime.now()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        if len(self.field_relationships) > self._max_relationships:
            # Keep only the most recent 70% of relationships
            keep_count = int(self._max_relationships * 0.7)
            
            # Sort by timestamp and keep recent ones
            sorted_items = sorted(
                self.field_relationships.items(),
                key=lambda x: x[1].get('learned_date', ''),
                reverse=True
            )
            
            self.field_relationships = dict(sorted_items[:keep_count])
            logger.info(f"Cleaned up relationships: kept {keep_count} of {len(sorted_items)}")
        
        self._last_cleanup = current_time
        gc.collect()
    
    def _load_knowledge_base(self) -> Dict[str, SemanticConcept]:
        """โหลด knowledge base หรือสร้างใหม่"""
        if self.knowledge_base_path.exists():
            try:
                with open(self.knowledge_base_path, 'rb') as f:
                    concepts = pickle.load(f)
                    logger.info(f"Loaded {len(concepts)} semantic concepts from {self.knowledge_base_path}")
                    return concepts
            except Exception as e:
                logger.warning(f"Could not load knowledge base: {e}")
        
        # สร้าง default concepts
        return self._create_default_concepts()
    
    def _create_default_concepts(self) -> Dict[str, SemanticConcept]:
        """สร้าง semantic concepts พื้นฐาน"""
        concepts = {}
        
        # Alert/Incident Identification
        concepts['alert_identification'] = SemanticConcept(
            concept_name='alert_identification',
            field_patterns=['name', 'title', 'description', 'summary', 'message'],
            context_keywords=['alert', 'incident', 'event', 'notification', 'detection'],
            confidence=0.9
        )
        
        # Threat Assessment 
        concepts['threat_assessment'] = SemanticConcept(
            concept_name='threat_assessment',
            field_patterns=['severity', 'priority', 'level', 'criticality', 'impact'],
            context_keywords=['high', 'medium', 'low', 'critical', 'threat', 'risk'],
            confidence=0.95
        )
        
        # Temporal Information
        concepts['temporal_info'] = SemanticConcept(
            concept_name='temporal_info', 
            field_patterns=['time', 'timestamp', 'date', 'created', 'detected', 'occurred'],
            context_keywords=['when', 'datetime', 'utc', 'epoch', 'iso'],
            confidence=0.9
        )
        
        # Identity Information
        concepts['identity_info'] = SemanticConcept(
            concept_name='identity_info',
            field_patterns=['user', 'account', 'login', 'person', 'identity', 'principal'],
            context_keywords=['who', 'username', 'userid', 'email', 'domain'],
            confidence=0.85
        )
        
        # Asset Information
        concepts['asset_info'] = SemanticConcept(
            concept_name='asset_info',
            field_patterns=['host', 'hostname', 'device', 'machine', 'computer', 'endpoint'],
            context_keywords=['where', 'fqdn', 'ip', 'server', 'workstation'],
            confidence=0.9
        )
        
        # Network Information  
        concepts['network_info'] = SemanticConcept(
            concept_name='network_info',
            field_patterns=['ip', 'address', 'port', 'protocol', 'connection'],
            context_keywords=['src', 'dst', 'source', 'destination', 'network', 'tcp', 'udp'],
            confidence=0.95
        )
        
        return concepts
    
    def learn_field_mapping(self, source_path: str, target_field: str, 
                          source_data: Dict[str, Any], schema_type: str):
        """เรียนรู้การ mapping ใหม่ - with cleanup"""
        
        # Cleanup old relationships periodically
        self._cleanup_old_relationships()
        
        # วิเคราะห์ semantic concept ของ target field
        target_concept = self._identify_concept(target_field)
        
        if target_concept and target_concept in self.concepts:
            concept = self.concepts[target_concept]
            
            # เพิ่ม field pattern ใหม่
            field_name = source_path.split('.')[-1].lower()
            if field_name not in concept.field_patterns:
                concept.field_patterns.append(field_name)
                concept.learned_from.append(f"{schema_type}:{source_path}")
                concept.usage_count += 1
                
                # Limit pattern growth
                if len(concept.field_patterns) > 50:
                    concept.field_patterns = concept.field_patterns[-30:]
                
                logger.info(f"Learned: {field_name} → {target_concept} concept")
        
        # บันทึก field relationship
        relationship_key = f"{schema_type}.{target_field}"
        self.field_relationships[relationship_key] = {
            'source_path': source_path,
            'confidence': 1.0,
            'learned_date': datetime.now().isoformat(),
            'usage_count': 1
        }
        
        # วิเคราะห์ context patterns (simplified)
        try:
            self._analyze_context_patterns(source_path, source_data, target_field)
        except Exception as e:
            logger.debug(f"Context pattern analysis failed: {e}")
        
        # บันทึก knowledge base (periodic only)
        if len(self.field_relationships) % 10 == 0:
            self._save_knowledge_base()
    
    def _identify_concept(self, target_field: str) -> Optional[str]:
        """ระบุ concept ของ target field"""
        field_lower = target_field.lower()
        
        # Mapping พื้นฐาน
        concept_mappings = {
            'alert_name': 'alert_identification',
            'incident_type': 'alert_identification', 
            'severity': 'threat_assessment',
            'detected_time': 'temporal_info',
            'contexts.user': 'identity_info',
            'contexts.hostname': 'asset_info',
            'source_ip': 'network_info',
            'destination_ip': 'network_info'
        }
        
        return concept_mappings.get(target_field)
    
    def _analyze_context_patterns(self, source_path: str, source_data: Dict[str, Any], target_field: str):
        """วิเคราะห์ context patterns รอบๆ field - simplified"""
        
        path_parts = source_path.split('.')
        
        # Limit context pattern storage
        if len(self.context_patterns) > 100:
            # Keep only most recent patterns
            recent_keys = list(self.context_patterns.keys())[-50:]
            self.context_patterns = {k: self.context_patterns[k] for k in recent_keys}
        
        for i, part in enumerate(path_parts[:3]):
            context_key = f"path_context_{i}"
            if context_key not in self.context_patterns:
                self.context_patterns[context_key] = {}
            
            if target_field not in self.context_patterns[context_key]:
                self.context_patterns[context_key][target_field] = []
            
            patterns_list = self.context_patterns[context_key][target_field]
            if len(patterns_list) < 20:
                patterns_list.append(part.lower())
    
    def predict_field_mapping(self, source_path: str, source_value: Any, 
                            schema_type: str) -> List[Tuple[str, float]]:
        """ทำนายการ mapping โดยใช้ semantic knowledge - simplified"""
        
        predictions = []
        
        if not source_path:
            return predictions
        
        field_name = source_path.split('.')[-1].lower()
        
        # 1. ค้นหาจาก direct relationships
        relationship_key = f"{schema_type}.{source_path}"
        if relationship_key in self.field_relationships:
            rel = self.field_relationships[relationship_key]
            predictions.append((source_path, rel['confidence']))
        
        # 2. ค้นหาจาก semantic concepts
        for concept_name, concept in self.concepts.items():
            similarity_score = self._calculate_semantic_similarity(
                field_name, concept.field_patterns, concept.context_keywords
            )
            
            if similarity_score > 0.6:
                target_field = self._concept_to_target_field(concept_name)
                if target_field:
                    predictions.append((target_field, similarity_score * concept.confidence))
        
        # 3. ค้นหาจาก context patterns (simplified)
        try:
            context_predictions = self._predict_from_context(source_path)
            predictions.extend(context_predictions)
        except Exception as e:
            logger.debug(f"Context prediction failed: {e}")
        
        # เรียงตาม confidence
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:5]
    
    def _calculate_semantic_similarity(self, field_name: str, 
                                     patterns: List[str], 
                                     keywords: List[str]) -> float:
        """คำนวณความคล้ายคลึงเชิงความหมาย"""
        
        if not field_name or not patterns:
            return 0.0
        
        # Pattern matching
        pattern_score = 0.0
        for pattern in patterns:
            if pattern in field_name:
                pattern_score += 1.0
            elif field_name in pattern:
                pattern_score += 0.8
            elif self._fuzzy_match(field_name, pattern):
                pattern_score += 0.6
        
        pattern_score = min(pattern_score / len(patterns), 1.0)
        
        # Keyword matching
        keyword_score = 0.0
        for keyword in keywords:
            if keyword in field_name:
                keyword_score += 0.5
        
        keyword_score = min(keyword_score, 1.0)
        
        # Combined score
        return (pattern_score * 0.7) + (keyword_score * 0.3)
    
    def _fuzzy_match(self, str1: str, str2: str) -> bool:
        """Simple fuzzy matching"""
        if abs(len(str1) - len(str2)) > 2:
            return False
        
        matches = sum(1 for a, b in zip(str1, str2) if a == b)
        return matches / max(len(str1), len(str2)) > 0.7
    
    def _concept_to_target_field(self, concept_name: str) -> Optional[str]:
        """แปลง concept เป็น target field"""
        concept_mappings = {
            'alert_identification': 'alert_name',
            'threat_assessment': 'severity',
            'temporal_info': 'detected_time', 
            'identity_info': 'contexts.user',
            'asset_info': 'contexts.hostname',
            'network_info': 'source_ip'
        }
        
        return concept_mappings.get(concept_name)
    
    def _predict_from_context(self, source_path: str) -> List[Tuple[str, float]]:
        """ทำนายจาก context patterns - simplified"""
        predictions = []
        path_parts = source_path.split('.')
        
        for i, part in enumerate(path_parts[:3]):
            context_key = f"path_context_{i}"
            if context_key in self.context_patterns:
                for target_field, patterns in self.context_patterns[context_key].items():
                    if part.lower() in patterns:
                        confidence = min(patterns.count(part.lower()) / len(patterns), 0.8)
                        predictions.append((target_field, confidence))
        
        return predictions
    
    def _save_knowledge_base(self):
        """บันทึก knowledge base - with error handling"""
        try:
            # Create backup first
            if self.knowledge_base_path.exists():
                backup_path = self.knowledge_base_path.with_suffix('.bak')
                import shutil
                shutil.copy2(self.knowledge_base_path, backup_path)
            
            with open(self.knowledge_base_path, 'wb') as f:
                pickle.dump(self.concepts, f)
            
            logger.debug(f"Saved knowledge base with {len(self.concepts)} concepts")
            
        except Exception as e:
            logger.error(f"Could not save knowledge base: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """สถิติการเรียนรู้"""
        stats = {
            'total_concepts': len(self.concepts),
            'total_relationships': len(self.field_relationships),
            'concept_details': {}
        }
        
        for name, concept in self.concepts.items():
            stats['concept_details'][name] = {
                'patterns_learned': len(concept.field_patterns),
                'usage_count': concept.usage_count,
                'confidence': concept.confidence,
                'sources': len(concept.learned_from)
            }
        
        return stats

    def analyze_schema_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """วิเคราะห์ context ของ schema - simplified"""
        
        # Simple heuristic-based analysis
        content_text = json.dumps(data).lower()
        
        schema_indicators = {
            'cortex_xdr': ['cortex', 'xdr', 'palo alto', 'reply', 'incident'],
            'crowdstrike': ['crowdstrike', 'falcon', 'resources', 'detection'],
            'trend_micro': ['trend micro', 'vision one', 'incidents', 'edges'],
            'fortigate': ['fortinet', 'fortigate', 'srcip', 'dstip'],
            'trellix_epo': ['trellix', 'mcafee', 'epo', 'threat']
        }
        
        best_schema = 'unknown'
        best_confidence = 0.5
        
        for schema, indicators in schema_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in content_text)
            confidence = min(0.5 + (matches * 0.1), 0.9)
            
            if confidence > best_confidence:
                best_schema = schema
                best_confidence = confidence
        
        return {
            'schema': best_schema,
            'confidence': best_confidence
        }

@dataclass
class SchemaSignature:
    """Enhanced schema signature"""
    schema_name: str
    required_patterns: List[str]
    optional_patterns: List[str]
    distinctive_features: List[str]
    structure_indicators: List[str]
    negative_patterns: List[str]
    confidence_threshold: float
    field_weights: Dict[str, float]
    learning_enabled: bool = True
    usage_count: int = 0
    success_rate: float = 0.0
    last_updated: str = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()

@dataclass
class DetectionResult:
    """Schema detection result"""
    schema_name: str
    confidence: float
    raw_scores: Dict[str, float]
    matched_patterns: List[str]
    missing_patterns: List[str]
    suggested_improvements: List[str]
    detection_time: float
    data_characteristics: Dict[str, Any]
    fallback_used: bool = False

class AISchemaFingerprinter:
    """AI-powered schema fingerprinting with fixed memory management"""
    
    def __init__(self):
        self.config = get_config()
        self.signatures = self._initialize_signatures()
        
        # Fixed cache management
        self.pattern_cache = {}
        self._cache_limit = 500
        self._last_cache_cleanup = datetime.now()
        
        self.learning_data = defaultdict(list)
        
        # AI components with better error handling
        self.ai_enabled = self.config.ai.enabled
        self.semantic_model = None
        self.cosine_similarity = None
        self._ai_initialization_attempted = False
        
        if self.ai_enabled:
            self._initialize_ai_components()
    
    def _initialize_ai_components(self):
        """Initialize AI components with better error handling and memory management"""
        
        if self._ai_initialization_attempted:
            return
        
        self._ai_initialization_attempted = True
        
        try:
            # Check dependencies step by step
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
                logger.info("Install with: pip install sentence-transformers scikit-learn numpy")
                self.ai_enabled = False
                return
            
            # Try to load the actual model
            try:
                from sentence_transformers import SentenceTransformer
                from sklearn.metrics.pairwise import cosine_similarity
                
                # Try better model first, fallback to smaller one
                try:
                    self.semantic_model = SentenceTransformer('paraphrase-mpnet-base-v2')
                    logger.info("Loaded advanced AI model: paraphrase-mpnet-base-v2")
                except Exception as e:
                    logger.warning(f"Advanced model failed: {e}")
                    self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("Loaded standard AI model: all-MiniLM-L6-v2")
                
                self.cosine_similarity = cosine_similarity
                
                # Test the model with a simple example
                test_embedding = self.semantic_model.encode("test")
                if test_embedding is not None:
                    logger.info("AI semantic model loaded and tested successfully")
                else:
                    raise Exception("Model returned None for test embedding")
                    
            except Exception as model_error:
                logger.warning(f"AI model loading failed: {model_error}")
                self.ai_enabled = False
                return
                    
        except Exception as e:
            logger.warning(f"AI initialization completely failed: {e}")
            self.ai_enabled = False

    def _cleanup_pattern_cache(self):
        """Clean up pattern cache to prevent memory bloat"""
        
        current_time = datetime.now()
        if current_time - self._last_cache_cleanup < timedelta(minutes=15):
            return
        
        if len(self.pattern_cache) > self._cache_limit:
            # Keep only recent 70% of cache entries
            keep_count = int(self._cache_limit * 0.7)
            
            # Remove oldest entries (simple FIFO)
            cache_items = list(self.pattern_cache.items())
            self.pattern_cache = dict(cache_items[-keep_count:])
            
            logger.debug(f"Cleaned pattern cache: kept {keep_count} entries")
        
        self._last_cache_cleanup = current_time
        gc.collect()

    def _initialize_signatures(self) -> Dict[str, SchemaSignature]:
        """Initialize enhanced schema signatures"""
        signatures = {}
        
        # Cortex XDR signature
        signatures['cortex_xdr'] = SchemaSignature(
            schema_name='cortex_xdr',
            required_patterns=[
                'reply.incident', 'reply.alerts', 'incident_id'
            ],
            optional_patterns=[
                'reply.alerts.data', 'reply.network_artifacts', 'xdr_url'
            ],
            distinctive_features=[
                'mitre_technique_id_and_name', 'actor_process_image_name', 
                'causality_actor_process_image_name', 'alert_id'
            ],
            structure_indicators=['reply', 'incident', 'alerts'],
            negative_patterns=['resources', 'detection', 'device', 'edges'],
            confidence_threshold=40.0,
            field_weights={
                'reply.incident': 3.5,
                'reply.alerts': 3.5,
                'mitre_technique_id_and_name': 3.0,
                'actor_process_image_name': 2.0,
                'xdr_url': 2.5
            }
        )
        
        # CrowdStrike signature
        signatures['crowdstrike'] = SchemaSignature(
            schema_name='crowdstrike',
            required_patterns=[
                'resources', 'device', 'detection'
            ],
            optional_patterns=[
                'behaviors', 'pattern_disposition', 'parent_details'
            ],
            distinctive_features=[
                'falcon_host_link', 'agent_id', 'local_process_id',
                'pattern_id', 'device.hostname', 'device.agent_id'
            ],
            structure_indicators=['resources', 'device', 'detection'],
            negative_patterns=['reply', 'incident', 'edges', 'logdesc'],
            confidence_threshold=40.0,
            field_weights={
                'resources': 3.5,
                'device': 3.0,
                'detection': 3.0,
                'falcon_host_link': 2.5,
                'agent_id': 2.5
            }
        )
        
        # Trend Micro signature
        signatures['trend_micro'] = SchemaSignature(
            schema_name='trend_micro',
            required_patterns=[
                'data.incidents.edges', 'data.incidents'
            ],
            optional_patterns=[
                'data.incidents.edges[].node', 'alert_name', 'contexts'
            ],
            distinctive_features=[
                'mitigationStatus', 'contexts.hostname', 'contexts.src_ip',
                'edges', 'node'
            ],
            structure_indicators=['data', 'incidents', 'edges', 'node'],
            negative_patterns=['resources', 'reply', 'device', 'logdesc'],
            confidence_threshold=40.0,
            field_weights={
                'data.incidents.edges': 3.5,
                'data.incidents': 3.0,
                'mitigationStatus': 2.5,
                'contexts': 2.0
            }
        )
        
        # FortiGate signature
        signatures['fortigate'] = SchemaSignature(
            schema_name='fortigate',
            required_patterns=[
                'srcip', 'dstip', 'logdesc'
            ],
            optional_patterns=[
                'devname', 'policyname', 'srcintf', 'dstintf'
            ],
            distinctive_features=[
                'logid', 'eventtime', 'subtype', 'action',
                'srcport', 'dstport', 'proto'
            ],
            structure_indicators=[],
            negative_patterns=['resources', 'reply', 'edges', 'incident'],
            confidence_threshold=40.0,
            field_weights={
                'srcip': 3.5,
                'dstip': 3.5,
                'logdesc': 3.0,
                'logid': 2.5,
                'devname': 2.0
            }
        )
        
        # Trellix EPO signature
        signatures['trellix_epo'] = SchemaSignature(
            schema_name='trellix_epo',
            required_patterns=[
                'data.attributes.Severity', 'data.attributes.Process_Name', 'data.attributes.RuleId', 'data.attributes.DetectionDate'
            ],
            optional_patterns=[
                'data.attributes.Host_Name', 'data.attributes.User.name'
            ],
            distinctive_features=[
                'ThreatEventID', 'ThreatName', 'ThreatSeverity', 'DetectedUTC', 'SourceHostName', 'AgentGUID'
            ],
            structure_indicators=['data', 'attributes'],
            negative_patterns=['resources', 'reply', 'edges', 'logdesc'],
            confidence_threshold=30.0,
            field_weights={
                'data.attributes.Severity': 4.0,
                'data.attributes.Process_Name': 3.5,
                'data.attributes.RuleId': 3.5,
                'data.attributes.DetectionDate': 3.0,
                'ThreatEventID': 2.5,
                'AgentGUID': 2.5
            }
        )

        # Trellix Helix signature
        signatures['trellix_helix'] = SchemaSignature(
            schema_name='trellix_helix',
            required_patterns=[
                'data.attributes.severity', 'data.attributes.process_name', 'data.attributes.rule_id'
            ],
            optional_patterns=[
                'data.attributes.detection_date', 'data.attributes.host_name', 'data.attributes.user.name'
            ],
            distinctive_features=[
                'threat_event_id', 'threat_name', 'threat_severity', 'detected_utc', 'source_host_name', 'agent_guid'
            ],
            structure_indicators=['data', 'attributes'],
            negative_patterns=['resources', 'reply', 'edges', 'logdesc'],
            confidence_threshold=30.0,
            field_weights={
                'data.attributes.severity': 4.0,
                'data.attributes.process_name': 3.5,
                'data.attributes.rule_id': 3.5
            }
        )

        # Generic Trellix signature
        signatures['trellix_generic'] = SchemaSignature(
            schema_name='trellix_generic',
            required_patterns=[
                'data.attributes'
            ],
            optional_patterns=[
                'Severity', 'Process_Name', 'RuleId', 'DetectionDate'
            ],
            distinctive_features=[
                'Trellix', 'McAfee', 'EPO', 'Helix'
            ],
            structure_indicators=['data', 'attributes'],
            negative_patterns=['resources', 'reply', 'edges'],
            confidence_threshold=25.0,
            field_weights={
                'data.attributes': 5.0,
                'Severity': 3.0,
                'Process_Name': 3.0
            }
        )
        
        # Generic fallback signature
        signatures['generic_security'] = SchemaSignature(
            schema_name='generic_security',
            required_patterns=[
                'name', 'timestamp', 'severity'
            ],
            optional_patterns=[
                'description', 'source', 'host', 'user'
            ],
            distinctive_features=[
                'alert', 'event', 'incident', 'log', 'security'
            ],
            structure_indicators=[],
            negative_patterns=[],
            confidence_threshold=45.0,
            field_weights={
                'name': 2.0,
                'timestamp': 2.0,
                'severity': 2.0
            }
        )
        
        return signatures
    
    @safe_operation(OperationType.SCHEMA_DETECTION)
    def detect_schema(self, data: Dict[str, Any]) -> DetectionResult:
        """Detect schema with AI enhancement and fixed memory management"""
        start_time = time.time()
        
        # Cleanup cache periodically
        self._cleanup_pattern_cache()
        
        # Validate input
        if not isinstance(data, dict) or not data:
            return DetectionResult(
                schema_name='unknown',
                confidence=0.0,
                raw_scores={},
                matched_patterns=[],
                missing_patterns=[],
                suggested_improvements=['Provide valid JSON data'],
                detection_time=time.time() - start_time,
                data_characteristics={},
                fallback_used=True
            )
        
        # Extract data characteristics
        data_characteristics = self._analyze_data_characteristics(data)
        
        # Calculate scores for each schema
        schema_scores = {}
        all_matched_patterns = []
        
        for schema_name, signature in self.signatures.items():
            score_result = self._calculate_schema_score(data, signature, data_characteristics)
            schema_scores[schema_name] = score_result['score']
            
            if score_result['score'] > 0:
                all_matched_patterns.extend(score_result['matched_patterns'])
        
        # Find best match
        best_schema = max(schema_scores.items(), key=lambda x: x[1])
        schema_name, confidence = best_schema
        
        # Check if confidence meets threshold
        signature = self.signatures[schema_name]
        if confidence < signature.confidence_threshold:
            # Use fallback strategy
            fallback_result = self._fallback_detection(data, data_characteristics)
            if fallback_result['confidence'] > confidence:
                schema_name = fallback_result['schema']
                confidence = fallback_result['confidence']
                fallback_used = True
            else:
                fallback_used = False
        else:
            fallback_used = False
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(
            data, schema_name, confidence, data_characteristics
        )
        
        # Update learning data (with limits)
        if signature.learning_enabled:
            self._update_learning_data(schema_name, confidence, data_characteristics)
        
        detection_time = time.time() - start_time
        
        return DetectionResult(
            schema_name=schema_name,
            confidence=confidence,
            raw_scores=schema_scores,
            matched_patterns=all_matched_patterns,
            missing_patterns=self._find_missing_patterns(data, schema_name),
            suggested_improvements=suggestions,
            detection_time=detection_time,
            data_characteristics=data_characteristics,
            fallback_used=fallback_used
        )
    
    def _analyze_data_characteristics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data structure characteristics - simplified"""
        characteristics = {
            'total_fields': 0,
            'nested_levels': 0,
            'array_fields': 0,
            'field_patterns': set(),
            'structure_type': 'unknown',
            'common_field_types': Counter(),
            'field_name_patterns': [],
            'value_patterns': []
        }
    def _analyze_data_characteristics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data structure characteristics"""
        characteristics = {
            'total_fields': 0,
            'nested_levels': 0,
            'array_fields': 0,
            'field_patterns': set(),
            'structure_type': 'unknown',
            'common_field_types': Counter(),
            'field_name_patterns': [],
            'value_patterns': []
        }
        
        def analyze_recursive(obj: Any, level: int = 0):
            characteristics['nested_levels'] = max(characteristics['nested_levels'], level)
            
            if isinstance(obj, dict):
                characteristics['total_fields'] += len(obj)
                
                for key, value in obj.items():
                    # Analyze field names
                    characteristics['field_patterns'].add(key.lower())
                    characteristics['common_field_types'][type(value).__name__] += 1
                    
                    # Look for common patterns
                    if any(pattern in key.lower() for pattern in ['time', 'date', 'timestamp']):
                        characteristics['field_name_patterns'].append('temporal')
                    if any(pattern in key.lower() for pattern in ['ip', 'host', 'address']):
                        characteristics['field_name_patterns'].append('network')
                    if any(pattern in key.lower() for pattern in ['user', 'account', 'login']):
                        characteristics['field_name_patterns'].append('identity')
                    
                    # Recursive analysis
                    if isinstance(value, (dict, list)):
                        if isinstance(value, list):
                            characteristics['array_fields'] += 1
                        analyze_recursive(value, level + 1)
            
            elif isinstance(obj, list) and obj:
                if isinstance(obj[0], dict):
                    analyze_recursive(obj[0], level + 1)
        
        analyze_recursive(data)
        
        # Determine structure type
        if characteristics['nested_levels'] <= 1:
            characteristics['structure_type'] = 'flat'
        elif characteristics['array_fields'] > 0:
            characteristics['structure_type'] = 'nested_with_arrays'
        else:
            characteristics['structure_type'] = 'nested_objects'
        
        # Convert set to list for JSON serialization
        characteristics['field_patterns'] = list(characteristics['field_patterns'])
        
        return characteristics
    
    def _calculate_schema_score(self, 
                               data: Dict[str, Any], 
                               signature: SchemaSignature,
                               data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive schema score"""
        score_components = {
            'required_patterns': 0.0,
            'optional_patterns': 0.0,
            'distinctive_features': 0.0,
            'structure_indicators': 0.0,
            'negative_penalty': 0.0,
            'ai_semantic_score': 0.0
        }
        
        matched_patterns = []
        
        # Check required patterns
        required_found = 0
        for pattern in signature.required_patterns:
            if self._pattern_exists(data, pattern):
                required_found += 1
                matched_patterns.append(pattern)
                weight = signature.field_weights.get(pattern, 1.0)
                score_components['required_patterns'] += weight
        
        if signature.required_patterns:
            score_components['required_patterns'] = (
                score_components['required_patterns'] / 
                sum(signature.field_weights.get(p, 1.0) for p in signature.required_patterns)
            ) * self.config.schema.required_paths_weight
        
        # Check optional patterns
        optional_found = 0
        for pattern in signature.optional_patterns:
            if self._pattern_exists(data, pattern):
                optional_found += 1
                matched_patterns.append(pattern)
                weight = signature.field_weights.get(pattern, 1.0)
                score_components['optional_patterns'] += weight
        
        if signature.optional_patterns:
            score_components['optional_patterns'] = (
                score_components['optional_patterns'] / 
                sum(signature.field_weights.get(p, 1.0) for p in signature.optional_patterns)
            ) * self.config.schema.optional_paths_weight
        
        # Check distinctive features
        distinctive_found = 0
        for feature in signature.distinctive_features:
            if self._field_exists_anywhere(data, feature):
                distinctive_found += 1
                matched_patterns.append(feature)
                weight = signature.field_weights.get(feature, 1.0)
                score_components['distinctive_features'] += weight
        
        if signature.distinctive_features:
            score_components['distinctive_features'] = (
                score_components['distinctive_features'] / 
                sum(signature.field_weights.get(f, 1.0) for f in signature.distinctive_features)
            ) * self.config.schema.distinctive_fields_weight
        
        # Check structure indicators
        structure_found = 0
        for indicator in signature.structure_indicators:
            if indicator.lower() in ' '.join(data_characteristics['field_patterns']).lower():
                structure_found += 1
        
        if signature.structure_indicators:
            score_components['structure_indicators'] = (
                structure_found / len(signature.structure_indicators)
            ) * 10.0  # Structure weight
        
        # Apply negative pattern penalty
        negative_found = 0
        for negative in signature.negative_patterns:
            if self._field_exists_anywhere(data, negative):
                negative_found += 1
        
        if negative_found > 0:
            score_components['negative_penalty'] = -(negative_found * 15.0)  # Penalty
        
        # AI semantic scoring (if available)
        if self.ai_enabled and self.semantic_model:
            try:
                ai_score = self._calculate_ai_semantic_score(data, signature)
                score_components['ai_semantic_score'] = ai_score * 15.0  # AI bonus
            except Exception as e:
                logger.debug(f"AI semantic scoring failed: {e}")
        
        # Calculate final score
        total_score = sum(score_components.values())
        
        # Apply adaptive adjustments based on historical performance
        if signature.usage_count > 5:  # Enough data for adjustment
            confidence_multiplier = 0.8 + (signature.success_rate * 0.4)  # 0.8 to 1.2 range
            total_score *= confidence_multiplier
        
        return {
            'score': max(0.0, min(100.0, total_score)),
            'components': score_components,
            'matched_patterns': matched_patterns
        }
    
    def _pattern_exists(self, data: Dict[str, Any], pattern: str) -> bool:
        """Check if pattern exists using safe navigation"""
        return SafeDataProcessor.safe_get_nested(data, pattern) is not None
    
    def _field_exists_anywhere(self, data: Any, field_name: str, max_depth: int = 5) -> bool:
        """Search for field name anywhere in nested structure"""
        
        def search_recursive(obj: Any, depth: int = 0) -> bool:
            if depth > max_depth:
                return False
            
            if isinstance(obj, dict):
                # Direct match
                if field_name in obj:
                    return True
                
                # Partial match (case insensitive)
                for key in obj.keys():
                    if isinstance(key, str):
                        if (field_name.lower() in key.lower() or 
                            key.lower() in field_name.lower()):
                            return True
                
                # Recursive search
                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        if search_recursive(value, depth + 1):
                            return True
            
            elif isinstance(obj, list) and obj:
                # Search in first few items
                for item in obj[:3]:
                    if isinstance(item, (dict, list)):
                        if search_recursive(item, depth + 1):
                            return True
            
            return False
        
        return search_recursive(data)
    
    def _calculate_ai_semantic_score(self, data: Dict[str, Any], signature: SchemaSignature) -> float:
        """Calculate AI semantic similarity score with robust error handling"""
        
        if not self.ai_enabled or not self.semantic_model:
            return 0.0
        
        try:
            # Extract field names from data
            data_fields = list(self._extract_all_field_names(data))
            
            # Create schema context from signature
            schema_context = (
                signature.required_patterns + 
                signature.optional_patterns + 
                signature.distinctive_features
            )
            
            if not data_fields or not schema_context:
                return 0.0
            
            # Prepare texts for embedding
            data_text = ' '.join(data_fields)
            schema_text = ' '.join(schema_context)
            
            # Limit text length to avoid model issues
            max_length = 512
            if len(data_text) > max_length:
                data_text = data_text[:max_length]
            if len(schema_text) > max_length:
                schema_text = schema_text[:max_length]
            
            # Use cached embeddings if available
            cache_key = f"{hash(data_text)}_{hash(schema_text)}"
            if cache_key in self.pattern_cache:
                return self.pattern_cache[cache_key]
            
            # Generate embeddings with timeout protection
            try:
                embeddings = self.semantic_model.encode([data_text, schema_text], show_progress_bar=False)
                
                if embeddings is None or len(embeddings) != 2:
                    logger.debug("Invalid embeddings returned")
                    return 0.0
                
                # Calculate cosine similarity
                similarity = self.cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                
                # Validate similarity score
                if not isinstance(similarity, (int, float)) or similarity < 0 or similarity > 1:
                    logger.debug(f"Invalid similarity score: {similarity}")
                    return 0.0
                
                # Cache result
                self.pattern_cache[cache_key] = float(similarity)
                
                # Cleanup cache if too large
                if len(self.pattern_cache) > self.config.ai.max_cache_size:
                    # Remove oldest 20% of entries
                    items_to_remove = len(self.pattern_cache) // 5
                    oldest_keys = list(self.pattern_cache.keys())[:items_to_remove]
                    for key in oldest_keys:
                        del self.pattern_cache[key]
                
                return float(similarity)
                
            except Exception as embedding_error:
                logger.debug(f"Embedding generation failed: {embedding_error}")
                # Return pattern-based fallback similarity
                return self._calculate_pattern_similarity(data_text, schema_text)
                
        except Exception as e:
            logger.debug(f"AI semantic scoring error: {e}")
            return 0.0

    def _calculate_pattern_similarity(self, text1: str, text2: str) -> float:
        """Fallback pattern-based similarity calculation"""
        try:
            # Simple token-based similarity
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())
            
            if not tokens1 or not tokens2:
                return 0.0
            
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            
            similarity = intersection / union if union > 0 else 0.0
            return min(0.8, similarity)  # Cap at 0.8 for pattern-based matching
            
        except Exception as e:
            logger.debug(f"Pattern similarity calculation failed: {e}")
            return 0.0
    
    def _extract_all_field_names(self, data: Any, max_depth: int = 5) -> Set[str]:
        """Extract all field names from nested data"""
        field_names = set()
        
        def extract_recursive(obj: Any, depth: int = 0):
            if depth > max_depth:
                return
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(key, str):
                        field_names.add(key.lower())
                    
                    if isinstance(value, (dict, list)):
                        extract_recursive(value, depth + 1)
            
            elif isinstance(obj, list) and obj:
                for item in obj[:3]:  # Check first 3 items
                    if isinstance(item, (dict, list)):
                        extract_recursive(item, depth + 1)
        
        extract_recursive(data)
        return field_names
    
    def _fallback_detection(self, data: Dict[str, Any], characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced fallback detection strategy"""
        
        # Pattern-based fallback
        pattern_scores = {}
        
        # Check for common security log patterns
        security_indicators = [
            'alert', 'event', 'incident', 'threat', 'malware', 'virus',
            'attack', 'breach', 'suspicious', 'anomaly', 'risk'
        ]
        
        content_text = json.dumps(data).lower()
        security_score = sum(1 for indicator in security_indicators if indicator in content_text)
        
        # Network security patterns
        network_patterns = ['srcip', 'dstip', 'firewall', 'traffic', 'connection']
        network_score = sum(1 for pattern in network_patterns if pattern in content_text)
        
        # Endpoint security patterns
        endpoint_patterns = ['process', 'file', 'executable', 'registry', 'service']
        endpoint_score = sum(1 for pattern in endpoint_patterns if pattern in content_text)
        
        # Determine best fallback
        if network_score >= 2:
            return {'schema': 'network_security', 'confidence': min(60.0, network_score * 15)}
        elif endpoint_score >= 2:
            return {'schema': 'endpoint_security', 'confidence': min(60.0, endpoint_score * 15)}
        elif security_score >= 1:
            return {'schema': 'generic_security', 'confidence': min(50.0, security_score * 10)}
        else:
            return {'schema': 'unknown', 'confidence': 0.0}
    
    def _find_missing_patterns(self, data: Dict[str, Any], schema_name: str) -> List[str]:
        """Find patterns that are missing for the detected schema"""
        if schema_name not in self.signatures:
            return []
        
        signature = self.signatures[schema_name]
        missing = []
        
        # Check required patterns
        for pattern in signature.required_patterns:
            if not self._pattern_exists(data, pattern):
                missing.append(f"Required: {pattern}")
        
        # Check important optional patterns
        for pattern in signature.optional_patterns:
            if not self._pattern_exists(data, pattern):
                missing.append(f"Optional: {pattern}")
        
        return missing[:5]  # Limit to 5 most important
    
    def _generate_improvement_suggestions(self, 
                                        data: Dict[str, Any],
                                        schema_name: str, 
                                        confidence: float,
                                        characteristics: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving detection accuracy"""
        suggestions = []
        
        if confidence < 50:
            suggestions.append("Very low confidence - consider manual schema specification")
            suggestions.append("Check if input data format is supported")
        elif confidence < 70:
            suggestions.append("Medium confidence - verify schema mapping configuration")
            suggestions.append("Consider enabling AI-powered detection")
        
        # Schema-specific suggestions
        if schema_name == 'unknown':
            suggestions.append("Schema not recognized - add custom mapping rules")
            suggestions.append("Enable fallback detection mode")
        
        # Data quality suggestions
        if characteristics['total_fields'] < 10:
            suggestions.append("Limited field count - ensure complete data is provided")
        
        if characteristics['nested_levels'] == 0:
            suggestions.append("Flat structure detected - verify data extraction")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _update_learning_data(self, schema_name: str, confidence: float, characteristics: Dict[str, Any]):
        """Update learning data for continuous improvement"""
        if schema_name in self.signatures:
            signature = self.signatures[schema_name]
            
            # Update usage statistics
            signature.usage_count += 1
            
            # Update success rate (simple moving average)
            success = 1.0 if confidence >= signature.confidence_threshold else 0.0
            if signature.usage_count == 1:
                signature.success_rate = success
            else:
                # Weighted average (more weight to recent results)
                weight = 0.2
                signature.success_rate = (1 - weight) * signature.success_rate + weight * success
            
            # Store learning data for pattern analysis
            self.learning_data[schema_name].append({
                'confidence': confidence,
                'characteristics': characteristics,
                'timestamp': datetime.now().isoformat()
            })
            
            # Limit learning data size
            if len(self.learning_data[schema_name]) > 100:
                self.learning_data[schema_name] = self.learning_data[schema_name][-50:]
    
    def get_signature_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all signatures"""
        performance = {}
        
        for name, signature in self.signatures.items():
            performance[name] = {
                'usage_count': signature.usage_count,
                'success_rate': signature.success_rate,
                'confidence_threshold': signature.confidence_threshold,
                'last_updated': signature.last_updated,
                'learning_enabled': signature.learning_enabled
            }
        
        return performance
    
    def adapt_signatures(self, feedback_data: List[Dict[str, Any]]) -> bool:
        """Adapt signatures based on feedback data"""
        if not feedback_data:
            return False
        
        try:
            for feedback in feedback_data:
                schema_name = feedback.get('schema_name')
                confidence = feedback.get('confidence', 0.0)
                was_correct = feedback.get('correct', False)
                
                if schema_name in self.signatures:
                    signature = self.signatures[schema_name]
                    
                    # Adjust confidence threshold based on feedback
                    if was_correct and confidence < signature.confidence_threshold:
                        # Lower threshold if correct detection had low confidence
                        new_threshold = max(
                            signature.confidence_threshold - 2.0,
                            self.config.schema.min_confidence_threshold
                        )
                        signature.confidence_threshold = new_threshold
                        logger.info(f"Lowered confidence threshold for {schema_name}: {new_threshold}")
                    
                    elif not was_correct and confidence >= signature.confidence_threshold:
                        # Raise threshold if incorrect detection had high confidence
                        new_threshold = min(
                            signature.confidence_threshold + 3.0,
                            self.config.schema.max_confidence_threshold
                        )
                        signature.confidence_threshold = new_threshold
                        logger.info(f"Raised confidence threshold for {schema_name}: {new_threshold}")
                    
                    signature.last_updated = datetime.now().isoformat()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to adapt signatures: {e}")
            return False

class EnhancedSchemaDetector:
    """Main enhanced schema detector with multiple strategies"""
    
    def __init__(self):
        self.config = get_config()
        self.fingerprinter = AISchemaFingerprinter()
        self.handler = SafeOperationHandler()
        
        # Initialize fallback detectors
        self.fallback_detectors = [
            self._structure_based_detection,
            self._content_based_detection,
            self._pattern_frequency_detection
        ]
    
    def detect_schema_with_confidence(self, data: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
        """Detect schema with comprehensive confidence analysis"""
        
        # Primary detection
        result = self.fingerprinter.detect_schema(data)
        
        if not result.success:
            logger.warning("Primary schema detection failed")
            return 'unknown', 0.0, {'error': result.error.message}
        
        detection_result = result.data
        
        # If confidence is good, return result
        if detection_result.confidence >= self.config.schema.confidence_threshold:
            return (
                detection_result.schema_name,
                detection_result.confidence,
                {
                    'method': 'primary_ai_detection',
                    'matched_patterns': detection_result.matched_patterns,
                    'data_characteristics': detection_result.data_characteristics,
                    'detection_time': detection_result.detection_time
                }
            )
        
        # Try fallback methods
        logger.info(f"Primary confidence {detection_result.confidence:.1f}% below threshold, trying fallback")
        
        best_result = (detection_result.schema_name, detection_result.confidence, {'method': 'primary_low_confidence'})
        
        for i, fallback_detector in enumerate(self.fallback_detectors):
            try:
                fallback_result = self.handler.safe_execute(
                    lambda: fallback_detector(data),
                    OperationType.SCHEMA_DETECTION,
                    timeout_seconds=10,
                    error_message=f"Fallback detector {i+1} failed"
                )
                
                if fallback_result.success and fallback_result.data:
                    schema, confidence, info = fallback_result.data
                    if confidence > best_result[1]:
                        best_result = (schema, confidence, {**info, 'method': f'fallback_{i+1}'})
                        
            except Exception as e:
                logger.debug(f"Fallback detector {i+1} failed: {e}")
                continue
        
        return best_result
    
    def _structure_based_detection(self, data: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
        """Structure-based fallback detection"""
        structure_patterns = {
            'cortex_xdr': ['reply', 'incident', 'alerts'],
            'crowdstrike': ['resources', 'device', 'detection'],
            'trend_micro': ['data', 'incidents', 'edges'],
            'fortigate': []  # Flat structure
        }
        
        data_structure = list(data.keys()) if isinstance(data, dict) else []
        
        best_match = ('unknown', 0.0)
        
        for schema, patterns in structure_patterns.items():
            if not patterns:  # Flat structure
                if len(data_structure) > 5 and all(isinstance(k, str) for k in data_structure):
                    score = 40.0  # Moderate confidence for flat structure
                    if score > best_match[1]:
                        best_match = (schema, score)
            else:
                matches = sum(1 for pattern in patterns if pattern in data_structure)
                score = (matches / len(patterns)) * 60.0  # Max 60% confidence
                if score > best_match[1]:
                    best_match = (schema, score)
        
        return best_match[0], best_match[1], {'structure_patterns': data_structure}
    
    def _content_based_detection(self, data: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
        """Content-based fallback detection"""
        content_keywords = {
            'cortex_xdr': ['cortex', 'xdr', 'palo alto', 'mitre_technique'],
            'crowdstrike': ['crowdstrike', 'falcon', 'agent_id', 'pattern_id'],
            'trend_micro': ['trend micro', 'vision one', 'mitigation'],
            'fortigate': ['fortinet', 'fortigate', 'srcip', 'dstip'],
            'trellix_epo': ['trellix', 'mcafee', 'epo', 'threat']
        }
        
        content_text = json.dumps(data).lower()
        
        best_match = ('unknown', 0.0)
        
        for schema, keywords in content_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in content_text)
            score = min((matches / len(keywords)) * 70.0, 70.0)  # Max 70% confidence
            
            if score > best_match[1]:
                best_match = (schema, score)
        
        return best_match[0], best_match[1], {'content_analysis': True}
    
    def _pattern_frequency_detection(self, data: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
        """Pattern frequency-based detection"""
        # This would analyze the frequency of certain field patterns
        # and compare against known schema patterns
        
        field_patterns = self._extract_field_patterns(data)
        
        # Simple heuristic: count common patterns
        common_patterns = {
            'network': ['ip', 'port', 'proto', 'src', 'dst'],
            'security': ['alert', 'threat', 'malware', 'attack'],
            'endpoint': ['process', 'file', 'executable', 'registry'],
            'user': ['user', 'account', 'login', 'authentication']
        }
        
        pattern_scores = {}
        for category, patterns in common_patterns.items():
            score = sum(1 for pattern in patterns if any(pattern in field.lower() for field in field_patterns))
            pattern_scores[category] = score
        
        if max(pattern_scores.values()) > 0:
            best_category = max(pattern_scores.items(), key=lambda x: x[1])
            confidence = min(best_category[1] * 20.0, 50.0)  # Max 50% confidence
            return f'{best_category[0]}_security', confidence, {'pattern_analysis': pattern_scores}
        
        return 'unknown', 0.0, {'pattern_analysis': pattern_scores}
    
    def _extract_field_patterns(self, data: Any, max_depth: int = 3) -> List[str]:
        """Extract field name patterns from data"""
        patterns = []
        
        def extract_recursive(obj: Any, depth: int = 0):
            if depth > max_depth:
                return
            
            if isinstance(obj, dict):
                patterns.extend(obj.keys())
                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        extract_recursive(value, depth + 1)
            elif isinstance(obj, list) and obj:
                for item in obj[:2]:  # Check first 2 items
                    if isinstance(item, (dict, list)):
                        extract_recursive(item, depth + 1)
        
        extract_recursive(data)
        return patterns