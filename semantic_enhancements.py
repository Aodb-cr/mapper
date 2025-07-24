#!/usr/bin/env python3
"""
semantic_enhancements.py - Fixed Semantic Learning System
แก้ไข: Memory management, circular imports, performance issues

Changes:
1. Fixed memory management and cleanup
2. Removed circular import dependencies
3. Better error handling and graceful degradation
4. Simplified complex logic
"""

import json
import pickle
import gc
import weakref
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class SimpleRecord:
    """Simple record for semantic learning - ไม่เปลี่ยน"""
    field_name: str = ""
    schema_type: str = ""
    corrected_source_path: str = ""
    user_action: str = ""

@dataclass
class SemanticConcept:
    """แนวคิดเชิงความหมายของ field - with memory management"""
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
        
        # Memory management settings
        self._max_relationships = 500  # Limit total relationships
        self._max_learned_from = 20    # Limit learned_from entries per concept
        self._last_cleanup = datetime.now()
        self._cleanup_interval = timedelta(minutes=30)
        
        # Performance settings
        self._batch_save_count = 0
        self._save_frequency = 10  # Save every 10 operations
        
    def _cleanup_memory(self):
        """Clean up memory periodically"""
        current_time = datetime.now()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        # Clean up field relationships
        if len(self.field_relationships) > self._max_relationships:
            # Keep only recent 70% of relationships
            keep_count = int(self._max_relationships * 0.7)
            
            # Sort by learned_date and keep recent ones
            sorted_items = sorted(
                self.field_relationships.items(),
                key=lambda x: x[1].get('learned_date', ''),
                reverse=True
            )
            
            self.field_relationships = dict(sorted_items[:keep_count])
            logger.debug(f"Cleaned field relationships: kept {keep_count} of {len(sorted_items)}")
        
        # Clean up learned_from in concepts
        for concept in self.concepts.values():
            if len(concept.learned_from) > self._max_learned_from:
                concept.learned_from = concept.learned_from[-self._max_learned_from:]
        
        self._last_cleanup = current_time
        
        # Force garbage collection
        gc.collect()
        
    def _load_knowledge_base(self) -> Dict[str, SemanticConcept]:
        """โหลด knowledge base หรือสร้างใหม่"""
        if self.knowledge_base_path.exists():
            try:
                with open(self.knowledge_base_path, 'rb') as f:
                    concepts = pickle.load(f)
                logger.info(f"Loaded {len(concepts)} semantic concepts")
                return concepts
            except Exception as e:
                logger.warning(f"Could not load knowledge base: {e}")
        
        return self._create_default_concepts()
    
    def _create_default_concepts(self) -> Dict[str, SemanticConcept]:
        """สร้าง semantic concepts พื้นฐาน - ไม่เปลี่ยน"""
        concepts = {}
        
        concepts['alert_identification'] = SemanticConcept(
            concept_name='alert_identification',
            field_patterns=['name', 'title', 'description', 'summary', 'message'],
            context_keywords=['alert', 'incident', 'event', 'notification', 'detection'],
            confidence=0.9
        )
        
        concepts['threat_assessment'] = SemanticConcept(
            concept_name='threat_assessment',
            field_patterns=['severity', 'priority', 'level', 'criticality', 'impact'],
            context_keywords=['high', 'medium', 'low', 'critical', 'threat', 'risk'],
            confidence=0.95
        )
        
        concepts['temporal_info'] = SemanticConcept(
            concept_name='temporal_info', 
            field_patterns=['time', 'timestamp', 'date', 'created', 'detected', 'occurred'],
            context_keywords=['when', 'datetime', 'utc', 'epoch', 'iso'],
            confidence=0.9
        )
        
        concepts['identity_info'] = SemanticConcept(
            concept_name='identity_info',
            field_patterns=['user', 'account', 'login', 'person', 'identity', 'principal'],
            context_keywords=['who', 'username', 'userid', 'email', 'domain'],
            confidence=0.85
        )
        
        concepts['asset_info'] = SemanticConcept(
            concept_name='asset_info',
            field_patterns=['host', 'hostname', 'device', 'machine', 'computer', 'endpoint'],
            context_keywords=['where', 'fqdn', 'ip', 'server', 'workstation'],
            confidence=0.9
        )
        
        concepts['network_info'] = SemanticConcept(
            concept_name='network_info',
            field_patterns=['ip', 'address', 'port', 'protocol', 'connection'],
            context_keywords=['src', 'dst', 'source', 'destination', 'network', 'tcp', 'udp'],
            confidence=0.95
        )
        
        return concepts
    
    def learn_field_mapping(self, source_path: str, target_field: str, 
                          source_data: Dict[str, Any], schema_type: str):
        """เรียนรู้การ mapping ใหม่ - with memory management"""
        
        # Clean up memory periodically
        self._cleanup_memory()
        
        target_concept = self._identify_concept(target_field)
        
        if target_concept and target_concept in self.concepts:
            concept = self.concepts[target_concept]
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
        
        # Batch save to improve performance
        self._batch_save_count += 1
        if self._batch_save_count >= self._save_frequency:
            self._save_knowledge_base()
            self._batch_save_count = 0
    
    def _identify_concept(self, target_field: str) -> Optional[str]:
        """ระบุ concept ของ target field - ไม่เปลี่ยน"""
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
    
    def predict_field_mapping(self, source_path: str, source_value: Any, 
                            schema_type: str) -> List[Tuple[str, float]]:
        """ทำนายการ mapping โดยใช้ semantic knowledge - simplified"""
        
        predictions = []
        
        if not source_path:
            return predictions
            
        field_name = source_path.split('.')[-1].lower()
        
        # ค้นหาจาก semantic concepts
        for concept_name, concept in self.concepts.items():
            similarity_score = self._calculate_semantic_similarity(
                field_name, concept.field_patterns, concept.context_keywords
            )
            
            if similarity_score > 0.6:  # Lower threshold for more matches
                target_field = self._concept_to_target_field(concept_name)
                if target_field:
                    predictions.append((target_field, similarity_score * concept.confidence))
        
        # เรียงตาม confidence
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:5]  # คืนค่า top 5
    
    def _calculate_semantic_similarity(self, field_name: str, 
                                     patterns: List[str], 
                                     keywords: List[str]) -> float:
        """คำนวณความคล้ายคลึงเชิงความหมาย - ไม่เปลี่ยน"""
        
        if not field_name or not patterns:
            return 0.0
        
        pattern_score = 0.0
        for pattern in patterns:
            if pattern in field_name:
                pattern_score += 1.0
            elif field_name in pattern:
                pattern_score += 0.8
        
        pattern_score = min(pattern_score / len(patterns), 1.0)
        
        keyword_score = 0.0
        for keyword in keywords:
            if keyword in field_name:
                keyword_score += 0.5
        
        keyword_score = min(keyword_score, 1.0)
        
        return (pattern_score * 0.7) + (keyword_score * 0.3)
    
    def _concept_to_target_field(self, concept_name: str) -> Optional[str]:
        """แปลง concept เป็น target field - ไม่เปลี่ยน"""
        concept_mappings = {
            'alert_identification': 'alert_name',
            'threat_assessment': 'severity',
            'temporal_info': 'detected_time', 
            'identity_info': 'contexts.user',
            'asset_info': 'contexts.hostname',
            'network_info': 'source_ip'
        }
        
        return concept_mappings.get(concept_name)
    
    def _save_knowledge_base(self):
        """บันทึก knowledge base - with better error handling"""
        try:
            # Create backup first
            if self.knowledge_base_path.exists():
                backup_path = self.knowledge_base_path.with_suffix('.bak')
                import shutil
                shutil.copy2(self.knowledge_base_path, backup_path)
            
            # Ensure directory exists
            self.knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.knowledge_base_path, 'wb') as f:
                pickle.dump(self.concepts, f)
            
            logger.debug(f"Saved knowledge base with {len(self.concepts)} concepts")
            
        except Exception as e:
            logger.error(f"Could not save knowledge base: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """สถิติการเรียนรู้ - ไม่เปลี่ยน"""
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
        
        # Simple heuristic analysis
        try:
            content_text = json.dumps(data, default=str).lower()
        except Exception:
            content_text = str(data).lower()
        
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

class SemanticFeedbackCollector:
    """Collector ที่รวบรวม semantic feedback - simplified to avoid circular imports"""
    
    def __init__(self, semantic_engine: SemanticLearningEngine):
        self.semantic_engine = semantic_engine
    
    def collect_semantic_feedback(self, feedback_data: Dict[str, Any], 
                                source_data: Dict[str, Any]) -> Dict[str, Any]:
        """รวบรวม feedback พร้อม semantic analysis - without circular dependencies"""
        
        try:
            if feedback_data.get('corrected_source_path'):
                self.semantic_engine.learn_field_mapping(
                    feedback_data['corrected_source_path'],
                    feedback_data['field_name'],
                    source_data,
                    feedback_data['schema_type']
                )
            
            # Add semantic analysis without importing other modules
            enhanced_feedback = feedback_data.copy()
            enhanced_feedback['semantic_processed'] = True
            enhanced_feedback['timestamp'] = datetime.now().isoformat()
            
            return enhanced_feedback
            
        except Exception as e:
            logger.warning(f"Semantic feedback collection failed: {e}")
            return feedback_data

# Factory function to avoid circular imports
def create_semantic_engine() -> Optional[SemanticLearningEngine]:
    """Create semantic engine safely"""
    try:
        return SemanticLearningEngine()
    except Exception as e:
        logger.warning(f"Could not create semantic engine: {e}")
        return None

def create_semantic_collector(engine: SemanticLearningEngine) -> Optional[SemanticFeedbackCollector]:
    """Create semantic collector safely"""
    try:
        return SemanticFeedbackCollector(engine)
    except Exception as e:
        logger.warning(f"Could not create semantic collector: {e}")
        return None