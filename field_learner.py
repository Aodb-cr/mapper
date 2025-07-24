
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re
import logging

from learning_database import LearningDatabase # Import the actual database class

logger = logging.getLogger(__name__)

@dataclass
class FieldKnowledge:
    """ความรู้เกี่ยวกับ field - ใช้สำหรับ internal representation เท่านั้น"""
    field_name: str
    semantic_type: str  # 'identity', 'network', 'temporal', 'threat', 'asset'
    common_patterns: List[str] = field(default_factory=list)
    value_examples: List[str] = field(default_factory=list)
    confidence: float = 0.0
    usage_count: int = 0
    
    def add_example(self, value: Any):
        """เพิ่มตัวอย่างค่า"""
        str_value = str(value)[:100]
        if str_value not in self.value_examples:
            self.value_examples.append(str_value)
            if len(self.value_examples) > 10:
                self.value_examples.pop(0)

class SmartFieldLearner:
    """ระบบเรียนรู้ field และความหมาย - เชื่อมต่อกับ LearningDatabase โดยตรง"""
    
    def __init__(self, db: LearningDatabase):
        self.db = db # Use the provided LearningDatabase instance
        self.field_knowledge: Dict[str, FieldKnowledge] = {}
        self.semantic_patterns = self._init_semantic_patterns()
        self._lock = threading.RLock()
        # No _load_knowledge() here, as knowledge is loaded via load_all_knowledge for extractor
        
    def _init_semantic_patterns(self) -> Dict[str, Dict[str, Any]]:
        """เริ่มต้น semantic patterns"""
        return {
            'identity': {
                'keywords': ['user', 'account', 'login', 'name', 'id', 'email'],
                'patterns': [r'user.*', r'.*_id$', r'.*name$', r'account.*'],
                'value_patterns': [r'^[a-zA-Z][a-zA-Z0-9_]*$', r'^.+@.+\..+$']
            },
            'network': {
                'keywords': ['ip', 'port', 'address', 'host', 'domain'],
                'patterns': [r'.*ip.*', r'.*port.*', r'.*host.*'],
                'value_patterns': [r'^\d+\.\d+\.\d+\.\d+$', r':\d+$']
            },
            'temporal': {
                'keywords': ['time', 'date', 'timestamp', 'when', 'created'],
                'patterns': [r'.*time.*', r'.*date.*', r'.*timestamp.*'],
                'value_patterns': [r'\d{4}-\d{2}-\d{2}', r'\d{10,13}']
            },
            'threat': {
                'keywords': ['threat', 'malware', 'virus', 'severity', 'risk'],
                'patterns': [r'threat.*', r'.*severity.*', r'malware.*'],
                'value_patterns': [r'(high|medium|low)', r'(critical|warning|info)']
            },
            'asset': {
                'keywords': ['device', 'computer', 'machine', 'hostname'],
                'patterns': [r'device.*', r'.*host.*', r'machine.*'],
                'value_patterns': [r'^[a-zA-Z][a-zA-Z0-9-]*$']
            }
        }
    
    def analyze_field(self, field_path: str, value: Any) -> Optional[str]:
        """วิเคราะห์ semantic type ของ field"""
        if not field_path or value is None:
            return None
            
        field_lower = field_path.lower()
        value_str = str(value).lower()
        
        scores = defaultdict(float)
        
        for semantic_type, patterns in self.semantic_patterns.items():
            # ตรวจ keywords
            for keyword in patterns['keywords']:
                if keyword in field_lower:
                    scores[semantic_type] += 2.0
            
            # ตรวจ patterns
            for pattern in patterns['patterns']:
                if re.search(pattern, field_lower):
                    scores[semantic_type] += 1.5
            
            # ตรวจ value patterns
            for pattern in patterns['value_patterns']:
                if re.search(pattern, value_str, re.IGNORECASE):
                    scores[semantic_type] += 1.0
        
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            if best_type[1] >= 1.0:
                return best_type[0]
        
        return 'unknown'
    
    def learn_field(self, field_path: str, value: Any, schema_type: str = 'unknown'):
        """เรียนรู้ field ใหม่ - บันทึกลง DB ผ่าน PracticalAutoLearner"""
        # This method is primarily for internal learning within the extractor
        # Manual teaching goes through PracticalAutoLearner.manual_teach_pattern
        # For now, we'll just update internal knowledge, not persist here.
        with self._lock:
            semantic_type = self.analyze_field(field_path, value)
            
            if field_path not in self.field_knowledge:
                self.field_knowledge[field_path] = FieldKnowledge(
                    field_name=field_path,
                    semantic_type=semantic_type or 'unknown',
                    confidence=0.5 if semantic_type else 0.1
                )
            
            knowledge = self.field_knowledge[field_path]
            knowledge.usage_count += 1
            knowledge.add_example(value)
            
            if semantic_type and semantic_type != 'unknown':
                if knowledge.semantic_type == 'unknown' or knowledge.confidence < 0.7:
                    knowledge.semantic_type = semantic_type
                    knowledge.confidence = min(knowledge.confidence + 0.1, 0.9)
    
    def suggest_mapping(self, source_field: str, target_fields: List[str]) -> List[Tuple[str, float]]:
        """แนะนำ field mapping"""
        suggestions = []
        
        if source_field not in self.field_knowledge:
            return suggestions
        
        source_knowledge = self.field_knowledge[source_field]
        
        for target_field in target_fields:
            score = 0.0
            
            # ตรวจ semantic type
            if target_field in self.field_knowledge:
                target_knowledge = self.field_knowledge[target_field]
                if source_knowledge.semantic_type == target_knowledge.semantic_type:
                    score += 3.0 * source_knowledge.confidence
            
            # ตรวจ keyword similarity
            if self._keyword_similarity(source_field, target_field) > 0.5:
                score += 2.0
            
            if score >= 1.0:
                suggestions.append((target_field, min(score / 5.0, 1.0)))
        
        return sorted(suggestions, key=lambda x: x[1], reverse=True)[:3]
    
    def _keyword_similarity(self, field1: str, field2: str) -> float:
        """คำนวณความคล้ายคลึงของ keywords"""
        words1 = set(re.findall(r'\w+', field1.lower()))
        words2 = set(re.findall(r'\w+', field2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    # Removed _load_knowledge and save_knowledge as persistence is handled by LearningDatabase
    
    def get_stats(self) -> Dict[str, Any]:
        """สถิติการเรียนรู้"""
        semantic_types = Counter(k.semantic_type for k in self.field_knowledge.values())
        return {
            'total_fields': len(self.field_knowledge),
            'semantic_distribution': dict(semantic_types),
            'high_confidence': len([k for k in self.field_knowledge.values() if k.confidence > 0.7])
        }

    def load_all_knowledge(self) -> Dict[str, List[Tuple[str, float]]]:
        """Loads all learned patterns from the database for the extractor."""
        try:
            all_patterns_from_db = self.db.load_all_patterns() # Load from the actual DB
            if not all_patterns_from_db:
                logger.info("No learned patterns found in the database.")
                return {}

            knowledge_for_extractor = defaultdict(list)
            for key, pattern_data in all_patterns_from_db.items():
                source_path = pattern_data.get('source_path')
                target_field = pattern_data.get('target_field')
                success_count = pattern_data.get('success_count', 0)
                total_attempts = pattern_data.get('total_attempts', 1)

                # Calculate confidence score
                if total_attempts > 0:
                    success_rate = success_count / total_attempts
                    if total_attempts < 3:
                        confidence = success_rate * 0.6
                    elif total_attempts < 5:
                        confidence = success_rate * 0.8
                    else:
                        confidence = min(success_rate * 1.0, 0.95)
                else:
                    confidence = 0.0

                if source_path and target_field and confidence > 0.5: # Only load reasonably confident patterns
                    knowledge_for_extractor[target_field].append(
                        (source_path, confidence)
                    )
            
            logger.info(f"Successfully loaded {len(all_patterns_from_db)} patterns from the database for extractor.")
            return dict(knowledge_for_extractor)

        except Exception as e:
            logger.error(f"Could not load knowledge from database for extractor: {e}")
            return {}
