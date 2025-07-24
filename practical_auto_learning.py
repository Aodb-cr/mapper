import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from datetime import datetime
import logging

from safe_utils import SafeDataProcessor
from core_config import get_config
from learning_database import LearningDatabase
from field_learner import SmartFieldLearner # Import SmartFieldLearner

logger = logging.getLogger(__name__)

@dataclass
class SuccessPattern:
    """Pattern ที่สำเร็จ - เก็บง่ายๆ"""
    source_path: str
    target_field: str
    schema_type: str
    success_count: int
    total_attempts: int
    last_used: str
    contexts: List[str]

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_attempts if self.total_attempts > 0 else 0.0

    @property
    def confidence(self) -> float:
        """คำนวณ confidence จาก success rate และ sample size"""
        if self.total_attempts < 3:
            return self.success_rate * 0.6
        elif self.total_attempts < 5:
            return self.success_rate * 0.8
        else:
            return min(self.success_rate * 1.0, 0.95)

class PracticalAutoLearner:
    """Auto-Learner ที่ใช้ได้จริงกับโปรเจคคุณ"""

    def __init__(self, db_file: str = "auto_learning_knowledge.db"):
        self.db = LearningDatabase(db_file)
        self.config = get_config()
        
        self.success_patterns: Dict[str, SuccessPattern] = {}
        self.schema_signatures = {}
        
        self.min_confidence_auto_apply = 0.40
        self.learning_enabled = True
        self.max_patterns_per_field = 10
        
        self._load_knowledge()
        
        logger.info(f"Auto-Learner initialized with {len(self.success_patterns)} patterns from DB.")

    def analyze_successful_extraction(self, 
                                    extraction_results: Dict[str, Any],
                                    source_data: Dict[str, Any],
                                    schema_type: str,
                                    quality_score: float) -> Dict[str, Any]:
        """วิเคราะห์การ extraction ที่สำเร็จเพื่อเรียนรู้"""
        if not self.learning_enabled or quality_score < 0.4:
            return {'learned': False, 'reason': 'quality_too_low'}
        
        learned_patterns = []
        
        for field_name, result in extraction_results.items():
            source_path = getattr(result, 'source_path', None)
            confidence = getattr(result, 'confidence', 0.0)
            value = getattr(result, 'value', None)
            
            if source_path and source_path != 'not_found' and value is not None and confidence > 0.6:
                pattern_learned = self._record_successful_pattern(
                    source_path, field_name, schema_type, source_data, confidence
                )
                
                if pattern_learned:
                    learned_patterns.append({
                        'field': field_name,
                        'source_path': source_path,
                        'confidence': confidence
                    })
        
        self._update_schema_signature(source_data, schema_type)
        
        if learned_patterns:
            self._save_knowledge()
            logger.info(f"Learned and saved {len(learned_patterns)} new patterns to database.")
        
        return {
            'learned': len(learned_patterns) > 0,
            'patterns': learned_patterns,
            'quality_score': quality_score
        }

    def _record_successful_pattern(self, 
                                 source_path: str,
                                 target_field: str, 
                                 schema_type: str,
                                 source_data: Dict[str, Any],
                                 confidence: float) -> bool:
        """บันทึก pattern ที่สำเร็จ"""
        pattern_key = f"{schema_type}_{target_field}_{source_path}"
        context = self._create_simple_context(source_data, source_path)
        
        if pattern_key in self.success_patterns:
            pattern = self.success_patterns[pattern_key]
            pattern.success_count += 1
            pattern.total_attempts += 1
            pattern.last_used = datetime.now().isoformat()
            if len(pattern.contexts) < 5:
                pattern.contexts.append(context)
        else:
            self.success_patterns[pattern_key] = SuccessPattern(
                source_path=source_path,
                target_field=target_field,
                schema_type=schema_type,
                success_count=1,
                total_attempts=1,
                last_used=datetime.now().isoformat(),
                contexts=[context]
            )
        
        self._limit_patterns_per_field(target_field, schema_type)
        return True

    def _create_simple_context(self, source_data: Dict[str, Any], source_path: str) -> str:
        """สร้าง context signature ง่ายๆ"""
        context_info = []
        path_parts = source_path.split('.')
        if len(path_parts) > 1:
            parent_path = '.'.join(path_parts[:-1])
            parent_obj = SafeDataProcessor.safe_get_nested(source_data, parent_path)
            if isinstance(parent_obj, dict):
                nearby_fields = list(parent_obj.keys())[:5]
                context_info.extend(nearby_fields)
        if isinstance(source_data, dict):
            top_keys = list(source_data.keys())[:3]
            context_info.extend(top_keys)
        return json.dumps(sorted(context_info))

    def _limit_patterns_per_field(self, target_field: str, schema_type: str):
        """จำกัดจำนวน patterns ต่อ field เพื่อไม่ให้ขยายไม่หยุด"""
        field_patterns = [
            (key, pattern) for key, pattern in self.success_patterns.items()
            if pattern.target_field == target_field and pattern.schema_type == schema_type
        ]
        
        if len(field_patterns) > self.max_patterns_per_field:
            field_patterns.sort(key=lambda x: x[1].confidence, reverse=True)
            patterns_to_remove = field_patterns[self.max_patterns_per_field:]
            for key, _ in patterns_to_remove:
                del self.success_patterns[key]
            logger.debug(f"Limited patterns for {target_field}: kept {self.max_patterns_per_field}")

    def predict_better_extraction(self, 
                                available_fields: List[str],
                                target_fields: List[str],
                                schema_type: str) -> Dict[str, List[Tuple[str, float]]]:
        """ทำนาย field mappings ที่ดีกว่าจาก patterns ที่เรียนรู้"""
        predictions = {}
        for target_field in target_fields:
            field_predictions = []
            relevant_patterns = [
                pattern for pattern in self.success_patterns.values()
                if (pattern.target_field == target_field and 
                    pattern.schema_type == schema_type and
                    pattern.source_path in available_fields)
            ]
            relevant_patterns.sort(key=lambda p: p.confidence, reverse=True)
            for pattern in relevant_patterns[:3]:
                if pattern.confidence >= 0.5:
                    field_predictions.append((pattern.source_path, pattern.confidence))
            if field_predictions:
                predictions[target_field] = field_predictions
        return predictions

    def enhance_field_extractor(self, field_extractor):
        """เสริม field extractor ด้วย learned patterns"""
        if not hasattr(field_extractor, 'field_mappings'):
            logger.warning("Field extractor ไม่มี field_mappings attribute")
            return
        
        enhanced_count = 0
        for pattern in self.success_patterns.values():
            if pattern.confidence >= self.min_confidence_auto_apply:
                if pattern.target_field in field_extractor.field_mappings:
                    mapping = field_extractor.field_mappings[pattern.target_field]
                    if pattern.source_path not in mapping.learned_patterns:
                        mapping.learned_patterns.insert(0, pattern.source_path)
                        enhanced_count += 1
                        if len(mapping.learned_patterns) > 15:
                            mapping.learned_patterns = mapping.learned_patterns[:15]
        
        if enhanced_count > 0:
            logger.info(f"Enhanced field extractor with {enhanced_count} learned patterns.")

    def _load_knowledge(self):
        """Load knowledge from the database."""
        patterns_from_db = self.db.load_all_patterns()
        for key, data in patterns_from_db.items():
            self.success_patterns[key] = SuccessPattern(**data)

    def _save_knowledge(self):
        """Save all success patterns to the database."""
        for key, pattern in self.success_patterns.items():
            self.db.save_pattern(key, asdict(pattern))

    # ... (the rest of the methods remain the same) ...
    def _update_schema_signature(self, source_data: Dict[str, Any], schema_type: str):
        """อัปเดต signature ของ schema"""
        
        # สร้าง signature ง่ายๆ
        signature = {
            'top_level_keys': list(source_data.keys()) if isinstance(source_data, dict) else [],
            'structure_depth': self._calculate_depth(source_data),
            'field_count': self._count_total_fields(source_data),
            'has_arrays': self._has_arrays(source_data)
        }
        
        if schema_type not in self.schema_signatures:
            self.schema_signatures[schema_type] = []
        
        self.schema_signatures[schema_type].append(signature)
        
        # เก็บแค่ 10 signatures ล่าสุด
        if len(self.schema_signatures[schema_type]) > 10:
            self.schema_signatures[schema_type] = self.schema_signatures[schema_type][-10:]
    
    def suggest_schema_improvements(self, schema_type: str) -> List[str]:
        """แนะนำการปรับปรุง schema detection"""
        
        suggestions = []
        
        if schema_type in self.schema_signatures:
            signatures = self.schema_signatures[schema_type]
            
            # วิเคราะห์ common patterns
            common_keys = Counter()
            for sig in signatures:
                for key in sig.get('top_level_keys', []):
                    common_keys[key] += 1
            
            # แนะนำ keys ที่เกิดขึ้นบ่อย
            frequent_keys = [key for key, count in common_keys.items() if count >= len(signatures) * 0.7]
            
            if frequent_keys:
                suggestions.append(f"Consider adding these frequent keys to schema signature: {frequent_keys}")
            
            # วิเคราะห์ structure patterns
            avg_depth = sum(sig['structure_depth'] for sig in signatures) / len(signatures)
            avg_fields = sum(sig['field_count'] for sig in signatures) / len(signatures)
            
            suggestions.append(f"Typical structure: depth={avg_depth:.1f}, fields={avg_fields:.0f}")
        
        return suggestions
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """สถิติการเรียนรู้"""
        
        total_patterns = len(self.success_patterns)
        high_confidence = len([p for p in self.success_patterns.values() if p.confidence >= 0.8])
        
        # สถิติตาม schema
        schema_stats = defaultdict(int)
        for pattern in self.success_patterns.values():
            schema_stats[pattern.schema_type] += 1
        
        # สถิติตาม field
        field_stats = defaultdict(int)
        for pattern in self.success_patterns.values():
            field_stats[pattern.target_field] += 1
        
        return {
            'total_patterns': total_patterns,
            'high_confidence_patterns': high_confidence,
            'schemas_learned': len(schema_stats),
            'top_schemas': dict(Counter(schema_stats).most_common(5)),
            'top_fields': dict(Counter(field_stats).most_common(10)),
            'auto_apply_threshold': self.min_confidence_auto_apply,
            'knowledge_file_size': self.db.db_file.stat().st_size if self.db.db_file.exists() else 0
        }
    
    def _calculate_depth(self, obj: Any, current_depth: int = 0) -> int:
        """คำนวณความลึกของ structure"""
        if current_depth > 5:  # จำกัดเพื่อป้องกัน infinite recursion
            return current_depth
            
        max_depth = current_depth
        
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    depth = self._calculate_depth(value, current_depth + 1)
                    max_depth = max(max_depth, depth)
        elif isinstance(obj, list) and obj:
            for item in obj[:3]:  # ตรวจแค่ 3 items แรก
                if isinstance(item, (dict, list)):
                    depth = self._calculate_depth(item, current_depth + 1)
                    max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _count_total_fields(self, obj: Any) -> int:
        """นับจำนวน fields ทั้งหมด"""
        count = 0
        
        if isinstance(obj, dict):
            count += len(obj)
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    count += self._count_total_fields(value)
        elif isinstance(obj, list):
            for item in obj[:3]:  # ตรวจแค่ 3 items แรก
                if isinstance(item, (dict, list)):
                    count += self._count_total_fields(item)
        
        return count
    
    def _has_arrays(self, obj: Any) -> bool:
        """ตรวจว่ามี arrays หรือไม่"""
        if isinstance(obj, list):
            return True
        elif isinstance(obj, dict):
            for value in obj.values():
                if self._has_arrays(value):
                    return True
        return False

class AutoLearningIntegrator:
    """ตัวเชื่อมระบบ Auto-Learning เข้ากับ Smart Mapper ที่มีอยู่"""
    
    def __init__(self, smart_mapper_core):
        self.smart_mapper = smart_mapper_core
        self.auto_learner = PracticalAutoLearner()
        
        logger.info(" Auto-Learning Integrator initialized")
    
    def enhanced_process_file(self, 
                            input_file: str,
                            output_file: Optional[str] = None,
                            options: Optional[Any] = None) -> Dict[str, Any]:
        """ประมวลผลไฟล์ด้วยระบบ Auto-Learning"""
        
        # 1. เพิ่ม learned patterns เข้าใน field extractor
        field_extractor = get_field_extractor()
        if field_extractor:
            self.auto_learner.enhance_field_extractor(field_extractor)
        
        # 2. ทำการ process ปกติ
        result = self.smart_mapper.process_file(input_file, output_file, options)
        
        # 3. ถ้าสำเร็จและมี quality ดี ให้เรียนรู้
        if result.success and result.overall_quality_score >= 70.0:
            
            # โหลดข้อมูลต้นฉบับเพื่อวิเคราะห์
            try:
                load_result = self.smart_mapper._load_input_file(input_file)
                if load_result.success:
                    source_data = load_result.data
                    
                    # สมมติว่ามี extraction results ใน metadata
                    extraction_results = result.metadata.get('extraction_results', {})
                    
                    # เรียนรู้จากผลลัพธ์ที่ดี
                    learning_result = self.auto_learner.analyze_successful_extraction(
                        extraction_results,
                        source_data,
                        result.metadata.get('schema', 'unknown'),
                        result.overall_quality_score
                    )
                    
                    # เพิ่มข้อมูลการเรียนรู้เข้าไปใน result
                    result.metadata['auto_learning'] = learning_result
                    
            except Exception as e:
                logger.warning(f"Auto-learning failed: {e}")
        
        return result
    
    def get_intelligence_report(self) -> Dict[str, Any]:
        """รายงานความฉลาดของระบบ"""
        
        learning_stats = self.auto_learner.get_learning_stats()
        
        report = {
            'auto_learning_enabled': self.auto_learner.learning_enabled,
            'learning_statistics': learning_stats,
            'improvement_suggestions': [],
            'next_steps': []
        }
        
        # แนะนำการปรับปรุง
        if learning_stats['total_patterns'] < 10:
            report['next_steps'].append("Process more files to build learning knowledge")
        
        if learning_stats['high_confidence_patterns'] < learning_stats['total_patterns'] * 0.5:
            report['improvement_suggestions'].append("Consider adjusting confidence thresholds")
        
        return report
    
    def manual_teach_pattern(self, 
                           field_name: str,
                           source_path: str, 
                           schema_type: str,
                           context_data: Dict[str, Any]):
        """สอนระบบด้วยตัวเอง (สำหรับ edge cases)"""
        
        pattern_learned = self.auto_learner._record_successful_pattern(
            source_path, field_name, schema_type, context_data, 0.9  # high confidence
        )
        
        if pattern_learned:
            self.auto_learner._save_knowledge()
            logger.info(f" Manually taught pattern: {field_name} → {source_path}")
            return True
        
        return False