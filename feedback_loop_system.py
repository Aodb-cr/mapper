#!/usr/bin/env python3
"""
feedback_loop_system.py - Fixed AI Learning System from User Feedback
แก้ไข: Circular imports, database concurrency, memory management

Changes:
1. Removed circular import dependencies
2. Added database connection pooling and thread safety
3. Better memory management and cleanup
4. Simplified complex logic
"""

import json
import sqlite3
import threading
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging

from core_config import get_config
from safe_utils import SafeOperationHandler, OperationType, OperationResult

logger = logging.getLogger(__name__)

@dataclass
class FeedbackRecord:
    """บันทึกการแก้ไขของ user - ไม่เปลี่ยน"""
    id: Optional[int] = None
    timestamp: str = ""
    field_name: str = ""
    schema_type: str = ""
    original_value: Any = None
    corrected_value: Any = None
    original_source_path: str = ""
    corrected_source_path: str = ""
    confidence_before: float = 0.0
    extraction_method: str = ""
    file_source: str = ""
    user_action: str = ""  # "corrected", "confirmed", "added"
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class LearnedPattern:
    """Pattern ที่เรียนรู้จาก feedback - ไม่เปลี่ยน"""
    pattern_id: str
    field_name: str
    schema_type: str
    source_pattern: str
    success_count: int
    total_count: int
    confidence: float
    learned_from: List[str]  # feedback IDs
    last_updated: str = ""
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()
    
    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_count if self.total_count > 0 else 0.0

class FeedbackDatabase:
    """ฐานข้อมูลสำหรับเก็บ feedback และ learned patterns - Fixed thread safety"""
    
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        self._connection_pool = {}
        self._max_connections = 5
        
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-safe database connection"""
        thread_id = threading.get_ident()
        
        if thread_id not in self._connection_pool:
            if len(self._connection_pool) >= self._max_connections:
                # Clean up old connections
                oldest_thread = min(self._connection_pool.keys())
                try:
                    self._connection_pool[oldest_thread].close()
                except:
                    pass
                del self._connection_pool[oldest_thread]
            
            # Create new connection
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
            conn.execute("PRAGMA synchronous=NORMAL")  # Better performance
            self._connection_pool[thread_id] = conn
        
        return self._connection_pool[thread_id]
    
    def _init_database(self):
        """สร้างตารางฐานข้อมูล - ไม่เปลี่ยน"""
        with self._lock:
            conn = self._get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    schema_type TEXT NOT NULL,
                    original_value TEXT,
                    corrected_value TEXT,
                    original_source_path TEXT,
                    corrected_source_path TEXT,
                    confidence_before REAL,
                    extraction_method TEXT,
                    file_source TEXT,
                    user_action TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    field_name TEXT NOT NULL,
                    schema_type TEXT NOT NULL,
                    source_pattern TEXT NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    total_count INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0.0,
                    learned_from TEXT,
                    last_updated TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_field_schema 
                ON feedback_records(field_name, schema_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_field_schema 
                ON learned_patterns(field_name, schema_type)
            """)
            
            conn.commit()
    
    def save_feedback(self, feedback: FeedbackRecord) -> int:
        """บันทึก feedback record - with transaction safety"""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("""
                    INSERT INTO feedback_records (
                        timestamp, field_name, schema_type, original_value, corrected_value,
                        original_source_path, corrected_source_path, confidence_before,
                        extraction_method, file_source, user_action
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.timestamp, feedback.field_name, feedback.schema_type,
                    json.dumps(feedback.original_value, default=str), 
                    json.dumps(feedback.corrected_value, default=str),
                    feedback.original_source_path, feedback.corrected_source_path,
                    feedback.confidence_before, feedback.extraction_method,
                    feedback.file_source, feedback.user_action
                ))
                conn.commit()
                return cursor.lastrowid
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to save feedback: {e}")
                raise
    
    def get_feedback_history(self, field_name: str = None, schema_type: str = None, 
                           days: int = 30) -> List[FeedbackRecord]:
        """ดึง feedback history - with limits"""
        with self._lock:
            where_conditions = ["datetime(timestamp) >= datetime('now', '-{} days')".format(days)]
            params = []
            
            if field_name:
                where_conditions.append("field_name = ?")
                params.append(field_name)
            
            if schema_type:
                where_conditions.append("schema_type = ?")
                params.append(schema_type)
            
            query = """
                SELECT * FROM feedback_records 
                WHERE {} 
                ORDER BY timestamp DESC
                LIMIT 1000
            """.format(" AND ".join(where_conditions))
            
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            
            return [
                FeedbackRecord(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    field_name=row['field_name'],
                    schema_type=row['schema_type'],
                    original_value=self._safe_json_loads(row['original_value']),
                    corrected_value=self._safe_json_loads(row['corrected_value']),
                    original_source_path=row['original_source_path'],
                    corrected_source_path=row['corrected_source_path'],
                    confidence_before=row['confidence_before'],
                    extraction_method=row['extraction_method'],
                    file_source=row['file_source'],
                    user_action=row['user_action']
                )
                for row in rows
            ]
    
    def _safe_json_loads(self, value: str) -> Any:
        """Safely load JSON value"""
        try:
            return json.loads(value or 'null')
        except:
            return None
    
    def save_learned_pattern(self, pattern: LearnedPattern):
        """บันทึก learned pattern - with transaction safety"""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO learned_patterns (
                        pattern_id, field_name, schema_type, source_pattern,
                        success_count, total_count, confidence, learned_from, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_id, pattern.field_name, pattern.schema_type,
                    pattern.source_pattern, pattern.success_count, pattern.total_count,
                    pattern.confidence, json.dumps(pattern.learned_from), pattern.last_updated
                ))
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to save learned pattern: {e}")
                raise
    
    def get_learned_patterns(self, field_name: str = None, schema_type: str = None) -> List[LearnedPattern]:
        """ดึง learned patterns - with limits"""
        with self._lock:
            where_conditions = []
            params = []
            
            if field_name:
                where_conditions.append("field_name = ?")
                params.append(field_name)
            
            if schema_type:
                where_conditions.append("schema_type = ?")
                params.append(schema_type)
            
            query = "SELECT * FROM learned_patterns"
            if where_conditions:
                query += " WHERE " + " AND ".join(where_conditions)
            query += " ORDER BY confidence DESC LIMIT 500"
            
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            
            return [
                LearnedPattern(
                    pattern_id=row['pattern_id'],
                    field_name=row['field_name'],
                    schema_type=row['schema_type'],
                    source_pattern=row['source_pattern'],
                    success_count=row['success_count'],
                    total_count=row['total_count'],
                    confidence=row['confidence'],
                    learned_from=self._safe_json_loads(row['learned_from']) or [],
                    last_updated=row['last_updated']
                )
                for row in rows
            ]
    
    def cleanup_old_records(self, days: int = 90):
        """Clean up old records to prevent database bloat"""
        with self._lock:
            conn = self._get_connection()
            try:
                # Clean old feedback records
                conn.execute("""
                    DELETE FROM feedback_records 
                    WHERE datetime(timestamp) < datetime('now', '-{} days')
                """.format(days))
                
                # Clean old learned patterns with low confidence
                conn.execute("""
                    DELETE FROM learned_patterns 
                    WHERE confidence < 0.3 AND datetime(last_updated) < datetime('now', '-{} days')
                """.format(days // 2))  # More aggressive cleanup for low confidence patterns
                
                conn.commit()
                logger.info("Cleaned up old database records")
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to cleanup database: {e}")
    
    def close(self):
        """Close all database connections"""
        with self._lock:
            for conn in self._connection_pool.values():
                try:
                    conn.close()
                except:
                    pass
            self._connection_pool.clear()

class FeedbackReviewInterface:
    """Interface สำหรับ review และแก้ไขผลลัพธ์ - simplified"""
    
    def __init__(self, feedback_db: FeedbackDatabase):
        self.db = feedback_db
    
    def review_mapping_results(self, source_data: Dict[str, Any], 
                             mapped_results: Dict[str, Any],
                             schema_type: str,
                             file_source: str = "") -> List[FeedbackRecord]:
        """Review และรับ feedback จาก user - simplified version"""
        
        print(f"\n🔍 Review Mapping Results for {schema_type}")
        print("=" * 60)
        
        feedback_records = []
        
        # แสดงผลลัพธ์ที่ได้ (limited for performance)
        flattened_results = self._flatten_dict(mapped_results)
        
        # Limit review to important fields only
        important_fields = ['alert_name', 'severity', 'detected_time', 'log_source', 'contexts.hostname']
        review_fields = {k: v for k, v in flattened_results.items() 
                        if any(important in k for important in important_fields)}
        
        for field_path, value in list(review_fields.items())[:10]:  # Limit to 10 fields
            if field_path == 'rawAlert':
                continue
                
            print(f"\n📝 Field: {field_path}")
            print(f"   Value: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
            
            # ให้ user review (simplified)
            action = self._get_user_feedback_simple(field_path, value, source_data)
            
            if action['type'] != 'skip':
                feedback = FeedbackRecord(
                    field_name=field_path,
                    schema_type=schema_type,
                    original_value=value,
                    corrected_value=action.get('corrected_value'),
                    original_source_path=action.get('original_path', ''),
                    corrected_source_path=action.get('corrected_path', ''),
                    confidence_before=action.get('confidence', 0.0),
                    extraction_method=action.get('method', 'unknown'),
                    file_source=file_source,
                    user_action=action['type']
                )
                
                feedback_records.append(feedback)
                
                # บันทึกลงฐานข้อมูล
                try:
                    feedback.id = self.db.save_feedback(feedback)
                    logger.info(f"Feedback recorded: {field_path} - {action['type']}")
                except Exception as e:
                    logger.error(f"Failed to save feedback: {e}")
        
        return feedback_records
    
    def _get_user_feedback_simple(self, field_name: str, current_value: Any, 
                                source_data: Dict[str, Any]) -> Dict[str, Any]:
        """รับ feedback จาก user สำหรับ field เดียว - simplified"""
        
        while True:
            print(f"\nActions for '{field_name}':")
            print("1. Correct")
            print("2. Wrong (provide correction)")
            print("3. Skip")
            
            choice = input("เลือก (1-3): ").strip()
            
            if choice == '1':
                return {'type': 'confirmed', 'confidence': 1.0}
            
            elif choice == '2':
                corrected = input(f"แก้ไขค่าใหม่: ").strip()
                if corrected:
                    return {
                        'type': 'corrected',
                        'corrected_value': corrected,
                        'confidence': 0.0
                    }
                else:
                    print("ไม่ได้ใส่ค่าใหม่")
            
            elif choice == '3':
                return {'type': 'skip'}
            
            else:
                print("ตัวเลือกไม่ถูกต้อง")
    
    def _flatten_dict(self, data: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """แปลง nested dict เป็น flat dict - simplified"""
        result = {}
        
        def flatten_recursive(obj: Any, current_prefix: str, depth: int = 0):
            if depth > 5:  # Limit recursion depth
                return
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{current_prefix}.{key}" if current_prefix else key
                    
                    if isinstance(value, dict):
                        flatten_recursive(value, new_key, depth + 1)
                    elif isinstance(value, list) and value and isinstance(value[0], dict):
                        flatten_recursive(value[0], f"{new_key}[0]", depth + 1)
                    else:
                        result[new_key] = value
        
        flatten_recursive(data, prefix)
        return result

class PatternLearningEngine:
    """Engine สำหรับเรียนรู้ patterns จาก feedback - simplified"""
    
    def __init__(self, feedback_db: FeedbackDatabase):
        self.db = feedback_db
        self.config = get_config()
    
    def analyze_feedback_patterns(self, min_occurrences: int = 3) -> List[LearnedPattern]:
        """วิเคราะห์และเรียนรู้ patterns จาก feedback"""
        
        logger.info("Analyzing feedback patterns...")
        
        # ดึง feedback records ล่าสุด
        feedback_records = self.db.get_feedback_history(days=30)
        
        # จัดกลุ่มตาม field และ schema
        pattern_candidates = defaultdict(list)
        
        for record in feedback_records:
            if record.user_action in ['corrected', 'added'] and record.corrected_source_path:
                key = (record.field_name, record.schema_type)
                pattern_candidates[key].append(record)
        
        learned_patterns = []
        
        for (field_name, schema_type), records in pattern_candidates.items():
            # นับความถี่ของแต่ละ source path
            path_counter = Counter(r.corrected_source_path for r in records)
            
            for source_path, count in path_counter.items():
                if count >= min_occurrences:
                    # สร้าง learned pattern
                    pattern_id = f"{field_name}_{schema_type}_{hash(source_path)}"
                    
                    # คำนวณ confidence จาก success rate
                    successful_records = [r for r in records if r.corrected_source_path == source_path]
                    confidence = min(0.9, count / len(records))  # max 90%
                    
                    pattern = LearnedPattern(
                        pattern_id=pattern_id,
                        field_name=field_name,
                        schema_type=schema_type,
                        source_pattern=source_path,
                        success_count=count,
                        total_count=len(records),
                        confidence=confidence,
                        learned_from=[str(r.id) for r in successful_records]
                    )
                    
                    learned_patterns.append(pattern)
                    
                    # บันทึกลงฐานข้อมูล
                    try:
                        self.db.save_learned_pattern(pattern)
                        logger.info(f"Learned pattern: {field_name} -> {source_path} (confidence: {confidence:.2f})")
                    except Exception as e:
                        logger.error(f"Failed to save learned pattern: {e}")
        
        return learned_patterns
    
    def update_field_mappings(self, field_extractor=None) -> int:
        """อัปเดต field mappings ด้วย learned patterns - without circular imports"""
        
        if field_extractor is None:
            # Use lazy import to avoid circular dependency
            try:
                from universal_field_extractor import SmartFieldExtractor
                field_extractor = SmartFieldExtractor()
            except ImportError as e:
                logger.error(f"Cannot import SmartFieldExtractor: {e}")
                return 0
        
        learned_patterns = self.db.get_learned_patterns()
        updates_count = 0
        
        for pattern in learned_patterns:
            if pattern.confidence >= 0.7:  # threshold สำหรับ auto-update
                try:
                    field_mapping = field_extractor.field_mappings.get(pattern.field_name)
                    
                    if field_mapping:
                        # เพิ่ม learned pattern เข้าไปใน source patterns
                        if pattern.source_pattern not in field_mapping.source_patterns:
                            # เพิ่มที่ด้านหน้าเพื่อให้ priority สูง
                            field_mapping.source_patterns.insert(0, pattern.source_pattern)
                            updates_count += 1
                            
                            logger.info(f"Updated mapping for {pattern.field_name}: added {pattern.source_pattern}")
                except Exception as e:
                    logger.warning(f"Failed to update field mapping for {pattern.field_name}: {e}")
        
        return updates_count
    
    def get_field_performance_report(self) -> Dict[str, Any]:
        """สร้างรายงานประสิทธิภาพของแต่ละ field"""
        
        feedback_records = self.db.get_feedback_history(days=30)
        
        # วิเคราะห์ประสิทธิภาพ
        field_stats = defaultdict(lambda: {
            'total_extractions': 0,
            'corrections': 0,
            'confirmations': 0,
            'accuracy': 0.0
        })
        
        for record in feedback_records:
            stats = field_stats[record.field_name]
            stats['total_extractions'] += 1
            
            if record.user_action == 'corrected':
                stats['corrections'] += 1
            elif record.user_action == 'confirmed':
                stats['confirmations'] += 1
        
        # คำนวณ accuracy
        for field_name, stats in field_stats.items():
            total = stats['total_extractions']
            if total > 0:
                stats['accuracy'] = stats['confirmations'] / total
        
        return dict(field_stats)

class FeedbackLoopManager:
    """Manager หลักสำหรับ Feedback Loop System - simplified"""
    
    def __init__(self, db_path: str = "feedback.db"):
        self.db = FeedbackDatabase(db_path)
        self.review_interface = FeedbackReviewInterface(self.db)
        self.learning_engine = PatternLearningEngine(self.db)
        self.handler = SafeOperationHandler()
        
        # Periodic cleanup
        self._last_cleanup = datetime.now()
        self._cleanup_interval = timedelta(days=7)
    
    def process_with_feedback(self, source_data: Dict[str, Any],
                            mapped_results: Dict[str, Any],
                            schema_type: str,
                            file_source: str = "",
                            auto_learn: bool = True) -> Dict[str, Any]:
        """ประมวลผลพร้อม feedback collection"""
        
        # Collect feedback
        feedback_records = self.review_interface.review_mapping_results(
            source_data, mapped_results, schema_type, file_source
        )
        
        logger.info(f"Collected {len(feedback_records)} feedback records")
        
        # Auto-learn จาก feedback ใหม่
        if auto_learn and feedback_records:
            self.trigger_learning_cycle()
        
        # Periodic cleanup
        self._periodic_cleanup()
        
        return mapped_results
    
    def trigger_learning_cycle(self, field_extractor=None) -> Dict[str, Any]:
        """เรียกใช้ learning cycle"""
        
        logger.info("Starting learning cycle...")
        
        try:
            # วิเคราะห์และเรียนรู้ patterns
            learned_patterns = self.learning_engine.analyze_feedback_patterns()
            
            # อัปเดต field mappings (ส่ง field_extractor จากภายนอก)
            updates_count = self.learning_engine.update_field_mappings(field_extractor)
            
            # สร้างรายงาน
            performance_report = self.learning_engine.get_field_performance_report()
            
            result = {
                'learned_patterns': len(learned_patterns),
                'mapping_updates': updates_count,
                'performance_report': performance_report,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Learning cycle completed: {len(learned_patterns)} patterns, {updates_count} updates")
            
            return result
            
        except Exception as e:
            logger.error(f"Learning cycle failed: {e}")
            return {'error': str(e)}
    
    def _periodic_cleanup(self):
        """Periodic database cleanup"""
        current_time = datetime.now()
        if current_time - self._last_cleanup > self._cleanup_interval:
            try:
                self.db.cleanup_old_records()
                self._last_cleanup = current_time
                gc.collect()  # Force garbage collection
            except Exception as e:
                logger.error(f"Periodic cleanup failed: {e}")
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """ดึงสถิติ feedback"""
        
        try:
            feedback_records = self.db.get_feedback_history(days=30)
            learned_patterns = self.db.get_learned_patterns()
            
            return {
                'total_feedback_records': len(feedback_records),
                'learned_patterns': len(learned_patterns),
                'recent_activity': len([r for r in feedback_records if 
                                      (datetime.now() - datetime.fromisoformat(r.timestamp)).days <= 7]),
                'top_corrected_fields': self._get_top_corrected_fields(feedback_records)
            }
        except Exception as e:
            logger.error(f"Failed to get feedback statistics: {e}")
            return {'error': str(e)}
    
    def _get_top_corrected_fields(self, feedback_records: List[FeedbackRecord]) -> List[Tuple[str, int]]:
        """หา fields ที่ถูกแก้ไขบ่อยที่สุด"""
        
        corrections = Counter()
        for record in feedback_records:
            if record.user_action == 'corrected':
                corrections[record.field_name] += 1
        
        return corrections.most_common(5)  # Top 5 only
    
    def close(self):
        """Clean shutdown"""
        try:
            self.db.close()
        except Exception as e:
            logger.error(f"Error closing feedback database: {e}")

# CLI Interface สำหรับทดสอบ - simplified
def main():
    """CLI สำหรับทดสอบ Feedback Loop System"""
    
    try:
        manager = FeedbackLoopManager()
        
        while True:
            print(f"\n🔄 Feedback Loop System")
            print("-" * 30)
            print("1. Trigger Learning Cycle")
            print("2. View Feedback Statistics")
            print("3. Exit")
            
            choice = input("\nเลือกตัวเลือก (1-3): ").strip()
            
            if choice == '1':
                result = manager.trigger_learning_cycle()
                print(f"Learning cycle completed:")
                print(f"   Learned patterns: {result.get('learned_patterns', 0)}")
                print(f"   Mapping updates: {result.get('mapping_updates', 0)}")
            
            elif choice == '2':
                stats = manager.get_feedback_statistics()
                print(f" Feedback Statistics:")
                print(f"   Total records: {stats.get('total_feedback_records', 0)}")
                print(f"   Learned patterns: {stats.get('learned_patterns', 0)}")
                print(f"   Recent activity: {stats.get('recent_activity', 0)}")
            
            elif choice == '3':
                break
            
            else:
                print("ตัวเลือกไม่ถูกต้อง")
        
        manager.close()
        
    except KeyboardInterrupt:
        print("\n Interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()