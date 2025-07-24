
import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import asdict

# It's better to import the dataclass itself to avoid circular dependencies
# but for simplicity in this step, we'll assume the structure.
# from practical_auto_learning import SuccessPattern

logger = logging.getLogger(__name__)

class LearningDatabase:
    """Manages the SQLite database for storing and retrieving learned patterns."""

    def __init__(self, db_file: str = "auto_learning_knowledge.db"):
        self.db_file = Path(db_file)
        self.conn = None
        self._connect()
        self._create_table()

    def _connect(self):
        """Establish a connection to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Successfully connected to learning database: {self.db_file}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database {self.db_file}: {e}")
            raise

    def _create_table(self):
        """Create the success_patterns table if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS success_patterns (
            pattern_key TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            target_field TEXT NOT NULL,
            schema_type TEXT NOT NULL,
            success_count INTEGER NOT NULL,
            total_attempts INTEGER NOT NULL,
            last_used TEXT NOT NULL,
            contexts TEXT
        );
        """
        try:
            with self.conn:
                self.conn.execute(create_table_sql)
        except sqlite3.Error as e:
            logger.error(f"Error creating table: {e}")

    def save_pattern(self, pattern_key: str, pattern_data: Dict):
        """Save or update a single success pattern."""
        sql = """
        INSERT OR REPLACE INTO success_patterns (
            pattern_key, source_path, target_field, schema_type, 
            success_count, total_attempts, last_used, contexts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        try:
            # Convert list of contexts to a JSON string for storage
            contexts_json = json.dumps(pattern_data.get('contexts', []))
            
            params = (
                pattern_key,
                pattern_data['source_path'],
                pattern_data['target_field'],
                pattern_data['schema_type'],
                pattern_data['success_count'],
                pattern_data['total_attempts'],
                pattern_data['last_used'],
                contexts_json
            )
            with self.conn:
                self.conn.execute(sql, params)
        except sqlite3.Error as e:
            logger.error(f"Failed to save pattern {pattern_key}: {e}")

    def load_all_patterns(self) -> Dict:
        """Load all success patterns from the database."""
        sql = "SELECT * FROM success_patterns;"
        patterns = {}
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            for row in cursor.fetchall():
                pattern_key = row['pattern_key']
                patterns[pattern_key] = {
                    "source_path": row["source_path"],
                    "target_field": row["target_field"],
                    "schema_type": row["schema_type"],
                    "success_count": row["success_count"],
                    "total_attempts": row["total_attempts"],
                    "last_used": row["last_used"],
                    # Convert JSON string back to a list
                    "contexts": json.loads(row["contexts"])
                }
            logger.info(f"Loaded {len(patterns)} patterns from the database.")
            return patterns
        except sqlite3.Error as e:
            logger.error(f"Failed to load patterns from database: {e}")
            return {}

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")
