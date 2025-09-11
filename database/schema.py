"""Database schema definitions and migration utilities."""

import logging
from typing import List

from .connection import DatabaseConnection

logger = logging.getLogger(__name__)

# SQL DDL for creating tables
CREATE_CLAIMS_TABLE = """
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    hospital_id TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    patient_name TEXT NOT NULL,
    package_code TEXT,
    package_name TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    confidence_score REAL,
    risk_score REAL,
    approval_probability REAL,
    portal_reference TEXT,
    estimated_amount DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    chroma_query_time REAL,
    document_confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    retry_count INTEGER DEFAULT 0
);
"""

CREATE_STATUS_UPDATES_TABLE = """
CREATE TABLE IF NOT EXISTS status_updates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id TEXT NOT NULL,
    old_status TEXT,
    new_status TEXT NOT NULL,
    update_reason TEXT,
    updated_by TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id) REFERENCES claims (claim_id)
);
"""

CREATE_PROCESSING_STEPS_TABLE = """
CREATE TABLE IF NOT EXISTS processing_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id TEXT NOT NULL,
    step_name TEXT NOT NULL,
    step_status TEXT NOT NULL,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_ms REAL,
    error_message TEXT,
    metadata TEXT,  -- JSON string for additional data
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id) REFERENCES claims (claim_id)
);
"""

CREATE_VECTOR_QUERIES_TABLE = """
CREATE TABLE IF NOT EXISTS vector_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    similarity_scores TEXT,  -- JSON array of scores
    query_time_ms REAL,
    result_count INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id) REFERENCES claims (claim_id)
);
"""

CREATE_AGENT_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS agent_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    operation TEXT NOT NULL,
    execution_time_ms REAL,
    success BOOLEAN,
    error_message TEXT,
    input_size INTEGER,
    output_size INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id) REFERENCES claims (claim_id)
);
"""

# Indexes for better query performance
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_claims_hospital_id ON claims (hospital_id);",
    "CREATE INDEX IF NOT EXISTS idx_claims_status ON claims (status);",
    "CREATE INDEX IF NOT EXISTS idx_claims_created_at ON claims (created_at);",
    "CREATE INDEX IF NOT EXISTS idx_status_updates_claim_id ON status_updates (claim_id);",
    "CREATE INDEX IF NOT EXISTS idx_processing_steps_claim_id ON processing_steps (claim_id);",
    "CREATE INDEX IF NOT EXISTS idx_vector_queries_claim_id ON vector_queries (claim_id);",
    "CREATE INDEX IF NOT EXISTS idx_agent_metrics_claim_id ON agent_metrics (claim_id);",
]

# Complete schema creation script
FULL_SCHEMA = f"""
{CREATE_CLAIMS_TABLE}

{CREATE_STATUS_UPDATES_TABLE}

{CREATE_PROCESSING_STEPS_TABLE}

{CREATE_VECTOR_QUERIES_TABLE}

{CREATE_AGENT_METRICS_TABLE}

{chr(10).join(CREATE_INDEXES)}
"""


class DatabaseSchema:
    """Manages database schema creation and migrations."""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
    
    async def create_all_tables(self) -> None:
        """Create all database tables and indexes."""
        try:
            await self.db_connection.execute_script(FULL_SCHEMA)
            logger.info("All database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    async def drop_all_tables(self) -> None:
        """Drop all tables (for testing purposes)."""
        drop_script = """
        DROP TABLE IF EXISTS agent_metrics;
        DROP TABLE IF EXISTS vector_queries;
        DROP TABLE IF EXISTS processing_steps;
        DROP TABLE IF EXISTS status_updates;
        DROP TABLE IF EXISTS claims;
        """
        try:
            await self.db_connection.execute_script(drop_script)
            logger.info("All database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    async def get_table_info(self, table_name: str) -> List[dict]:
        """Get information about a specific table."""
        async with self.db_connection.get_connection() as conn:
            cursor = await conn.execute(f"PRAGMA table_info({table_name})")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        async with self.db_connection.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            result = await cursor.fetchone()
            return result is not None
    
    async def validate_schema(self) -> bool:
        """Validate that all required tables exist."""
        required_tables = [
            "claims", 
            "status_updates", 
            "processing_steps", 
            "vector_queries", 
            "agent_metrics"
        ]
        
        try:
            for table in required_tables:
                if not await self.table_exists(table):
                    logger.error(f"Required table '{table}' does not exist")
                    return False
            
            logger.info("Database schema validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False
