"""Database repository for CRUD operations on claims and related data."""

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from models.claims import ClaimSubmission, SubmissionStatus
from .connection import DatabaseConnection

logger = logging.getLogger(__name__)


class ClaimsRepository:
    """Repository for claims data operations."""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
    
    async def create_claim(self, claim: ClaimSubmission) -> str:
        """Create a new claim record."""
        async with self.db_connection.get_connection() as conn:
            try:
                await conn.execute("""
                    INSERT INTO claims (
                        claim_id, hospital_id, patient_id, patient_name,
                        package_code, package_name, status, confidence_score,
                        risk_score, approval_probability, portal_reference,
                        estimated_amount, total_amount, document_confidence,
                        created_at, updated_at, retry_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    claim.claim_id,
                    claim.hospital_id,
                    claim.patient_data.patient_info.patient_id,
                    claim.patient_data.patient_info.name,
                    claim.recommended_package.package_code,
                    claim.recommended_package.package_name,
                    claim.submission_status.value,
                    claim.recommended_package.confidence_score,
                    claim.risk_assessment.risk_score if claim.risk_assessment else None,
                    claim.recommended_package.approval_probability,
                    claim.portal_reference,
                    float(claim.recommended_package.estimated_amount),
                    float(claim.patient_data.total_amount),
                    claim.patient_data.document_confidence,
                    claim.created_at,
                    claim.updated_at,
                    claim.retry_count
                ))
                await conn.commit()
                logger.info(f"Created claim: {claim.claim_id}")
                return claim.claim_id
                
            except Exception as e:
                logger.error(f"Failed to create claim {claim.claim_id}: {e}")
                raise
    
    async def get_claim(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a claim by ID."""
        async with self.db_connection.get_connection() as conn:
            try:
                cursor = await conn.execute(
                    "SELECT * FROM claims WHERE claim_id = ?", (claim_id,)
                )
                row = await cursor.fetchone()
                return dict(row) if row else None
                
            except Exception as e:
                logger.error(f"Failed to get claim {claim_id}: {e}")
                raise
    
    async def update_claim_status(
        self, 
        claim_id: str, 
        new_status: SubmissionStatus,
        portal_reference: Optional[str] = None
    ) -> bool:
        """Update claim status and optionally portal reference."""
        async with self.db_connection.get_connection() as conn:
            try:
                # Get current status for audit trail
                cursor = await conn.execute(
                    "SELECT status FROM claims WHERE claim_id = ?", (claim_id,)
                )
                current_row = await cursor.fetchone()
                if not current_row:
                    logger.warning(f"Claim {claim_id} not found for status update")
                    return False
                
                old_status = current_row[0]
                
                # Update claim
                update_query = """
                    UPDATE claims 
                    SET status = ?, updated_at = ?, portal_reference = COALESCE(?, portal_reference)
                    WHERE claim_id = ?
                """
                await conn.execute(update_query, (
                    new_status.value, datetime.now(), portal_reference, claim_id
                ))
                
                # Log status change
                await conn.execute("""
                    INSERT INTO status_updates (claim_id, old_status, new_status, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (claim_id, old_status, new_status.value, datetime.now()))
                
                await conn.commit()
                logger.info(f"Updated claim {claim_id} status: {old_status} -> {new_status.value}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update claim {claim_id} status: {e}")
                raise
    
    async def increment_retry_count(self, claim_id: str) -> bool:
        """Increment the retry count for a claim."""
        async with self.db_connection.get_connection() as conn:
            try:
                await conn.execute("""
                    UPDATE claims 
                    SET retry_count = retry_count + 1, updated_at = ?
                    WHERE claim_id = ?
                """, (datetime.now(), claim_id))
                await conn.commit()
                logger.info(f"Incremented retry count for claim {claim_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to increment retry count for claim {claim_id}: {e}")
                raise
    
    async def get_claims_by_hospital(self, hospital_id: str) -> List[Dict[str, Any]]:
        """Get all claims for a specific hospital."""
        async with self.db_connection.get_connection() as conn:
            try:
                cursor = await conn.execute(
                    "SELECT * FROM claims WHERE hospital_id = ? ORDER BY created_at DESC",
                    (hospital_id,)
                )
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
                
            except Exception as e:
                logger.error(f"Failed to get claims for hospital {hospital_id}: {e}")
                raise
    
    async def get_claims_by_status(self, status: SubmissionStatus) -> List[Dict[str, Any]]:
        """Get all claims with a specific status."""
        async with self.db_connection.get_connection() as conn:
            try:
                cursor = await conn.execute(
                    "SELECT * FROM claims WHERE status = ? ORDER BY created_at DESC",
                    (status.value,)
                )
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
                
            except Exception as e:
                logger.error(f"Failed to get claims with status {status.value}: {e}")
                raise


class ProcessingStepsRepository:
    """Repository for processing steps tracking."""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
    
    async def log_step(
        self,
        claim_id: str,
        step_name: str,
        step_status: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Log a processing step."""
        async with self.db_connection.get_connection() as conn:
            try:
                duration_ms = None
                if start_time and end_time:
                    duration_ms = (end_time - start_time).total_seconds() * 1000
                
                metadata_json = json.dumps(metadata) if metadata else None
                
                cursor = await conn.execute("""
                    INSERT INTO processing_steps (
                        claim_id, step_name, step_status, start_time, end_time,
                        duration_ms, error_message, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    claim_id, step_name, step_status, start_time, end_time,
                    duration_ms, error_message, metadata_json
                ))
                await conn.commit()
                
                step_id = cursor.lastrowid
                logger.info(f"Logged processing step {step_name} for claim {claim_id}")
                return step_id
                
            except Exception as e:
                logger.error(f"Failed to log processing step: {e}")
                raise
    
    async def get_steps_for_claim(self, claim_id: str) -> List[Dict[str, Any]]:
        """Get all processing steps for a claim."""
        async with self.db_connection.get_connection() as conn:
            try:
                cursor = await conn.execute(
                    "SELECT * FROM processing_steps WHERE claim_id = ? ORDER BY timestamp",
                    (claim_id,)
                )
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
                
            except Exception as e:
                logger.error(f"Failed to get processing steps for claim {claim_id}: {e}")
                raise


class VectorQueriesRepository:
    """Repository for vector query logging."""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
    
    async def log_vector_query(
        self,
        claim_id: str,
        query_text: str,
        collection_name: str,
        similarity_scores: List[float],
        query_time_ms: float,
        result_count: int
    ) -> int:
        """Log a vector database query."""
        async with self.db_connection.get_connection() as conn:
            try:
                scores_json = json.dumps(similarity_scores)
                
                cursor = await conn.execute("""
                    INSERT INTO vector_queries (
                        claim_id, query_text, collection_name, similarity_scores,
                        query_time_ms, result_count
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    claim_id, query_text, collection_name, scores_json,
                    query_time_ms, result_count
                ))
                await conn.commit()
                
                query_id = cursor.lastrowid
                logger.info(f"Logged vector query for claim {claim_id}")
                return query_id
                
            except Exception as e:
                logger.error(f"Failed to log vector query: {e}")
                raise
    
    async def get_queries_for_claim(self, claim_id: str) -> List[Dict[str, Any]]:
        """Get all vector queries for a claim."""
        async with self.db_connection.get_connection() as conn:
            try:
                cursor = await conn.execute(
                    "SELECT * FROM vector_queries WHERE claim_id = ? ORDER BY timestamp",
                    (claim_id,)
                )
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
                
            except Exception as e:
                logger.error(f"Failed to get vector queries for claim {claim_id}: {e}")
                raise
