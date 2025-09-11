import pytest
import pytest_asyncio
import asyncio
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from database.connection import DatabaseConnection
from database.schema import DatabaseSchema
from database.repository import ClaimsRepository, ProcessingStepsRepository, VectorQueriesRepository
from models.medical import PatientInfo, MedicalRecord
from models.claims import (
    PackageRecommendation, 
    ClaimSubmission, 
    RiskAssessment,
    ComplianceStatus, 
    SubmissionStatus
)


@pytest_asyncio.fixture
async def test_db():
    """Create a test database connection."""
    test_db_path = "./test_sanjeevni.db"
    db_connection = DatabaseConnection(test_db_path)
    
    # Create schema
    schema = DatabaseSchema(db_connection)
    await schema.create_all_tables()
    
    yield db_connection
    
    # Cleanup
    await db_connection.disconnect()
    Path(test_db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_claim():
    """Create a sample claim for testing."""
    patient = PatientInfo(
        patient_id="P12345",
        name="John Doe",
        age=45,
        gender="Male"
    )
    
    medical_record = MedicalRecord(
        patient_info=patient,
        hospital_id="H001",
        admission_date=datetime(2024, 1, 15),
        discharge_date=datetime(2024, 1, 20),
        procedures=[],
        diagnoses=[],
        total_amount=Decimal("50000.00"),
        document_confidence=0.95
    )
    
    recommendation = PackageRecommendation(
        package_code="HBP-001",
        package_name="Cardiac Surgery Package",
        confidence_score=0.92,
        estimated_amount=Decimal("75000.00"),
        approval_probability=0.85,
        compliance_status=ComplianceStatus.COMPLIANT
    )
    
    risk_assessment = RiskAssessment(
        risk_score=0.3,
        denial_probability=0.15,
        risk_factors=["Minor documentation gaps"],
        mitigation_suggestions=["Provide additional medical history"]
    )
    
    return ClaimSubmission(
        claim_id="C12345",
        hospital_id="H001",
        patient_data=medical_record,
        recommended_package=recommendation,
        risk_assessment=risk_assessment,
        submission_status=SubmissionStatus.PENDING
    )


class TestDatabaseConnection:
    """Test database connection functionality."""
    
    @pytest.mark.asyncio
    async def test_connection_health_check(self, test_db):
        """Test database health check."""
        health = await test_db.health_check()
        assert health is True
    
    @pytest.mark.asyncio
    async def test_connection_context_manager(self, test_db):
        """Test connection context manager."""
        async with test_db.get_connection() as conn:
            cursor = await conn.execute("SELECT 1")
            result = await cursor.fetchone()
            assert result[0] == 1


class TestDatabaseSchema:
    """Test database schema operations."""
    
    @pytest.mark.asyncio
    async def test_schema_creation(self, test_db):
        """Test schema creation."""
        schema = DatabaseSchema(test_db)
        await schema.create_all_tables()
        
        # Verify all tables exist
        assert await schema.table_exists("claims")
        assert await schema.table_exists("status_updates")
        assert await schema.table_exists("processing_steps")
        assert await schema.table_exists("vector_queries")
        assert await schema.table_exists("agent_metrics")
    
    @pytest.mark.asyncio
    async def test_schema_validation(self, test_db):
        """Test schema validation."""
        schema = DatabaseSchema(test_db)
        await schema.create_all_tables()
        
        validation_result = await schema.validate_schema()
        assert validation_result is True
    
    @pytest.mark.asyncio
    async def test_table_info(self, test_db):
        """Test getting table information."""
        schema = DatabaseSchema(test_db)
        await schema.create_all_tables()
        
        table_info = await schema.get_table_info("claims")
        assert len(table_info) > 0
        
        # Check for key columns
        column_names = [col["name"] for col in table_info]
        assert "claim_id" in column_names
        assert "hospital_id" in column_names
        assert "status" in column_names


class TestClaimsRepository:
    """Test claims repository operations."""
    
    @pytest.mark.asyncio
    async def test_create_claim(self, test_db, sample_claim):
        """Test creating a claim."""
        repo = ClaimsRepository(test_db)
        
        claim_id = await repo.create_claim(sample_claim)
        assert claim_id == sample_claim.claim_id
    
    @pytest.mark.asyncio
    async def test_get_claim(self, test_db, sample_claim):
        """Test retrieving a claim."""
        repo = ClaimsRepository(test_db)
        
        # Create claim first
        await repo.create_claim(sample_claim)
        
        # Retrieve claim
        retrieved_claim = await repo.get_claim(sample_claim.claim_id)
        assert retrieved_claim is not None
        assert retrieved_claim["claim_id"] == sample_claim.claim_id
        assert retrieved_claim["hospital_id"] == sample_claim.hospital_id
        assert retrieved_claim["status"] == sample_claim.submission_status.value
    
    @pytest.mark.asyncio
    async def test_update_claim_status(self, test_db, sample_claim):
        """Test updating claim status."""
        repo = ClaimsRepository(test_db)
        
        # Create claim first
        await repo.create_claim(sample_claim)
        
        # Update status
        success = await repo.update_claim_status(
            sample_claim.claim_id, 
            SubmissionStatus.SUBMITTED,
            "REF123456"
        )
        assert success is True
        
        # Verify update
        updated_claim = await repo.get_claim(sample_claim.claim_id)
        assert updated_claim["status"] == SubmissionStatus.SUBMITTED.value
        assert updated_claim["portal_reference"] == "REF123456"
    
    @pytest.mark.asyncio
    async def test_increment_retry_count(self, test_db, sample_claim):
        """Test incrementing retry count."""
        repo = ClaimsRepository(test_db)
        
        # Create claim first
        await repo.create_claim(sample_claim)
        
        # Increment retry count
        success = await repo.increment_retry_count(sample_claim.claim_id)
        assert success is True
        
        # Verify increment
        updated_claim = await repo.get_claim(sample_claim.claim_id)
        assert updated_claim["retry_count"] == 1
    
    @pytest.mark.asyncio
    async def test_get_claims_by_hospital(self, test_db, sample_claim):
        """Test getting claims by hospital."""
        repo = ClaimsRepository(test_db)
        
        # Create claim first
        await repo.create_claim(sample_claim)
        
        # Get claims by hospital
        claims = await repo.get_claims_by_hospital(sample_claim.hospital_id)
        assert len(claims) == 1
        assert claims[0]["claim_id"] == sample_claim.claim_id
    
    @pytest.mark.asyncio
    async def test_get_claims_by_status(self, test_db, sample_claim):
        """Test getting claims by status."""
        repo = ClaimsRepository(test_db)
        
        # Create claim first
        await repo.create_claim(sample_claim)
        
        # Get claims by status
        claims = await repo.get_claims_by_status(SubmissionStatus.PENDING)
        assert len(claims) == 1
        assert claims[0]["claim_id"] == sample_claim.claim_id


class TestProcessingStepsRepository:
    """Test processing steps repository."""
    
    @pytest.mark.asyncio
    async def test_log_step(self, test_db, sample_claim):
        """Test logging a processing step."""
        claims_repo = ClaimsRepository(test_db)
        steps_repo = ProcessingStepsRepository(test_db)
        
        # Create claim first
        await claims_repo.create_claim(sample_claim)
        
        # Log processing step
        start_time = datetime.now()
        step_id = await steps_repo.log_step(
            sample_claim.claim_id,
            "document_processing",
            "completed",
            start_time,
            metadata={"confidence": 0.95}
        )
        
        assert step_id is not None
        assert step_id > 0
    
    @pytest.mark.asyncio
    async def test_get_steps_for_claim(self, test_db, sample_claim):
        """Test getting processing steps for a claim."""
        claims_repo = ClaimsRepository(test_db)
        steps_repo = ProcessingStepsRepository(test_db)
        
        # Create claim first
        await claims_repo.create_claim(sample_claim)
        
        # Log processing step
        start_time = datetime.now()
        await steps_repo.log_step(
            sample_claim.claim_id,
            "document_processing",
            "completed",
            start_time
        )
        
        # Get steps
        steps = await steps_repo.get_steps_for_claim(sample_claim.claim_id)
        assert len(steps) == 1
        assert steps[0]["step_name"] == "document_processing"
        assert steps[0]["step_status"] == "completed"


class TestVectorQueriesRepository:
    """Test vector queries repository."""
    
    @pytest.mark.asyncio
    async def test_log_vector_query(self, test_db, sample_claim):
        """Test logging a vector query."""
        claims_repo = ClaimsRepository(test_db)
        vector_repo = VectorQueriesRepository(test_db)
        
        # Create claim first
        await claims_repo.create_claim(sample_claim)
        
        # Log vector query
        query_id = await vector_repo.log_vector_query(
            sample_claim.claim_id,
            "cardiac surgery guidelines",
            "pmjay_guidelines",
            [0.89, 0.87, 0.85],
            150.5,
            3
        )
        
        assert query_id is not None
        assert query_id > 0
    
    @pytest.mark.asyncio
    async def test_get_queries_for_claim(self, test_db, sample_claim):
        """Test getting vector queries for a claim."""
        claims_repo = ClaimsRepository(test_db)
        vector_repo = VectorQueriesRepository(test_db)
        
        # Create claim first
        await claims_repo.create_claim(sample_claim)
        
        # Log vector query
        await vector_repo.log_vector_query(
            sample_claim.claim_id,
            "cardiac surgery guidelines",
            "pmjay_guidelines",
            [0.89, 0.87, 0.85],
            150.5,
            3
        )
        
        # Get queries
        queries = await vector_repo.get_queries_for_claim(sample_claim.claim_id)
        assert len(queries) == 1
        assert queries[0]["query_text"] == "cardiac surgery guidelines"
        assert queries[0]["collection_name"] == "pmjay_guidelines"
