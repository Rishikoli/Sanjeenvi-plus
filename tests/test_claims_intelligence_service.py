"""
Unit tests for Claims Intelligence Service

Tests the core claims processing workflow including:
- Document processing integration
- RAG search and analysis
- Package recommendations
- Risk assessment
- Compliance checking
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from services.claims_intelligence_service import ClaimsIntelligenceService
from services.document_processing_service import DocumentProcessingService, DocumentProcessingResult, OCRResult
from services.chroma_service import ChromaService
from services.granite_service import GraniteLanguageDetectionService
from services.granite_embedding_service import GraniteEmbeddingService
from database.repository import ClaimsRepository, ProcessingStepsRepository, VectorQueriesRepository
from models.medical import PatientInfo, MedicalRecord, MedicalProcedure, Diagnosis
from models.claims import PackageRecommendation, RiskAssessment, ComplianceStatus, SubmissionStatus


# Module-level fixtures to be accessible across classes
@pytest_asyncio.fixture
async def mock_services():
    """Create mock services for testing (module-level)."""
    document_service = Mock(spec=DocumentProcessingService)
    chroma_service = Mock(spec=ChromaService)
    language_service = Mock(spec=GraniteLanguageDetectionService)
    embedding_service = Mock(spec=GraniteEmbeddingService)
    claims_repository = Mock(spec=ClaimsRepository)
    steps_repository = Mock(spec=ProcessingStepsRepository)
    vector_repository = Mock(spec=VectorQueriesRepository)

    # Configure async methods
    document_service.process_document = AsyncMock()
    document_service.health_check = AsyncMock()
    claims_repository.create_claim = AsyncMock()
    claims_repository.get_claim = AsyncMock()
    claims_repository.update_claim_status = AsyncMock()
    claims_repository.increment_retry_count = AsyncMock()
    steps_repository.log_step = AsyncMock()
    steps_repository.get_steps_for_claim = AsyncMock()
    vector_repository.log_vector_query = AsyncMock()
    vector_repository.get_queries_for_claim = AsyncMock()
    embedding_service.health_check = AsyncMock()

    return {
        "document_service": document_service,
        "chroma_service": chroma_service,
        "language_service": language_service,
        "embedding_service": embedding_service,
        "claims_repository": claims_repository,
        "steps_repository": steps_repository,
        "vector_repository": vector_repository,
    }


@pytest.fixture
def sample_medical_record():
    """Module-level sample medical record for performance tests."""
    patient = PatientInfo(
        patient_id="P12345",
        name="John Doe",
        age=45,
        gender="Male"
    )

    procedure = MedicalProcedure(
        procedure_name="Cardiac Bypass Surgery",
        procedure_code="CABG-001",
        procedure_date=datetime(2024, 1, 16)
    )

    diagnosis = Diagnosis(
        diagnosis_name="Coronary Artery Disease",
        diagnosis_code="I25.1",
        diagnosis_date=datetime(2024, 1, 15)
    )

    return MedicalRecord(
        patient_info=patient,
        hospital_id="H001",
        admission_date=datetime(2024, 1, 15),
        discharge_date=datetime(2024, 1, 20),
        procedures=[procedure],
        diagnoses=[diagnosis],
        total_amount=Decimal("150000.00"),
        document_confidence=0.92
    )


@pytest.fixture
def sample_ocr_result(sample_medical_record):
    """Module-level OCR result used by performance tests."""
    ocr_result = OCRResult(
        text="Sample medical document text",
        confidence=0.92,
        language="english",
        processing_time_ms=1500.0,
        page_count=2,
        method="docling",
        metadata={"pages": 2}
    )

    return DocumentProcessingResult(
        ocr_result=ocr_result,
        extracted_data=None,
        medical_record=sample_medical_record,
        validation_errors=[],
        requires_verification=False,
        processing_summary={"success": True, "confidence": 0.92}
    )


@pytest.fixture
def sample_search_results():
    """Module-level search results used by performance tests."""
    return {
        "pmjay_results": [
            {
                "document": "PM-JAY cardiac surgery package guidelines",
                "metadata": {
                    "package_code": "CARD-001",
                    "package_name": "Cardiac Surgery Package",
                    "doc_id": 1
                },
                "distance": 0.2
            },
            {
                "document": "Coronary artery disease treatment protocol",
                "metadata": {
                    "package_code": "CARD-002",
                    "package_name": "CAD Treatment Package",
                    "doc_id": 2
                },
                "distance": 0.3
            }
        ],
        "medical_codes_results": [
            {
                "document": "ICD-10 code I25.1 for coronary artery disease",
                "metadata": {"code": "I25.1", "category": "cardiovascular"},
                "distance": 0.1
            }
        ],
        "search_queries": [
            "PM-JAY package for Coronary Artery Disease",
            "Medical package Cardiac Bypass Surgery"
        ]
    }


class TestClaimsIntelligenceService:
    """Test suite for Claims Intelligence Service."""
    
    @pytest_asyncio.fixture
    async def mock_services(self):
        """Create mock services for testing."""
        document_service = Mock(spec=DocumentProcessingService)
        chroma_service = Mock(spec=ChromaService)
        language_service = Mock(spec=GraniteLanguageDetectionService)
        embedding_service = Mock(spec=GraniteEmbeddingService)
        claims_repository = Mock(spec=ClaimsRepository)
        steps_repository = Mock(spec=ProcessingStepsRepository)
        vector_repository = Mock(spec=VectorQueriesRepository)
        
        # Configure async methods
        document_service.process_document = AsyncMock()
        document_service.health_check = AsyncMock()
        claims_repository.create_claim = AsyncMock()
        claims_repository.get_claim = AsyncMock()
        claims_repository.update_claim_status = AsyncMock()
        claims_repository.increment_retry_count = AsyncMock()
        steps_repository.log_step = AsyncMock()
        steps_repository.get_steps_for_claim = AsyncMock()
        vector_repository.log_vector_query = AsyncMock()
        vector_repository.get_queries_for_claim = AsyncMock()
        embedding_service.health_check = AsyncMock()
        
        return {
            "document_service": document_service,
            "chroma_service": chroma_service,
            "language_service": language_service,
            "embedding_service": embedding_service,
            "claims_repository": claims_repository,
            "steps_repository": steps_repository,
            "vector_repository": vector_repository
        }
    
    @pytest_asyncio.fixture
    async def intelligence_service(self, mock_services):
        """Create Claims Intelligence Service with mocked dependencies."""
        return ClaimsIntelligenceService(
            document_service=mock_services["document_service"],
            chroma_service=mock_services["chroma_service"],
            language_service=mock_services["language_service"],
            embedding_service=mock_services["embedding_service"],
            claims_repository=mock_services["claims_repository"],
            steps_repository=mock_services["steps_repository"],
            vector_repository=mock_services["vector_repository"]
        )
    
    @pytest.fixture
    def sample_medical_record(self):
        """Create a sample medical record for testing."""
        patient = PatientInfo(
            patient_id="P12345",
            name="John Doe",
            age=45,
            gender="Male"
        )
        
        procedure = MedicalProcedure(
            procedure_name="Cardiac Bypass Surgery",
            procedure_code="CABG-001",
            procedure_date=datetime(2024, 1, 16)
        )
        
        diagnosis = Diagnosis(
            diagnosis_name="Coronary Artery Disease",
            diagnosis_code="I25.1",
            diagnosis_date=datetime(2024, 1, 15)
        )
        
        return MedicalRecord(
            patient_info=patient,
            hospital_id="H001",
            admission_date=datetime(2024, 1, 15),
            discharge_date=datetime(2024, 1, 20),
            procedures=[procedure],
            diagnoses=[diagnosis],
            total_amount=Decimal("150000.00"),
            document_confidence=0.92
        )
    
    @pytest.fixture
    def sample_ocr_result(self, sample_medical_record):
        """Create a sample OCR result."""
        ocr_result = OCRResult(
            text="Sample medical document text",
            confidence=0.92,
            language="english",
            processing_time_ms=1500.0,
            page_count=2,
            method="docling",
            metadata={"pages": 2}
        )
        
        return DocumentProcessingResult(
            ocr_result=ocr_result,
            extracted_data=None,
            medical_record=sample_medical_record,
            validation_errors=[],
            requires_verification=False,
            processing_summary={"success": True, "confidence": 0.92}
        )
    
    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results."""
        return {
            "pmjay_results": [
                {
                    "document": "PM-JAY cardiac surgery package guidelines",
                    "metadata": {
                        "package_code": "CARD-001",
                        "package_name": "Cardiac Surgery Package",
                        "doc_id": 1
                    },
                    "distance": 0.2
                },
                {
                    "document": "Coronary artery disease treatment protocol",
                    "metadata": {
                        "package_code": "CARD-002", 
                        "package_name": "CAD Treatment Package",
                        "doc_id": 2
                    },
                    "distance": 0.3
                }
            ],
            "medical_codes_results": [
                {
                    "document": "ICD-10 code I25.1 for coronary artery disease",
                    "metadata": {"code": "I25.1", "category": "cardiovascular"},
                    "distance": 0.1
                }
            ],
            "search_queries": ["PM-JAY package for Coronary Artery Disease", "Medical package Cardiac Bypass Surgery"]
        }
    
    @pytest.mark.asyncio
    async def test_process_claim_document_success(self, intelligence_service, mock_services, sample_ocr_result, sample_search_results):
        """Test successful claim document processing."""
        # Setup mocks
        mock_services["document_service"].process_document.return_value = sample_ocr_result
        mock_services["language_service"].detect_language.return_value = {
            "detected_language": "english",
            "confidence": 0.95,
            "processing_time_ms": 100
        }
        mock_services["chroma_service"].search_pmjay_guidelines.return_value = sample_search_results["pmjay_results"]
        mock_services["chroma_service"].search_medical_codes.return_value = sample_search_results["medical_codes_results"]
        
        # Execute
        result = await intelligence_service.process_claim_document(
            document_path="test_document.pdf",
            hospital_id="H001"
        )
        
        # Verify
        assert result["processing_status"] == "completed"
        assert "claim_id" in result
        assert result["document_analysis"]["ocr_confidence"] == 0.92
        assert result["document_analysis"]["language_detected"] == "english"
        assert len(result["package_recommendations"]) >= 0
        assert "risk_assessment" in result
        assert "compliance_status" in result
        
        # Verify service calls
        mock_services["document_service"].process_document.assert_called_once()
        mock_services["language_service"].detect_language.assert_called_once()
        mock_services["claims_repository"].create_claim.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_claim_document_ocr_failure(self, intelligence_service, mock_services):
        """Test claim processing with OCR failure."""
        # Setup mock to return failed OCR
        failed_ocr_result = OCRResult(
            text="",
            confidence=0.0,
            language="unknown",
            processing_time_ms=500.0,
            page_count=0,
            method="docling",
            metadata={"error": "Failed to process document"}
        )
        
        failed_processing = DocumentProcessingResult(
            ocr_result=failed_ocr_result,
            extracted_data=None,
            medical_record=None,
            validation_errors=["Failed to process document"],
            requires_verification=True,
            processing_summary={"success": False, "error": "Failed to process document"}
        )
        mock_services["document_service"].process_document.return_value = failed_processing
        
        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            await intelligence_service.process_claim_document(
                document_path="invalid_document.pdf",
                hospital_id="H001"
            )
        
        assert "Document processing failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_package_recommendations(self, intelligence_service, sample_medical_record, sample_search_results):
        """Test package recommendation generation."""
        # Execute private method (for testing purposes)
        recommendations = await intelligence_service._generate_package_recommendations(
            claim_id="TEST_001",
            medical_data=sample_medical_record,
            search_results=sample_search_results
        )
        
        # Verify
        assert len(recommendations) > 0
        assert all(isinstance(rec, PackageRecommendation) for rec in recommendations)
        assert all(rec.confidence_score >= 0.0 for rec in recommendations)
        assert all(rec.approval_probability >= 0.0 for rec in recommendations)
        
        # Check sorting by confidence
        if len(recommendations) > 1:
            assert recommendations[0].confidence_score >= recommendations[1].confidence_score
    
    @pytest.mark.asyncio
    async def test_perform_risk_assessment(self, intelligence_service, sample_medical_record):
        """Test risk assessment functionality."""
        # Create sample recommendations
        recommendation = PackageRecommendation(
            package_code="CARD-001",
            package_name="Cardiac Surgery Package",
            confidence_score=0.85,
            estimated_amount=Decimal("150000.00"),
            approval_probability=0.80,
            risk_factors=["High complexity surgery"],
            compliance_status=ComplianceStatus.COMPLIANT,
            chroma_similarity_scores=[0.85, 0.78]
        )
        
        # Execute
        risk_assessment = await intelligence_service._perform_risk_assessment(
            claim_id="TEST_001",
            medical_data=sample_medical_record,
            recommendations=[recommendation],
            search_results={}
        )
        
        # Verify
        assert isinstance(risk_assessment, RiskAssessment)
        assert 0.0 <= risk_assessment.overall_risk_score <= 1.0
        assert risk_assessment.risk_level in ["LOW", "MEDIUM", "HIGH"]
        assert isinstance(risk_assessment.risk_factors, list)
        assert risk_assessment.recommendation in ["AUTO_APPROVE", "MANUAL_REVIEW"]
    
    @pytest.mark.asyncio
    async def test_check_compliance_compliant(self, intelligence_service, sample_medical_record):
        """Test compliance checking with compliant data."""
        recommendations = [
            PackageRecommendation(
                package_code="CARD-001",
                package_name="Cardiac Surgery Package",
                confidence_score=0.85,
                estimated_amount=Decimal("150000.00"),
                approval_probability=0.80,
                risk_factors=[],
                compliance_status=ComplianceStatus.COMPLIANT,
                chroma_similarity_scores=[0.85]
            )
        ]
        
        # Execute
        compliance_status = await intelligence_service._check_compliance(
            medical_data=sample_medical_record,
            recommendations=recommendations,
            search_results={}
        )
        
        # Verify
        assert compliance_status == ComplianceStatus.COMPLIANT
    
    @pytest.mark.asyncio
    async def test_check_compliance_non_compliant(self, intelligence_service):
        """Test compliance checking with non-compliant data."""
        # Create medical record with compliance issues
        patient = PatientInfo(
            patient_id="",  # Missing patient ID
            name="John Doe",
            age=45,
            gender="Male"
        )
        
        medical_record = MedicalRecord(
            patient_info=patient,
            hospital_id="H001",
            admission_date=datetime(2024, 1, 20),  # Invalid: discharge before admission
            discharge_date=datetime(2024, 1, 15),
            procedures=[],  # No procedures
            diagnoses=[],   # No diagnoses
            total_amount=Decimal("0.00"),  # Invalid amount
            document_confidence=0.92
        )
        
        # Execute
        compliance_status = await intelligence_service._check_compliance(
            medical_data=medical_record,
            recommendations=[],
            search_results={}
        )
        
        # Verify
        assert compliance_status == ComplianceStatus.NON_COMPLIANT
    
    @pytest.mark.asyncio
    async def test_get_claim_analysis_existing_claim(self, intelligence_service, mock_services):
        """Test retrieving analysis for existing claim."""
        claim_id = "TEST_001"
        
        # Setup mocks
        mock_services["claims_repository"].get_claim.return_value = {"claim_id": claim_id}
        mock_services["steps_repository"].get_steps_for_claim.return_value = [
            {"step_name": "document_processing", "status": "completed"}
        ]
        mock_services["vector_repository"].get_queries_for_claim.return_value = [
            {"query_text": "test query", "results_count": 5}
        ]
        
        # Execute
        result = await intelligence_service.get_claim_analysis(claim_id)
        
        # Verify
        assert result is not None
        assert "claim" in result
        assert "processing_steps" in result
        assert "vector_queries" in result
        
        # Verify service calls
        mock_services["claims_repository"].get_claim.assert_called_once_with(claim_id)
        mock_services["steps_repository"].get_steps_for_claim.assert_called_once_with(claim_id)
        mock_services["vector_repository"].get_queries_for_claim.assert_called_once_with(claim_id)
    
    @pytest.mark.asyncio
    async def test_get_claim_analysis_nonexistent_claim(self, intelligence_service, mock_services):
        """Test retrieving analysis for non-existent claim."""
        claim_id = "NONEXISTENT"
        
        # Setup mock to return None
        mock_services["claims_repository"].get_claim.return_value = None
        
        # Execute
        result = await intelligence_service.get_claim_analysis(claim_id)
        
        # Verify
        assert result is None
    
    @pytest.mark.asyncio
    async def test_reprocess_claim_success(self, intelligence_service, mock_services):
        """Test successful claim reprocessing."""
        claim_id = "TEST_001"
        
        # Setup mock
        mock_services["claims_repository"].get_claim.return_value = {"claim_id": claim_id}
        
        # Execute
        result = await intelligence_service.reprocess_claim(claim_id)
        
        # Verify
        assert result["claim_id"] == claim_id
        assert result["status"] == "reprocessed"
        
        # Verify service calls
        mock_services["claims_repository"].get_claim.assert_called_once_with(claim_id)
        mock_services["claims_repository"].increment_retry_count.assert_called_once_with(claim_id)
        mock_services["claims_repository"].update_claim_status.assert_called_once_with(
            claim_id, SubmissionStatus.PROCESSING
        )
    
    @pytest.mark.asyncio
    async def test_reprocess_claim_not_found(self, intelligence_service, mock_services):
        """Test reprocessing non-existent claim."""
        claim_id = "NONEXISTENT"
        
        # Setup mock to return None
        mock_services["claims_repository"].get_claim.return_value = None
        
        # Execute and verify exception
        with pytest.raises(ValueError) as exc_info:
            await intelligence_service.reprocess_claim(claim_id)
        
        assert f"Claim {claim_id} not found" in str(exc_info.value)
    
    def test_extract_text_for_analysis(self, intelligence_service, sample_medical_record):
        """Test text extraction for analysis."""
        # Execute
        text = intelligence_service._extract_text_for_analysis(sample_medical_record)
        
        # Verify
        assert "John Doe" in text
        assert "45" in text
        assert "Male" in text
        assert "Cardiac Bypass Surgery" in text
        assert "Coronary Artery Disease" in text
    
    def test_generate_search_queries(self, intelligence_service, sample_medical_record):
        """Test search query generation."""
        # Execute
        text = intelligence_service._extract_text_for_analysis(sample_medical_record)
        queries = intelligence_service._generate_search_queries(sample_medical_record, text)
        
        # Verify
        assert len(queries) > 0
        assert any("Coronary Artery Disease" in query for query in queries)
        assert any("Cardiac Bypass Surgery" in query for query in queries)
    
    def test_calculate_package_confidence(self, intelligence_service, sample_medical_record):
        """Test package confidence calculation."""
        package_info = {
            "similarity_score": 0.8,
            "package_code": "CARD-001"
        }
        
        # Execute
        confidence = intelligence_service._calculate_package_confidence(
            package_info=package_info,
            procedures_text="Cardiac Bypass Surgery",
            diagnoses_text="Coronary Artery Disease",
            medical_data=sample_medical_record
        )
        
        # Verify
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high given good inputs
    
    def test_estimate_approval_probability(self, intelligence_service, sample_medical_record):
        """Test approval probability estimation."""
        package_info = {"package_code": "CARD-001"}
        
        # Execute
        probability = intelligence_service._estimate_approval_probability(
            package_info=package_info,
            medical_data=sample_medical_record,
            search_results={}
        )
        
        # Verify
        assert 0.0 <= probability <= 1.0
    
    def test_estimate_package_amount(self, intelligence_service, sample_medical_record):
        """Test package amount estimation."""
        package_info = {"package_code": "CARD-001"}
        
        # Execute
        amount = intelligence_service._estimate_package_amount(
            package_info=package_info,
            medical_data=sample_medical_record
        )
        
        # Verify
        assert isinstance(amount, Decimal)
        assert amount > 0
        assert amount == sample_medical_record.total_amount  # Should use actual amount
    
    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, intelligence_service, mock_services):
        """Test health check with all services healthy."""
        # Setup mocks to return healthy status
        mock_services["document_service"].health_check.return_value = {"status": "healthy"}
        mock_services["chroma_service"].health_check.return_value = {"status": "healthy"}
        mock_services["language_service"].health_check.return_value = {"status": "healthy"}
        mock_services["embedding_service"].health_check.return_value = {"status": "healthy"}
        
        # Execute
        health = await intelligence_service.health_check()
        
        # Verify
        assert health["service"] == "claims_intelligence"
        assert health["status"] == "healthy"
        assert "dependencies" in health
        assert all(status == "healthy" for status in health["dependencies"].values())
    
    @pytest.mark.asyncio
    async def test_health_check_with_failures(self, intelligence_service, mock_services):
        """Test health check with some service failures."""
        # Setup mocks with some failures
        mock_services["document_service"].health_check.side_effect = Exception("Service down")
        mock_services["chroma_service"].health_check.return_value = {"status": "healthy"}
        mock_services["language_service"].health_check.return_value = {"status": "healthy"}
        mock_services["embedding_service"].health_check.return_value = {"status": "healthy"}
        
        # Execute
        health = await intelligence_service.health_check()
        
        # Verify
        assert health["service"] == "claims_intelligence"
        assert health["status"] == "degraded"  # Should be degraded due to document service failure
        assert "unhealthy: Service down" in health["dependencies"]["document_service"]


class TestClaimsIntelligenceServicePerformance:
    """Performance tests for Claims Intelligence Service."""
    
    @pytest_asyncio.fixture
    async def performance_service(self, mock_services):
        """Create service for performance testing."""
        return ClaimsIntelligenceService(
            document_service=mock_services["document_service"],
            chroma_service=mock_services["chroma_service"],
            language_service=mock_services["language_service"],
            embedding_service=mock_services["embedding_service"],
            claims_repository=mock_services["claims_repository"],
            steps_repository=mock_services["steps_repository"],
            vector_repository=mock_services["vector_repository"]
        )
    
    @pytest.mark.asyncio
    async def test_processing_time_requirement(self, performance_service, mock_services, sample_ocr_result, sample_search_results):
        """Test that claim processing completes within reasonable time."""
        # Setup mocks for fast responses
        mock_services["document_service"].process_document.return_value = sample_ocr_result
        mock_services["language_service"].detect_language.return_value = {
            "detected_language": "english",
            "confidence": 0.95,
            "processing_time_ms": 50
        }
        mock_services["chroma_service"].search_pmjay_guidelines.return_value = sample_search_results["pmjay_results"]
        mock_services["chroma_service"].search_medical_codes.return_value = sample_search_results["medical_codes_results"]
        
        import time
        start_time = time.time()
        
        # Execute
        result = await performance_service.process_claim_document(
            document_path="test_document.pdf",
            hospital_id="H001"
        )
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        # Verify performance requirement
        assert processing_time < 10000  # Less than 10 seconds
        assert result["processing_time_ms"] > 0
        assert result["processing_status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, performance_service, mock_services, sample_ocr_result, sample_search_results):
        """Test concurrent claim processing."""
        # Setup mocks
        mock_services["document_service"].process_document.return_value = sample_ocr_result
        mock_services["language_service"].detect_language.return_value = {
            "detected_language": "english",
            "confidence": 0.95,
            "processing_time_ms": 50
        }
        mock_services["chroma_service"].search_pmjay_guidelines.return_value = sample_search_results["pmjay_results"]
        mock_services["chroma_service"].search_medical_codes.return_value = sample_search_results["medical_codes_results"]
        
        # Execute multiple concurrent processes
        tasks = []
        for i in range(3):
            task = performance_service.process_claim_document(
                document_path=f"test_document_{i}.pdf",
                hospital_id=f"H00{i+1}"
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        
        # Verify all completed successfully
        assert len(results) == 3
        assert all(result["processing_status"] == "completed" for result in results)
        assert all("claim_id" in result for result in results)
        
        # Verify unique claim IDs
        claim_ids = [result["claim_id"] for result in results]
        assert len(set(claim_ids)) == 3  # All unique
