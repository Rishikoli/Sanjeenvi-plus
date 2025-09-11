import pytest
from datetime import datetime
from decimal import Decimal

from models.medical import PatientInfo, MedicalProcedure, Diagnosis, MedicalRecord
from models.claims import (
    PackageRecommendation, 
    RiskAssessment, 
    ClaimSubmission,
    ComplianceStatus,
    SubmissionStatus
)


class TestMedicalModels:
    """Test cases for medical data models."""
    
    def test_patient_info_valid(self):
        """Test valid PatientInfo creation."""
        patient = PatientInfo(
            patient_id="P12345",
            name="John Doe",
            age=45,
            gender="Male"
        )
        assert patient.patient_id == "P12345"
        assert patient.name == "John Doe"
        assert patient.age == 45
        assert patient.gender == "Male"
    
    def test_patient_info_invalid_age(self):
        """Test PatientInfo with invalid age."""
        with pytest.raises(ValueError):
            PatientInfo(
                patient_id="P12345",
                name="John Doe",
                age=0,  # Invalid: age must be > 0
                gender="Male"
            )
    
    def test_medical_procedure_valid(self):
        """Test valid MedicalProcedure creation."""
        procedure = MedicalProcedure(
            code="ICD-10-001",
            description="Cardiac catheterization"
        )
        assert procedure.code == "ICD-10-001"
        assert procedure.description == "Cardiac catheterization"
    
    def test_diagnosis_valid(self):
        """Test valid Diagnosis creation."""
        diagnosis = Diagnosis(
            code="I25.10",
            description="Atherosclerotic heart disease"
        )
        assert diagnosis.code == "I25.10"
        assert diagnosis.description == "Atherosclerotic heart disease"
    
    def test_medical_record_valid(self):
        """Test valid MedicalRecord creation."""
        patient = PatientInfo(
            patient_id="P12345",
            name="John Doe",
            age=45,
            gender="Male"
        )
        
        procedure = MedicalProcedure(
            code="ICD-10-001",
            description="Cardiac catheterization"
        )
        
        diagnosis = Diagnosis(
            code="I25.10",
            description="Atherosclerotic heart disease"
        )
        
        record = MedicalRecord(
            patient_info=patient,
            hospital_id="H001",
            admission_date=datetime(2024, 1, 15),
            discharge_date=datetime(2024, 1, 20),
            procedures=[procedure],
            diagnoses=[diagnosis],
            total_amount=Decimal("50000.00"),
            document_confidence=0.95
        )
        
        assert record.patient_info.patient_id == "P12345"
        assert record.hospital_id == "H001"
        assert record.total_amount == Decimal("50000.00")
        assert record.document_confidence == 0.95
    
    def test_medical_record_invalid_confidence(self):
        """Test MedicalRecord with invalid confidence score."""
        patient = PatientInfo(
            patient_id="P12345",
            name="John Doe",
            age=45,
            gender="Male"
        )
        
        with pytest.raises(ValueError):
            MedicalRecord(
                patient_info=patient,
                hospital_id="H001",
                admission_date=datetime(2024, 1, 15),
                discharge_date=datetime(2024, 1, 20),
                procedures=[],
                diagnoses=[],
                total_amount=Decimal("50000.00"),
                document_confidence=1.5  # Invalid: must be <= 1
            )


class TestClaimsModels:
    """Test cases for claims processing models."""
    
    def test_package_recommendation_valid(self):
        """Test valid PackageRecommendation creation."""
        recommendation = PackageRecommendation(
            package_code="HBP-001",
            package_name="Cardiac Surgery Package",
            confidence_score=0.92,
            estimated_amount=Decimal("75000.00"),
            approval_probability=0.85,
            risk_factors=["High complexity procedure"],
            compliance_status=ComplianceStatus.COMPLIANT,
            chroma_similarity_scores=[0.89, 0.87, 0.85]
        )
        
        assert recommendation.package_code == "HBP-001"
        assert recommendation.confidence_score == 0.92
        assert recommendation.compliance_status == ComplianceStatus.COMPLIANT
        assert len(recommendation.chroma_similarity_scores) == 3
    
    def test_package_recommendation_invalid_confidence(self):
        """Test PackageRecommendation with invalid confidence score."""
        with pytest.raises(ValueError):
            PackageRecommendation(
                package_code="HBP-001",
                package_name="Cardiac Surgery Package",
                confidence_score=1.5,  # Invalid: must be <= 1
                estimated_amount=Decimal("75000.00"),
                approval_probability=0.85,
                compliance_status=ComplianceStatus.COMPLIANT
            )
    
    def test_risk_assessment_valid(self):
        """Test valid RiskAssessment creation."""
        risk = RiskAssessment(
            risk_score=0.3,
            denial_probability=0.15,
            risk_factors=["Minor documentation gaps"],
            mitigation_suggestions=["Provide additional medical history"]
        )
        
        assert risk.risk_score == 0.3
        assert risk.denial_probability == 0.15
        assert len(risk.risk_factors) == 1
        assert len(risk.mitigation_suggestions) == 1
    
    def test_claim_submission_valid(self):
        """Test valid ClaimSubmission creation."""
        # Create required nested objects
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
        
        claim = ClaimSubmission(
            claim_id="C12345",
            hospital_id="H001",
            patient_data=medical_record,
            recommended_package=recommendation,
            submission_status=SubmissionStatus.PENDING,
            portal_reference="REF123456"
        )
        
        assert claim.claim_id == "C12345"
        assert claim.submission_status == SubmissionStatus.PENDING
        assert claim.portal_reference == "REF123456"
        assert claim.retry_count == 0  # Default value
    
    def test_claim_submission_invalid_retry_count(self):
        """Test ClaimSubmission with invalid retry count."""
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
        
        with pytest.raises(ValueError):
            ClaimSubmission(
                claim_id="C12345",
                hospital_id="H001",
                patient_data=medical_record,
                recommended_package=recommendation,
                retry_count=-1  # Invalid: must be >= 0
            )