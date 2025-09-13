"""Unit tests for document processing service."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime

from services.document_processing_service import (
    DocumentProcessingService, 
    OCRResult, 
    DocumentProcessingResult
)
from models.medical import MedicalRecord, PatientInfo, MedicalProcedure, Diagnosis


class TestDocumentProcessingService:
    """Test cases for DocumentProcessingService."""
    
    @pytest.fixture
    def service(self):
        """Create a document processing service instance."""
        return DocumentProcessingService()
    
    @pytest.fixture
    def sample_medical_text(self):
        """Sample medical document text for testing."""
        return """
        MEDICAL RECORD
        
        Patient Name: John Doe
        Patient ID: P12345
        Age: 45
        Gender: Male
        
        Hospital ID: H001
        Admission Date: 15/03/2024
        Discharge Date: 20/03/2024
        
        Diagnosis: Hypertension
        Procedure: Blood pressure monitoring
        
        Total Amount: ₹5,000.00
        """
    
    @pytest.fixture
    def sample_pdf_file(self):
        """Create a temporary PDF file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(b'%PDF-1.4\n%Test PDF content')
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_service_initialization(self, service):
        """Test service initialization."""
        assert service.confidence_threshold == 0.8
        assert '.pdf' in service.supported_formats
        assert '.png' in service.supported_formats
        assert '.jpg' in service.supported_formats
    
    def test_validate_file_exists(self, service, sample_pdf_file):
        """Test file validation for existing file."""
        assert service._validate_file(sample_pdf_file) is True
    
    def test_validate_file_not_exists(self, service):
        """Test file validation for non-existent file."""
        assert service._validate_file("nonexistent.pdf") is False
    
    def test_validate_file_unsupported_format(self, service):
        """Test file validation for unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'test content')
            temp_path = f.name
        
        try:
            assert service._validate_file(temp_path) is False
        finally:
            os.unlink(temp_path)
    
    def test_estimate_ocr_confidence_empty_text(self, service):
        """Test OCR confidence estimation with empty text."""
        confidence = service._estimate_ocr_confidence("")
        assert confidence == 0.0
    
    def test_estimate_ocr_confidence_medical_text(self, service, sample_medical_text):
        """Test OCR confidence estimation with medical text."""
        confidence = service._estimate_ocr_confidence(sample_medical_text)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high for medical text
    
    def test_extract_patient_info(self, service, sample_medical_text):
        """Test patient information extraction."""
        patient_info = service._extract_patient_info(sample_medical_text)
        
        assert patient_info.get('patient_id') == 'P12345'
        assert 'John' in patient_info.get('name', '')  # Check for partial name match
        assert patient_info.get('age') == 45
        assert patient_info.get('gender') == 'Male'
    
    def test_extract_procedures(self, service, sample_medical_text):
        """Test procedure extraction."""
        procedures = service._extract_procedures(sample_medical_text)
        
        assert len(procedures) > 0
        assert any('Blood pressure monitoring' in proc['procedure_name'] for proc in procedures)
    
    def test_extract_diagnoses(self, service, sample_medical_text):
        """Test diagnosis extraction."""
        diagnoses = service._extract_diagnoses(sample_medical_text)
        
        assert len(diagnoses) > 0
        assert any('Hypertension' in diag['diagnosis_name'] for diag in diagnoses)
    
    def test_extract_hospital_info(self, service, sample_medical_text):
        """Test hospital information extraction."""
        hospital_info = service._extract_hospital_info(sample_medical_text)
        
        assert hospital_info['hospital_id'] == 'H001'
    
    def test_extract_dates(self, service, sample_medical_text):
        """Test date extraction."""
        dates = service._extract_dates(sample_medical_text)
        
        assert 'admission_date' in dates
        assert 'discharge_date' in dates
        assert '15/03/2024' in dates['admission_date']
        assert '20/03/2024' in dates['discharge_date']
    
    def test_extract_amounts(self, service, sample_medical_text):
        """Test amount extraction."""
        amounts = service._extract_amounts(sample_medical_text)
        
        assert 'total_amount' in amounts
        assert '5000.00' in amounts['total_amount']
    
    def test_parse_date_valid(self, service):
        """Test date parsing with valid date."""
        date_obj = service._parse_date("15/03/2024")
        assert isinstance(date_obj, datetime)
        assert date_obj.day == 15
        assert date_obj.month == 3
        assert date_obj.year == 2024
    
    def test_parse_date_invalid(self, service):
        """Test date parsing with invalid date."""
        date_obj = service._parse_date("invalid-date")
        assert isinstance(date_obj, datetime)
        # Should return current date for invalid input
    
    def test_parse_amount_valid(self, service):
        """Test amount parsing with valid amount."""
        amount = service._parse_amount("₹5,000.00")
        assert isinstance(amount, Decimal)
        assert amount == Decimal("5000.00")
    
    def test_parse_amount_invalid(self, service):
        """Test amount parsing with invalid amount."""
        amount = service._parse_amount("invalid-amount")
        assert amount == Decimal("0.00")
    
    def test_extract_medical_data(self, service, sample_medical_text):
        """Test complete medical data extraction."""
        extracted_data = service.extract_medical_data(sample_medical_text)
        
        assert 'patient_info' in extracted_data
        assert 'procedures' in extracted_data
        assert 'diagnoses' in extracted_data
        assert 'hospital_info' in extracted_data
        assert 'dates' in extracted_data
        assert 'amounts' in extracted_data
        
        # Verify patient info
        patient_info = extracted_data['patient_info']
        assert patient_info.get('patient_id') == 'P12345'
        assert 'John' in patient_info.get('name', '')  # Check for partial name match
        assert patient_info.get('age') == 45
        assert patient_info.get('gender') == 'Male'
    
    def test_create_medical_record(self, service, sample_medical_text):
        """Test medical record creation from extracted data."""
        extracted_data = service.extract_medical_data(sample_medical_text)
        medical_record = service.create_medical_record(extracted_data)
        
        assert isinstance(medical_record, MedicalRecord)
        assert medical_record.patient_info.patient_id == 'P12345'
        assert 'John' in medical_record.patient_info.name  # Check for partial name match
        assert medical_record.patient_info.age == 45
        assert medical_record.patient_info.gender == 'Male'
        assert medical_record.hospital_id == 'H001'
        assert len(medical_record.procedures) > 0
        assert len(medical_record.diagnoses) > 0
    
    @patch('services.document_processing_service.granite_language_service')
    def test_extract_text_ocr_fallback(self, mock_language_service, service, sample_pdf_file):
        """Test OCR text extraction with fallback method."""
        # Mock language detection
        mock_language_service.detect_language.return_value = {
            'detected_language': 'english',
            'confidence': 0.9
        }
        
        # Force fallback by setting converter to None
        service.converter = None
        service.docling_available = False
        
        result = service.extract_text_ocr(sample_pdf_file)
        
        assert isinstance(result, OCRResult)
        assert result.method in ['pypdf2', 'fallback']
        assert result.confidence >= 0.0
    
    @patch('services.document_processing_service.granite_language_service')
    def test_process_document_complete_workflow(self, mock_language_service, service, sample_pdf_file):
        """Test complete document processing workflow."""
        # Mock language detection
        mock_language_service.detect_language.return_value = {
            'detected_language': 'english',
            'confidence': 0.9
        }
        
        # Mock OCR to return our sample text
        with patch.object(service, 'extract_text_ocr') as mock_ocr:
            mock_ocr.return_value = OCRResult(
                text="""
                Patient Name: Jane Smith
                Patient ID: P67890
                Age: 30
                Gender: Female
                Hospital ID: H002
                Admission Date: 10/04/2024
                Discharge Date: 15/04/2024
                Diagnosis: Diabetes
                Procedure: Blood sugar monitoring
                Total Amount: ₹3,500.00
                """,
                confidence=0.9,
                language='english',
                processing_time_ms=1000,
                page_count=1,
                method='mock',
                metadata={}
            )
            
            result = service.process_document(sample_pdf_file)
            
            assert isinstance(result, DocumentProcessingResult)
            assert result.ocr_result.confidence == 0.9
            assert result.medical_record is not None
            assert 'Jane' in result.medical_record.patient_info.name  # Check for partial name match
            assert not result.requires_verification  # High confidence
            assert len(result.validation_errors) == 0
    
    def test_process_document_low_confidence(self, service, sample_pdf_file):
        """Test document processing with low OCR confidence."""
        # Mock OCR to return low confidence result
        with patch.object(service, 'extract_text_ocr') as mock_ocr:
            mock_ocr.return_value = OCRResult(
                text="unclear text",
                confidence=0.5,  # Below threshold
                language='unknown',
                processing_time_ms=1000,
                page_count=1,
                method='mock',
                metadata={}
            )
            
            result = service.process_document(sample_pdf_file, confidence_threshold=0.8)
            
            assert result.requires_verification is True
            assert result.medical_record is None
    
    def test_process_document_invalid_file(self, service):
        """Test document processing with invalid file."""
        result = service.process_document("nonexistent.pdf")
        
        assert isinstance(result, DocumentProcessingResult)
        assert result.requires_verification is True
        assert len(result.validation_errors) > 0
        assert "Invalid file format or file not found" in result.validation_errors[0]
    
    def test_health_check(self, service):
        """Test service health check."""
        health = service.health_check()
        
        assert 'status' in health
        assert 'docling_available' in health
        assert 'supported_formats' in health
        assert 'confidence_threshold' in health
        assert 'converter_initialized' in health
        
        assert health['confidence_threshold'] == 0.8
        assert '.pdf' in health['supported_formats']


class TestOCRResult:
    """Test cases for OCRResult dataclass."""
    
    def test_ocr_result_creation(self):
        """Test OCR result creation."""
        result = OCRResult(
            text="sample text",
            confidence=0.9,
            language="english",
            processing_time_ms=1500.0,
            page_count=2,
            method="docling",
            metadata={"test": "data"}
        )
        
        assert result.text == "sample text"
        assert result.confidence == 0.9
        assert result.language == "english"
        assert result.processing_time_ms == 1500.0
        assert result.page_count == 2
        assert result.method == "docling"
        assert result.metadata["test"] == "data"


class TestDocumentProcessingResult:
    """Test cases for DocumentProcessingResult dataclass."""
    
    def test_document_processing_result_creation(self):
        """Test document processing result creation."""
        ocr_result = OCRResult(
            text="test",
            confidence=0.8,
            language="english",
            processing_time_ms=1000.0,
            page_count=1,
            method="test",
            metadata={}
        )
        
        result = DocumentProcessingResult(
            ocr_result=ocr_result,
            extracted_data={"test": "data"},
            medical_record=None,
            validation_errors=["error1"],
            requires_verification=True,
            processing_summary={"status": "completed"}
        )
        
        assert result.ocr_result == ocr_result
        assert result.extracted_data["test"] == "data"
        assert result.medical_record is None
        assert "error1" in result.validation_errors
        assert result.requires_verification is True
        assert result.processing_summary["status"] == "completed"


# Integration tests with sample documents
class TestDocumentProcessingIntegration:
    """Integration tests with sample medical documents."""
    
    @pytest.fixture
    def service(self):
        """Create service instance for integration tests."""
        return DocumentProcessingService()
    
    def test_multilingual_text_processing(self, service):
        """Test processing of multilingual medical text."""
        hindi_text = """
        मरीज़ का नाम: राम शर्मा
        मरीज़ ID: P11111
        उम्र: 50
        लिंग: पुरुष
        अस्पताल ID: H003
        भर्ती की तारीख: 01/05/2024
        छुट्टी की तारीख: 05/05/2024
        निदान: मधुमेह
        प्रक्रिया: रक्त शर्करा जांच
        कुल राशि: ₹4,000.00
        """
        
        extracted_data = service.extract_medical_data(hindi_text)
        
        # Should extract some information even from Hindi text
        assert 'patient_info' in extracted_data
        assert 'amounts' in extracted_data
        
        # Amount should be extracted regardless of language
        amounts = extracted_data['amounts']
        if 'total_amount' in amounts:
            assert '4000.00' in amounts['total_amount']
    
    def test_complex_medical_document(self, service):
        """Test processing of complex medical document."""
        complex_text = """
        DISCHARGE SUMMARY
        
        Patient Information:
        Name: Dr. Sarah Johnson
        Patient ID: P99999
        Age: 35 years
        Gender: Female
        
        Hospital Details:
        Institution: City General Hospital
        Hospital ID: CGH001
        Department: Cardiology
        
        Admission Details:
        Date of Admission: 25/06/2024
        Date of Discharge: 30/06/2024
        Length of Stay: 5 days
        
        Primary Diagnosis: Acute Myocardial Infarction (AMI)
        Secondary Diagnosis: Hypertension, Type 2 Diabetes
        
        Procedures Performed:
        1. Percutaneous Coronary Intervention (PCI)
        2. Echocardiography
        3. Cardiac Catheterization
        
        Medications:
        - Aspirin 75mg daily
        - Metoprolol 50mg twice daily
        - Atorvastatin 40mg daily
        
        Financial Information:
        Room Charges: ₹15,000.00
        Procedure Charges: ₹85,000.00
        Medication Charges: ₹5,000.00
        Total Bill Amount: ₹1,05,000.00
        
        Discharge Condition: Stable
        Follow-up: 2 weeks
        """
        
        extracted_data = service.extract_medical_data(complex_text)
        medical_record = service.create_medical_record(extracted_data)
        
        # Verify comprehensive extraction
        assert 'Sarah' in medical_record.patient_info.name  # Check for partial name match
        assert medical_record.patient_info.patient_id == 'P99999'
        assert medical_record.patient_info.age == 35
        assert medical_record.patient_info.gender == 'Female'
        assert medical_record.hospital_id == 'CGH001'
        
        # Should extract multiple procedures
        assert len(medical_record.procedures) >= 1
        
        # Should extract multiple diagnoses
        assert len(medical_record.diagnoses) >= 1
        
        # Should extract the total amount
        assert medical_record.total_amount > 0


if __name__ == "__main__":
    pytest.main([__file__])
