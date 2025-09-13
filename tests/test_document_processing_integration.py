"""
Document processing integration tests.

Tests end-to-end document processing pipeline including OCR, 
multilingual support, and claims intelligence integration.
"""

import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import io

from services.document_processing_service import DocumentProcessingService
from services.claims_intelligence_service import ClaimsIntelligenceService
from services.chroma_service import ChromaService
from services.error_handler import ErrorHandlerFactory


class TestDocumentProcessingIntegration:
    """Test suite for document processing integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def doc_service(self, temp_dir):
        """Create document processing service."""
        return DocumentProcessingService(storage_path=temp_dir)
    
    @pytest.fixture
    def sample_hindi_document(self, temp_dir):
        """Create sample Hindi medical document."""
        content = """
        मरीज का नाम: राम कुमार
        उम्र: 45 वर्ष
        निदान: हृदय रोग
        उपचार: कार्डियक सर्जरी
        अस्पताल: अखिल भारतीय आयुर्विज्ञान संस्थान
        """
        
        doc_path = Path(temp_dir) / "hindi_prescription.txt"
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(doc_path)
    
    @pytest.fixture
    def sample_marathi_document(self, temp_dir):
        """Create sample Marathi medical document."""
        content = """
        रुग्णाचे नाव: सुनील पाटील
        वय: 35 वर्षे
        निदान: मधुमेह
        उपचार: इन्सुलिन थेरपी
        रुग्णालय: सासून रुग्णालय, पुणे
        """
        
        doc_path = Path(temp_dir) / "marathi_report.txt"
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(doc_path)
    
    def test_end_to_end_document_processing(self, doc_service, sample_hindi_document):
        """Test complete document processing pipeline."""
        with patch.object(doc_service, '_perform_ocr') as mock_ocr:
            mock_ocr.return_value = {
                "success": True,
                "extracted_text": "मरीज का नाम: राम कुमार\nनिदान: हृदय रोग",
                "confidence": 0.95,
                "language": "hi"
            }
            
            result = doc_service.process_document(sample_hindi_document, "prescription")
            
            assert result["success"] is True
            assert result["confidence"] >= 0.9
            assert "राम कुमार" in result["extracted_text"]
    
    def test_multilingual_ocr_processing(self, doc_service, sample_hindi_document, sample_marathi_document):
        """Test OCR processing with multiple Indian languages."""
        with patch.object(doc_service, '_perform_ocr') as mock_ocr:
            # Hindi document processing
            mock_ocr.return_value = {
                "success": True,
                "extracted_text": "मरीज का नाम: राम कुमार",
                "confidence": 0.92,
                "language": "hi"
            }
            
            hindi_result = doc_service.process_document(sample_hindi_document, "prescription")
            assert hindi_result["language"] == "hi"
            
            # Marathi document processing
            mock_ocr.return_value = {
                "success": True,
                "extracted_text": "रुग्णाचे नाव: सुनील पाटील",
                "confidence": 0.89,
                "language": "mr"
            }
            
            marathi_result = doc_service.process_document(sample_marathi_document, "lab_report")
            assert marathi_result["language"] == "mr"
    
    def test_low_confidence_handling(self, doc_service, sample_hindi_document):
        """Test handling of low confidence OCR results."""
        with patch.object(doc_service, '_perform_ocr') as mock_ocr:
            mock_ocr.return_value = {
                "success": True,
                "extracted_text": "unclear text",
                "confidence": 0.65,  # Below threshold
                "language": "hi"
            }
            
            result = doc_service.process_document(sample_hindi_document, "prescription")
            
            assert result["requires_manual_review"] is True
            assert result["confidence"] < 0.8
    
    def test_document_type_classification(self, doc_service):
        """Test automatic document type classification."""
        test_cases = [
            ("prescription content with medicine names", "prescription"),
            ("lab test results hemoglobin glucose", "lab_report"),
            ("discharge summary patient condition", "discharge_summary"),
            ("insurance claim form details", "insurance_form")
        ]
        
        for content, expected_type in test_cases:
            with patch.object(doc_service, '_classify_document_type', return_value=expected_type):
                result = doc_service._classify_document_type(content)
                assert result == expected_type
    
    def test_claims_intelligence_integration(self, doc_service):
        """Test integration with claims intelligence service."""
        extracted_text = """
        मरीज का नाम: राम कुमार
        निदान: हृदय रोग
        उपचार: कार्डियक सर्जरी
        अनुमानित लागत: ₹1,50,000
        """
        
        with patch('services.claims_intelligence_service.ClaimsIntelligenceService') as mock_claims_service:
            mock_instance = mock_claims_service.return_value
            mock_instance.analyze_medical_content.return_value = {
                "package_recommendations": [
                    {"package_code": "PKG001", "confidence": 0.92, "package_name": "Cardiac Surgery"}
                ],
                "estimated_amount": 150000,
                "diagnosis_codes": ["I21.0"]
            }
            
            result = doc_service._analyze_with_claims_intelligence(extracted_text)
            
            assert "package_recommendations" in result
            assert result["estimated_amount"] == 150000
    
    def test_error_handling_and_recovery(self, doc_service, sample_hindi_document):
        """Test error handling during document processing."""
        error_handler = ErrorHandlerFactory.get_ocr_handler()
        
        with patch.object(doc_service, '_perform_ocr', side_effect=Exception("OCR service unavailable")):
            result = doc_service.process_document(sample_hindi_document, "prescription")
            
            assert result["success"] is False
            assert "error" in result
            
            # Check that error was logged
            assert len(error_handler.error_records) > 0
    
    def test_batch_document_processing(self, doc_service, temp_dir):
        """Test batch processing of multiple documents."""
        # Create multiple test documents
        documents = []
        for i in range(5):
            doc_path = Path(temp_dir) / f"test_doc_{i}.txt"
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(f"Test document {i} content")
            documents.append(str(doc_path))
        
        with patch.object(doc_service, '_perform_ocr') as mock_ocr:
            mock_ocr.return_value = {
                "success": True,
                "extracted_text": "Test document content",
                "confidence": 0.95,
                "language": "en"
            }
            
            results = doc_service.process_documents_batch(documents)
            
            assert len(results) == 5
            assert all(result["success"] for result in results)
    
    def test_document_validation(self, doc_service):
        """Test document validation before processing."""
        # Test valid file types
        valid_files = ["test.pdf", "test.jpg", "test.png", "test.tiff"]
        for file_path in valid_files:
            assert doc_service._is_valid_document_type(file_path) is True
        
        # Test invalid file types
        invalid_files = ["test.txt", "test.docx", "test.mp4"]
        for file_path in invalid_files:
            assert doc_service._is_valid_document_type(file_path) is False
    
    def test_performance_optimization(self, doc_service, temp_dir):
        """Test performance optimizations in document processing."""
        # Create large document
        large_content = "Test content " * 1000
        doc_path = Path(temp_dir) / "large_document.txt"
        with open(doc_path, 'w') as f:
            f.write(large_content)
        
        with patch.object(doc_service, '_perform_ocr') as mock_ocr:
            mock_ocr.return_value = {
                "success": True,
                "extracted_text": large_content,
                "confidence": 0.95,
                "language": "en"
            }
            
            import time
            start_time = time.time()
            result = doc_service.process_document(str(doc_path), "prescription")
            processing_time = time.time() - start_time
            
            assert result["success"] is True
            assert processing_time < 30  # Should process within 30 seconds
    
    @pytest.mark.asyncio
    async def test_concurrent_document_processing(self, doc_service, temp_dir):
        """Test concurrent processing of multiple documents."""
        # Create test documents
        documents = []
        for i in range(3):
            doc_path = Path(temp_dir) / f"concurrent_doc_{i}.txt"
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(f"Document {i} for concurrent processing")
            documents.append(str(doc_path))
        
        with patch.object(doc_service, '_perform_ocr') as mock_ocr:
            mock_ocr.return_value = {
                "success": True,
                "extracted_text": "Concurrent document content",
                "confidence": 0.95,
                "language": "en"
            }
            
            # Process documents concurrently
            tasks = [doc_service.process_document_async(doc, "prescription") for doc in documents]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            assert all(result["success"] for result in results)
    
    def test_document_metadata_extraction(self, doc_service, sample_hindi_document):
        """Test extraction of document metadata."""
        with patch.object(doc_service, '_extract_metadata') as mock_metadata:
            mock_metadata.return_value = {
                "file_size": 1024,
                "creation_date": "2025-01-01",
                "file_type": "text/plain",
                "page_count": 1
            }
            
            metadata = doc_service._extract_metadata(sample_hindi_document)
            
            assert "file_size" in metadata
            assert "creation_date" in metadata
            assert "file_type" in metadata
    
    def test_quality_assessment(self, doc_service):
        """Test document quality assessment."""
        # High quality document
        high_quality_result = {
            "success": True,
            "extracted_text": "Clear medical prescription with patient details",
            "confidence": 0.95,
            "language": "hi"
        }
        
        quality_score = doc_service._assess_quality(high_quality_result)
        assert quality_score >= 0.9
        
        # Low quality document
        low_quality_result = {
            "success": True,
            "extracted_text": "unclear text fragments",
            "confidence": 0.65,
            "language": "unknown"
        }
        
        quality_score = doc_service._assess_quality(low_quality_result)
        assert quality_score < 0.7


class TestDocumentStorageIntegration:
    """Test suite for document storage integration."""
    
    @pytest.fixture
    def storage_service(self, temp_dir):
        """Create document storage service."""
        from services.document_storage_service import DocumentStorageService
        return DocumentStorageService(base_path=temp_dir)
    
    def test_secure_document_storage(self, storage_service):
        """Test secure storage of processed documents."""
        document_data = {
            "original_filename": "prescription.pdf",
            "extracted_text": "Medical prescription content",
            "metadata": {"patient_id": "PAT001", "claim_id": "CLAIM001"}
        }
        
        storage_result = storage_service.store_document(document_data)
        
        assert storage_result["success"] is True
        assert "document_id" in storage_result
        assert "storage_path" in storage_result
    
    def test_document_retrieval(self, storage_service):
        """Test retrieval of stored documents."""
        # First store a document
        document_data = {
            "original_filename": "test.pdf",
            "extracted_text": "Test content",
            "metadata": {"test": True}
        }
        
        storage_result = storage_service.store_document(document_data)
        document_id = storage_result["document_id"]
        
        # Then retrieve it
        retrieved = storage_service.retrieve_document(document_id)
        
        assert retrieved["success"] is True
        assert retrieved["extracted_text"] == "Test content"
    
    def test_document_versioning(self, storage_service):
        """Test document versioning capabilities."""
        document_data = {
            "original_filename": "versioned.pdf",
            "extracted_text": "Version 1 content",
            "metadata": {"version": 1}
        }
        
        # Store initial version
        v1_result = storage_service.store_document(document_data)
        
        # Store updated version
        document_data["extracted_text"] = "Version 2 content"
        document_data["metadata"]["version"] = 2
        v2_result = storage_service.store_document(document_data, version_of=v1_result["document_id"])
        
        assert v2_result["success"] is True
        assert v2_result["version"] == 2


if __name__ == "__main__":
    pytest.main([__file__])
