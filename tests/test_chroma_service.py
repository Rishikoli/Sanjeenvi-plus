"""Unit tests for ChromaDB service."""

import pytest
import tempfile
import shutil
from pathlib import Path

from services.chroma_service import ChromaService


@pytest.fixture
def temp_chroma_service():
    """Create a temporary ChromaDB service for testing."""
    temp_dir = tempfile.mkdtemp()
    service = ChromaService(persist_directory=temp_dir)
    
    yield service
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestChromaService:
    """Test ChromaDB service functionality."""
    
    def test_service_initialization(self, temp_chroma_service):
        """Test ChromaDB service initialization."""
        service = temp_chroma_service
        assert service.client is not None
        assert service.embedding_function is not None
        assert service.persist_directory.exists()
    
    def test_health_check(self, temp_chroma_service):
        """Test ChromaDB health check."""
        service = temp_chroma_service
        health = service.health_check()
        assert health is True
    
    def test_create_pmjay_collection(self, temp_chroma_service):
        """Test creating PM-JAY collection."""
        service = temp_chroma_service
        collection = service.create_pmjay_collection()
        
        assert collection is not None
        assert collection.name == "pmjay_guidelines"
        assert service.pmjay_collection is not None
    
    def test_create_medical_codes_collection(self, temp_chroma_service):
        """Test creating medical codes collection."""
        service = temp_chroma_service
        collection = service.create_medical_codes_collection()
        
        assert collection is not None
        assert collection.name == "medical_codes"
        assert service.medical_codes_collection is not None
    
    def test_add_pmjay_documents(self, temp_chroma_service):
        """Test adding PM-JAY documents."""
        service = temp_chroma_service
        
        documents = [
            "PM-JAY package HBP-001 covers cardiac surgery procedures with pre-authorization required.",
            "Eligibility criteria for PM-JAY includes income verification and empanelment status.",
            "Package rates are fixed and include all associated medical procedures and diagnostics."
        ]
        
        metadatas = [
            {"category": "packages", "package_code": "HBP-001", "version": "2025.1"},
            {"category": "eligibility", "section": "criteria", "version": "2025.1"},
            {"category": "rates", "section": "pricing", "version": "2025.1"}
        ]
        
        ids = ["pmjay_001", "pmjay_002", "pmjay_003"]
        
        service.add_pmjay_documents(documents, metadatas, ids)
        
        # Verify documents were added
        collection_info = service.get_collection_info("pmjay_guidelines")
        assert collection_info["count"] == 3
    
    def test_add_medical_codes(self, temp_chroma_service):
        """Test adding medical codes."""
        service = temp_chroma_service
        
        documents = [
            "ICD-10 code I25.10 - Atherosclerotic heart disease of native coronary artery",
            "ICD-10 code I21.9 - Acute myocardial infarction, unspecified",
            "CPT code 93458 - Catheter placement in coronary artery for coronary angiography"
        ]
        
        metadatas = [
            {"code_type": "ICD-10", "code": "I25.10", "category": "cardiovascular"},
            {"code_type": "ICD-10", "code": "I21.9", "category": "cardiovascular"},
            {"code_type": "CPT", "code": "93458", "category": "procedures"}
        ]
        
        ids = ["icd_001", "icd_002", "cpt_001"]
        
        service.add_medical_codes(documents, metadatas, ids)
        
        # Verify codes were added
        collection_info = service.get_collection_info("medical_codes")
        assert collection_info["count"] == 3
    
    def test_search_pmjay_guidelines(self, temp_chroma_service):
        """Test searching PM-JAY guidelines."""
        service = temp_chroma_service
        
        # Add test documents first
        documents = [
            "PM-JAY package HBP-001 covers cardiac surgery procedures with pre-authorization required.",
            "Eligibility criteria for PM-JAY includes income verification and empanelment status.",
            "Package rates are fixed and include all associated medical procedures and diagnostics."
        ]
        
        metadatas = [
            {"category": "packages", "package_code": "HBP-001"},
            {"category": "eligibility", "section": "criteria"},
            {"category": "rates", "section": "pricing"}
        ]
        
        ids = ["pmjay_001", "pmjay_002", "pmjay_003"]
        
        service.add_pmjay_documents(documents, metadatas, ids)
        
        # Search for cardiac surgery
        results = service.search_pmjay_guidelines("cardiac surgery procedures", n_results=2)
        
        assert "documents" in results
        assert "similarities" in results
        assert "query_time_ms" in results
        assert results["result_count"] > 0
        assert len(results["documents"]) <= 2
        
        # Check that query time is reasonable (< 1 second as per requirements)
        assert results["query_time_ms"] < 1000
    
    def test_search_medical_codes(self, temp_chroma_service):
        """Test searching medical codes."""
        service = temp_chroma_service
        
        # Add test codes first
        documents = [
            "ICD-10 code I25.10 - Atherosclerotic heart disease of native coronary artery",
            "ICD-10 code I21.9 - Acute myocardial infarction, unspecified",
            "CPT code 93458 - Catheter placement in coronary artery for coronary angiography"
        ]
        
        metadatas = [
            {"code_type": "ICD-10", "code": "I25.10", "category": "cardiovascular"},
            {"code_type": "ICD-10", "code": "I21.9", "category": "cardiovascular"},
            {"code_type": "CPT", "code": "93458", "category": "procedures"}
        ]
        
        ids = ["icd_001", "icd_002", "cpt_001"]
        
        service.add_medical_codes(documents, metadatas, ids)
        
        # Search for heart disease
        results = service.search_medical_codes("heart disease", n_results=2)
        
        assert "documents" in results
        assert "similarities" in results
        assert "query_time_ms" in results
        assert results["result_count"] > 0
        assert len(results["documents"]) <= 2
        
        # Check query performance
        assert results["query_time_ms"] < 1000
    
    def test_search_with_metadata_filter(self, temp_chroma_service):
        """Test searching with metadata filters."""
        service = temp_chroma_service
        
        # Add test documents with different categories
        documents = [
            "PM-JAY package HBP-001 covers cardiac surgery procedures.",
            "PM-JAY eligibility requires income verification.",
            "Package rates include all medical procedures."
        ]
        
        metadatas = [
            {"category": "packages", "package_code": "HBP-001"},
            {"category": "eligibility", "section": "criteria"},
            {"category": "rates", "section": "pricing"}
        ]
        
        ids = ["pmjay_001", "pmjay_002", "pmjay_003"]
        
        service.add_pmjay_documents(documents, metadatas, ids)
        
        # Search only in packages category
        results = service.search_pmjay_guidelines(
            "cardiac surgery", 
            n_results=5,
            where={"category": "packages"}
        )
        
        assert results["result_count"] > 0
        # All results should be from packages category
        for metadata in results["metadatas"]:
            assert metadata["category"] == "packages"
    
    def test_get_collection_info(self, temp_chroma_service):
        """Test getting collection information."""
        service = temp_chroma_service
        
        # Create collection and add some documents
        service.create_pmjay_collection()
        service.add_pmjay_documents(
            ["Test document"], 
            [{"test": "metadata"}], 
            ["test_001"]
        )
        
        info = service.get_collection_info("pmjay_guidelines")
        
        assert info["name"] == "pmjay_guidelines"
        assert info["count"] == 1
        assert "metadata" in info
    
    def test_delete_collection(self, temp_chroma_service):
        """Test deleting a collection."""
        service = temp_chroma_service
        
        # Create collection first
        service.create_pmjay_collection()
        
        # Verify it exists
        info = service.get_collection_info("pmjay_guidelines")
        assert info["name"] == "pmjay_guidelines"
        
        # Delete collection
        success = service.delete_collection("pmjay_guidelines")
        assert success is True
        
        # Verify it's deleted
        with pytest.raises(Exception):
            service.get_collection_info("pmjay_guidelines")
    
    def test_reset_database(self, temp_chroma_service):
        """Test resetting the database."""
        service = temp_chroma_service
        
        # Create collections and add data
        service.create_pmjay_collection()
        service.create_medical_codes_collection()
        
        # Reset database
        success = service.reset_database()
        assert success is True
        
        # Verify collections are gone
        collections = service.client.list_collections()
        assert len(collections) == 0


class TestChromaServicePerformance:
    """Test ChromaDB service performance requirements."""
    
    def test_search_performance_requirement(self, temp_chroma_service):
        """Test that vector search meets <1 second requirement."""
        service = temp_chroma_service
        
        # Add a reasonable number of documents
        documents = [f"PM-JAY guideline document {i} with medical procedures and eligibility criteria." for i in range(100)]
        metadatas = [{"doc_id": i, "category": "guidelines"} for i in range(100)]
        ids = [f"doc_{i:03d}" for i in range(100)]
        
        service.add_pmjay_documents(documents, metadatas, ids)
        
        # Perform search and measure time
        results = service.search_pmjay_guidelines("medical procedures eligibility", n_results=10)
        
        # Verify performance requirement (<1 second = 1000ms)
        assert results["query_time_ms"] < 1000, f"Search took {results['query_time_ms']}ms, exceeds 1000ms requirement"
        
        # Verify we got results
        assert results["result_count"] > 0
        assert len(results["similarities"]) > 0
