"""
Unit tests for Knowledge Base Management Service.

Tests for PM-JAY data ingestion, validation, version control, and conflict resolution.
"""

import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, Any, List

from services.knowledge_base_service import (
    KnowledgeBaseService, 
    PMJAYPackageSchema, 
    PMJAYGuidelineSchema, 
    MedicalCodeSchema
)
from services.chroma_service import ChromaService


class TestKnowledgeBaseService:
    """Test suite for Knowledge Base Management Service."""
    
    @pytest.fixture
    def temp_chroma_dir(self):
        """Create a temporary directory for ChromaDB."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def chroma_service(self, temp_chroma_dir):
        """Create a ChromaService instance for testing."""
        return ChromaService(persist_directory=temp_chroma_dir)
    
    @pytest.fixture
    def kb_service(self, chroma_service):
        """Create a KnowledgeBaseService instance for testing."""
        return KnowledgeBaseService(chroma_service)
    
    @pytest.fixture
    def sample_package_data(self):
        """Sample PM-JAY package data for testing."""
        return {
            "package_code": "PKG001",
            "package_name": "Cardiac Surgery Package",
            "specialty": "Cardiology",
            "procedure_type": "Surgical",
            "package_rate": 150000.0,
            "effective_date": "2025-01-01",
            "description": "Comprehensive cardiac surgery package",
            "inclusions": ["Surgery", "ICU", "Medicines"],
            "exclusions": ["Cosmetic procedures"]
        }
    
    @pytest.fixture
    def sample_guideline_data(self):
        """Sample PM-JAY guideline data for testing."""
        return {
            "guideline_id": "GL001",
            "title": "Pre-authorization Guidelines",
            "category": "Authorization",
            "content": "All high-value procedures require pre-authorization",
            "effective_date": "2025-01-01",
            "version": "1.0",
            "priority": 1,
            "circular_number": "PMJAY/2025/001"
        }
    
    @pytest.fixture
    def sample_medical_code_data(self):
        """Sample medical code data for testing."""
        return {
            "code": "I21.0",
            "code_system": "ICD-10",
            "description": "Acute transmural myocardial infarction of anterior wall",
            "category": "Cardiovascular",
            "version": "2025"
        }

    def test_init_version_control(self, kb_service):
        """Test initialization of version control collections."""
        # Version control should be initialized during service creation
        collections = kb_service.chroma_service.client.list_collections()
        collection_names = [c.name for c in collections]
        
        assert kb_service.COL_VERSION_HISTORY in collection_names
        assert kb_service.COL_AUDIT_LOG in collection_names

    def test_validate_pmjay_package_valid(self, kb_service, sample_package_data):
        """Test validation of valid PM-JAY package data."""
        result = kb_service.validate_pmjay_package(sample_package_data)
        
        assert result["valid"] is True
        assert "data" in result
        assert result["data"]["package_code"] == "PKG001"

    def test_validate_pmjay_package_invalid(self, kb_service):
        """Test validation of invalid PM-JAY package data."""
        invalid_data = {
            "package_code": "",  # Invalid: empty string
            "package_name": "Test Package",
            "specialty": "Test",
            "procedure_type": "Test",
            "package_rate": -100,  # Invalid: negative rate
            "effective_date": "2025-01-01"
        }
        
        result = kb_service.validate_pmjay_package(invalid_data)
        
        assert result["valid"] is False
        assert "errors" in result
        assert len(result["errors"]) > 0

    def test_validate_pmjay_guideline_valid(self, kb_service, sample_guideline_data):
        """Test validation of valid PM-JAY guideline data."""
        result = kb_service.validate_pmjay_guideline(sample_guideline_data)
        
        assert result["valid"] is True
        assert "data" in result
        assert result["data"]["guideline_id"] == "GL001"

    def test_validate_medical_code_valid(self, kb_service, sample_medical_code_data):
        """Test validation of valid medical code data."""
        result = kb_service.validate_medical_code(sample_medical_code_data)
        
        assert result["valid"] is True
        assert "data" in result
        assert result["data"]["code"] == "I21.0"

    def test_ingest_pmjay_packages_data_success(self, kb_service, sample_package_data):
        """Test successful ingestion of PM-JAY package data."""
        packages_data = [sample_package_data]
        
        result = kb_service.ingest_pmjay_packages_data(packages_data)
        
        assert result["success"] is True
        assert result["ingested_count"] == 1
        assert len(result["validation_errors"]) == 0

    def test_ingest_pmjay_packages_data_with_validation_errors(self, kb_service):
        """Test ingestion with validation errors."""
        invalid_packages = [
            {
                "package_code": "",  # Invalid
                "package_name": "Test Package",
                "specialty": "Test",
                "procedure_type": "Test",
                "package_rate": -100,  # Invalid
                "effective_date": "2025-01-01"
            }
        ]
        
        result = kb_service.ingest_pmjay_packages_data(invalid_packages)
        
        assert result["success"] is False
        assert "validation_errors" in result
        assert len(result["validation_errors"]) > 0

    def test_create_version_snapshot(self, kb_service, sample_package_data):
        """Test creation of version snapshots."""
        # First, ingest some data
        kb_service.ingest_pmjay_packages_data([sample_package_data])
        
        # Create a version snapshot
        result = kb_service.create_version_snapshot(
            kb_service.COL_AYUSHMAN_PACKAGE_SUCHI, 
            "v1.0"
        )
        
        assert result["success"] is True
        assert result["version_tag"] == "v1.0"
        assert "version_id" in result

    def test_rollback_to_version(self, kb_service, sample_package_data):
        """Test rollback to a previous version."""
        collection = kb_service.COL_AYUSHMAN_PACKAGE_SUCHI
        
        # Ingest initial data and create snapshot
        kb_service.ingest_pmjay_packages_data([sample_package_data])
        snapshot_result = kb_service.create_version_snapshot(collection, "v1.0")
        assert snapshot_result["success"] is True
        
        # Add more data
        new_package = sample_package_data.copy()
        new_package["package_code"] = "PKG002"
        kb_service.ingest_pmjay_packages_data([new_package])
        
        # Rollback to v1.0
        rollback_result = kb_service.rollback_to_version(collection, "v1.0")
        
        assert rollback_result["success"] is True
        assert rollback_result["version_tag"] == "v1.0"

    def test_rollback_to_nonexistent_version(self, kb_service):
        """Test rollback to a non-existent version."""
        result = kb_service.rollback_to_version(
            kb_service.COL_AYUSHMAN_PACKAGE_SUCHI, 
            "nonexistent_version"
        )
        
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_detect_conflicts_no_conflicts(self, kb_service, sample_guideline_data):
        """Test conflict detection when there are no conflicts."""
        # Ingest a single guideline
        guidelines = [sample_guideline_data]
        kb_service.ingest_pmjay_guidelines_data(guidelines)
        
        result = kb_service.detect_conflicts(kb_service.COL_PMJAY_MARGDARSHIKA)
        
        assert "conflicts" in result
        assert len(result["conflicts"]) == 0

    def test_detect_conflicts_duplicate_guideline_id(self, kb_service, sample_guideline_data):
        """Test conflict detection for duplicate guideline IDs."""
        # Create two guidelines with same ID but different content
        guideline1 = sample_guideline_data.copy()
        guideline2 = sample_guideline_data.copy()
        guideline2["content"] = "Different content for the same guideline"
        guideline2["version"] = "2.0"
        guideline2["priority"] = 2
        
        # Mock the chroma service to return these guidelines
        with patch.object(kb_service.chroma_service, 'list_items') as mock_list:
            mock_list.return_value = {
                "documents": [guideline1["content"], guideline2["content"]],
                "metadatas": [
                    {"guideline_id": "GL001", "version": "1.0", "priority": 1},
                    {"guideline_id": "GL001", "version": "2.0", "priority": 2}
                ]
            }
            
            result = kb_service.detect_conflicts(kb_service.COL_PMJAY_MARGDARSHIKA)
            
            assert "conflicts" in result
            assert len(result["conflicts"]) > 0
            assert result["conflicts"][0]["type"] == "duplicate_guideline_id"

    def test_resolve_conflicts_priority_based(self, kb_service):
        """Test conflict resolution using priority-based strategy."""
        # Mock conflict detection to return conflicts
        with patch.object(kb_service, 'detect_conflicts') as mock_detect:
            mock_detect.return_value = {
                "conflicts": [
                    {
                        "type": "duplicate_guideline_id",
                        "guideline_id": "GL001",
                        "versions": [
                            {"version": "1.0", "priority": 1},
                            {"version": "2.0", "priority": 2}
                        ]
                    }
                ]
            }
            
            result = kb_service.resolve_conflicts(
                kb_service.COL_PMJAY_MARGDARSHIKA, 
                "priority_based"
            )
            
            assert result["success"] is True
            assert result["conflicts_detected"] == 1
            assert result["conflicts_resolved"] == 1

    def test_content_similarity(self, kb_service):
        """Test content similarity calculation."""
        text1 = "This is a test document with some words"
        text2 = "This is a test document with different words"
        text3 = "Completely different content here"
        
        # Similar texts should have higher similarity
        similarity1 = kb_service._content_similarity(text1, text2)
        similarity2 = kb_service._content_similarity(text1, text3)
        
        assert similarity1 > similarity2
        assert 0 <= similarity1 <= 1
        assert 0 <= similarity2 <= 1

    def test_hot_reload_functionality(self, kb_service):
        """Test hot-reload functionality without system downtime."""
        result = kb_service.hot_reload()
        
        assert result["success"] is True
        assert "collections" in result
        
        # Check that all expected collections are present
        collections = result["collections"]
        assert kb_service.COL_PMJAY_MARGDARSHIKA in collections
        assert kb_service.COL_AYUSHMAN_PACKAGE_SUCHI in collections
        assert kb_service.COL_ROG_NIDAN_CODE_SANGRAH in collections

    def test_hot_reload_specific_collections(self, kb_service):
        """Test hot-reload for specific collections."""
        target_collections = [kb_service.COL_PMJAY_MARGDARSHIKA]
        
        result = kb_service.hot_reload(target_collections)
        
        assert result["success"] is True
        assert len(result["collections"]) == 1
        assert kb_service.COL_PMJAY_MARGDARSHIKA in result["collections"]

    def test_audit_logging(self, kb_service, sample_package_data):
        """Test that audit events are properly logged."""
        # Ingest data which should trigger audit logging
        kb_service.ingest_pmjay_packages_data([sample_package_data])
        
        # Check if audit log has entries
        audit_items = kb_service.chroma_service.list_items(
            kb_service.COL_AUDIT_LOG, 
            limit=10
        )
        
        assert "documents" in audit_items
        assert len(audit_items["documents"]) > 0

    def test_get_knowledge_base_stats(self, kb_service, sample_package_data):
        """Test knowledge base statistics retrieval."""
        # Ingest some data first
        kb_service.ingest_pmjay_packages_data([sample_package_data])
        
        stats = kb_service.get_knowledge_base_stats()
        
        assert "collections" in stats
        assert kb_service.COL_AYUSHMAN_PACKAGE_SUCHI in stats["collections"]
        assert stats["collections"][kb_service.COL_AYUSHMAN_PACKAGE_SUCHI]["count"] > 0

    def test_search_functionality(self, kb_service, sample_package_data):
        """Test search functionality after data ingestion."""
        # Ingest data
        kb_service.ingest_pmjay_packages_data([sample_package_data])
        
        # Search for the ingested data
        result = kb_service.search(
            kb_service.COL_AYUSHMAN_PACKAGE_SUCHI, 
            "cardiac surgery", 
            n_results=5
        )
        
        assert "documents" in result or "error" not in result

    def test_list_items_functionality(self, kb_service, sample_package_data):
        """Test listing items from collections."""
        # Ingest data
        kb_service.ingest_pmjay_packages_data([sample_package_data])
        
        # List items
        result = kb_service.list_items(
            kb_service.COL_AYUSHMAN_PACKAGE_SUCHI, 
            limit=10
        )
        
        assert "documents" in result or "error" not in result

    def test_delete_items_functionality(self, kb_service, sample_package_data):
        """Test deletion of items from collections."""
        # Ingest data
        ingest_result = kb_service.ingest_pmjay_packages_data([sample_package_data])
        assert ingest_result["success"] is True
        
        # List items to get IDs
        items = kb_service.list_items(kb_service.COL_AYUSHMAN_PACKAGE_SUCHI)
        
        if items.get("ids"):
            # Delete the first item
            delete_result = kb_service.delete_items(
                kb_service.COL_AYUSHMAN_PACKAGE_SUCHI, 
                [items["ids"][0]]
            )
            
            assert delete_result["success"] is True
            assert delete_result["deleted"] >= 0

    def test_error_handling_invalid_collection(self, kb_service):
        """Test error handling for invalid collection operations."""
        result = kb_service.search("nonexistent_collection", "test query")
        
        # Should handle the error gracefully
        assert "error" in result or "documents" in result

    @pytest.mark.parametrize("invalid_data", [
        {},  # Empty data
        {"package_code": ""},  # Empty package code
        {"package_code": "PKG001"},  # Missing required fields
    ])
    def test_validation_edge_cases(self, kb_service, invalid_data):
        """Test validation with various edge cases."""
        result = kb_service.validate_pmjay_package(invalid_data)
        
        assert result["valid"] is False
        assert "errors" in result


class TestPMJAYSchemas:
    """Test suite for PM-JAY data schemas."""
    
    def test_pmjay_package_schema_valid(self):
        """Test PMJAYPackageSchema with valid data."""
        valid_data = {
            "package_code": "PKG001",
            "package_name": "Test Package",
            "specialty": "Cardiology",
            "procedure_type": "Surgical",
            "package_rate": 100000.0,
            "effective_date": "2025-01-01"
        }
        
        package = PMJAYPackageSchema(**valid_data)
        assert package.package_code == "PKG001"
        assert package.package_rate == 100000.0

    def test_pmjay_guideline_schema_valid(self):
        """Test PMJAYGuidelineSchema with valid data."""
        valid_data = {
            "guideline_id": "GL001",
            "title": "Test Guideline",
            "category": "Authorization",
            "content": "Test content",
            "effective_date": "2025-01-01",
            "version": "1.0"
        }
        
        guideline = PMJAYGuidelineSchema(**valid_data)
        assert guideline.guideline_id == "GL001"
        assert guideline.priority == 1  # Default value

    def test_medical_code_schema_valid(self):
        """Test MedicalCodeSchema with valid data."""
        valid_data = {
            "code": "I21.0",
            "code_system": "ICD-10",
            "description": "Test description",
            "category": "Cardiovascular",
            "version": "2025"
        }
        
        code = MedicalCodeSchema(**valid_data)
        assert code.code == "I21.0"
        assert code.code_system == "ICD-10"


if __name__ == "__main__":
    pytest.main([__file__])
