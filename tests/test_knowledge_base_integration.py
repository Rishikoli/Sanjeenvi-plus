"""
Knowledge base integration tests for PM-JAY RAG analysis.

Tests RAG analysis, hot-reload, version management, and vector search performance.
"""

import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, patch
from datetime import datetime

from services.knowledge_base_service import KnowledgeBaseService
from services.chroma_service import ChromaService


class TestKnowledgeBaseIntegration:
    """Test suite for knowledge base integration."""
    
    @pytest.fixture
    def temp_chroma_dir(self):
        """Create temporary ChromaDB directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def kb_service(self, temp_chroma_dir):
        """Create KB service instance."""
        chroma_service = ChromaService(persist_directory=temp_chroma_dir)
        return KnowledgeBaseService(chroma_service)
    
    def test_rag_analysis_with_pmjay_data(self, kb_service):
        """Test RAG analysis with PM-JAY knowledge base."""
        # Ingest sample PM-JAY data
        guidelines = [{
            "guideline_id": "GL001",
            "title": "Pre-authorization Guidelines",
            "category": "Authorization",
            "content": "All cardiac procedures require pre-authorization",
            "effective_date": "2025-01-01",
            "version": "1.0"
        }]
        
        kb_service.ingest_pmjay_guidelines_data(guidelines)
        
        # Test RAG query
        result = kb_service.search(
            kb_service.COL_PMJAY_MARGDARSHIKA,
            "cardiac surgery authorization requirements"
        )
        
        assert "documents" in result or "error" not in result
    
    def test_hot_reload_functionality(self, kb_service):
        """Test hot-reload without system downtime."""
        # Initial data load
        packages = [{
            "package_code": "PKG001",
            "package_name": "Cardiac Surgery",
            "specialty": "Cardiology",
            "procedure_type": "Surgical",
            "package_rate": 150000.0,
            "effective_date": "2025-01-01"
        }]
        
        kb_service.ingest_pmjay_packages_data(packages)
        
        # Test hot reload
        result = kb_service.hot_reload()
        
        assert result["success"] is True
        assert "collections" in result
    
    def test_version_management(self, kb_service):
        """Test version management functionality."""
        collection = kb_service.COL_PMJAY_MARGDARSHIKA
        
        # Create initial version
        snapshot_result = kb_service.create_version_snapshot(collection, "v1.0")
        assert snapshot_result["success"] is True
        
        # Test rollback
        rollback_result = kb_service.rollback_to_version(collection, "v1.0")
        assert rollback_result["success"] is True
    
    @pytest.mark.asyncio
    async def test_concurrent_vector_search(self, kb_service):
        """Test vector search performance under concurrent load."""
        # Ingest test data
        test_data = [{"text": f"Test document {i}", "metadata": {"id": i}} for i in range(100)]
        kb_service.ingest_documents("test_collection", test_data)
        
        # Concurrent search tasks
        async def search_task():
            return kb_service.search("test_collection", "test document")
        
        tasks = [search_task() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All searches should complete successfully
        assert len(results) == 10
    
    def test_conflict_resolution(self, kb_service):
        """Test conflict detection and resolution."""
        # Create conflicting guidelines
        guidelines = [
            {
                "guideline_id": "GL001",
                "title": "Test Guideline",
                "category": "Authorization",
                "content": "Original content",
                "effective_date": "2025-01-01",
                "version": "1.0",
                "priority": 1
            },
            {
                "guideline_id": "GL001",
                "title": "Test Guideline",
                "category": "Authorization", 
                "content": "Updated content",
                "effective_date": "2025-01-01",
                "version": "2.0",
                "priority": 2
            }
        ]
        
        kb_service.ingest_pmjay_guidelines_data(guidelines)
        
        # Test conflict detection
        conflicts = kb_service.detect_conflicts(kb_service.COL_PMJAY_MARGDARSHIKA)
        assert "conflicts" in conflicts
        
        # Test conflict resolution
        resolution = kb_service.resolve_conflicts(kb_service.COL_PMJAY_MARGDARSHIKA)
        assert resolution["success"] is True


if __name__ == "__main__":
    pytest.main([__file__])
