import pytest
import tempfile
import os

from services.chroma_service import ChromaService
from services.knowledge_base_service import KnowledgeBaseService


class TestKBMS:
    @pytest.fixture(scope="function")
    def kb_service(self):
        # Create a temporary directory with a simple path
        temp_dir = tempfile.mkdtemp(prefix='chroma_test_')
        try:
            # Ensure we use default embeddings to avoid network issues
            os.environ["USE_HF_EMBEDDINGS"] = "0"
            # Ensure the directory exists and is empty
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            os.makedirs(temp_dir, exist_ok=True)
            
            # Initialize Chroma with the temp directory
            chroma = ChromaService(persist_directory=temp_dir)
            yield KnowledgeBaseService(chroma)
        finally:
            # Cleanup
            try:
                if os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Failed to clean up {temp_dir}: {e}")

    def test_hot_reload_collections(self, kb_service: KnowledgeBaseService):
        result = kb_service.hot_reload()
        assert result.get("success") is True
        cols = result.get("collections", {})
        assert "pmjay_margdarshika" in cols
        assert "ayushman_package_suchi" in cols
        assert "rog_nidan_code_sangrah" in cols

    def test_ingest_simple_documents(self, kb_service: KnowledgeBaseService):
        collection = kb_service.COL_PMJAY_MARGDARSHIKA
        docs = [
            {
                "id": "guide_001",
                "text": "Pre-auth must be filed within 24 hours.",
                "metadata": {
                    "lang": "en",
                    "type": "guideline"
                }
            },
            {
                "id": "guide_002", 
                "text": "Discharge summary must include ICD codes.",
                "metadata": {
                    "lang": "en",
                    "type": "sop"
                }
            }
        ]

        ingest_result = kb_service.ingest_documents(collection, docs)
        assert ingest_result.get("success") is True
        assert ingest_result.get("count") == 2

    def test_search_functionality(self, kb_service: KnowledgeBaseService):
        collection = kb_service.COL_PMJAY_MARGDARSHIKA
        # First ingest a document
        docs = [{"id": "test_doc", "text": "Pre-authorization guidelines", "metadata": {"type": "test"}}]
        kb_service.ingest_documents(collection, docs)
        
        # Then search
        search_result = kb_service.search(collection, "authorization", n_results=5)
        assert isinstance(search_result, dict)
        assert "documents" in search_result
        assert "metadatas" in search_result
        assert "ids" in search_result

    def test_list_functionality(self, kb_service: KnowledgeBaseService):
        collection = kb_service.COL_PMJAY_MARGDARSHIKA
        # Ingest a document first
        docs = [{"id": "list_test", "text": "Test document for listing", "metadata": {"type": "test"}}]
        kb_service.ingest_documents(collection, docs)
        
        # List items
        listing = kb_service.list_items(collection, limit=10, offset=0)
        assert isinstance(listing, dict)
        assert "total" in listing
        assert "items" in listing

    def test_delete_functionality(self, kb_service: KnowledgeBaseService):
        collection = kb_service.COL_PMJAY_MARGDARSHIKA
        # Ingest a document first
        docs = [{"id": "delete_test", "text": "Test document for deletion", "metadata": {"type": "test"}}]
        kb_service.ingest_documents(collection, docs)
        
        # Delete the document
        del_result = kb_service.delete_items(collection, ["delete_test"])
        assert del_result.get("success") is True

    def test_stats(self, kb_service: KnowledgeBaseService):
        stats = kb_service.get_knowledge_base_stats()
        assert "pmjay_margdarshika" in stats
        assert "ayushman_package_suchi" in stats
        assert "rog_nidan_code_sangrah" in stats
        assert "total_documents" in stats
        assert "status" in stats
