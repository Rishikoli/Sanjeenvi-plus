"""ChromaDB service for vector storage and retrieval operations."""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ChromaService:
    """Service for managing ChromaDB collections and vector operations."""
    
    def __init__(self, persist_directory: str = ".chroma"):
        """Initialize ChromaDB client with persistent storage."""
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function for IBM Granite
        self.embedding_function = self._get_embedding_function()
        
        # Collection references
        self.pmjay_collection = None
        self.medical_codes_collection = None
        
        logger.info(f"ChromaDB initialized with persist directory: {self.persist_directory}")
    
    def _get_embedding_function(self):
        """Get IBM Granite embedding function."""
        try:
            # Use Hugging Face embedding function with IBM Granite model
            hf_token = os.getenv("HF_TOKEN")
            granite_model = os.getenv("GRANITE_EMBEDDING_MODEL", "ibm-granite/granite-embedding-50m-english")
            
            if hf_token:
                return embedding_functions.HuggingFaceEmbeddingFunction(
                    api_key=hf_token,
                    model_name=granite_model
                )
            else:
                logger.warning("HF_TOKEN not found, using default sentence transformer")
                return embedding_functions.DefaultEmbeddingFunction()
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding function: {e}")
            # Fallback to default embedding function
            return embedding_functions.DefaultEmbeddingFunction()
    
    def create_pmjay_collection(self) -> chromadb.Collection:
        """Create or get PM-JAY guidelines collection."""
        try:
            collection_name = "pmjay_guidelines"
            
            # Try to get existing collection first
            try:
                self.pmjay_collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Retrieved existing collection: {collection_name}")
            except (ValueError, Exception):
                # Collection doesn't exist, create it
                self.pmjay_collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"description": "PM-JAY guidelines and package information"}
                )
                logger.info(f"Created new collection: {collection_name}")
            
            return self.pmjay_collection
            
        except Exception as e:
            logger.error(f"Failed to create PM-JAY collection: {e}")
            raise
    
    def create_medical_codes_collection(self) -> chromadb.Collection:
        """Create or get medical codes collection."""
        try:
            collection_name = "medical_codes"
            
            # Try to get existing collection first
            try:
                self.medical_codes_collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Retrieved existing collection: {collection_name}")
            except (ValueError, Exception):
                # Collection doesn't exist, create it
                self.medical_codes_collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"description": "Medical codes and procedures mapping"}
                )
                logger.info(f"Created new collection: {collection_name}")
            
            return self.medical_codes_collection
            
        except Exception as e:
            logger.error(f"Failed to create medical codes collection: {e}")
            raise
    
    def add_pmjay_documents(
        self, 
        documents: List[str], 
        metadatas: List[Dict[str, Any]], 
        ids: List[str]
    ) -> None:
        """Add PM-JAY guideline documents to the collection."""
        try:
            if not self.pmjay_collection:
                self.create_pmjay_collection()
            
            self.pmjay_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to PM-JAY collection")
            
        except Exception as e:
            logger.error(f"Failed to add PM-JAY documents: {e}")
            raise
    
    def add_medical_codes(
        self, 
        documents: List[str], 
        metadatas: List[Dict[str, Any]], 
        ids: List[str]
    ) -> None:
        """Add medical codes to the collection."""
        try:
            if not self.medical_codes_collection:
                self.create_medical_codes_collection()
            
            self.medical_codes_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} medical codes to collection")
            
        except Exception as e:
            logger.error(f"Failed to add medical codes: {e}")
            raise
    
    def search_pmjay_guidelines(
        self, 
        query: str, 
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search PM-JAY guidelines using vector similarity."""
        try:
            if not self.pmjay_collection:
                self.create_pmjay_collection()
            
            start_time = time.time()
            
            results = self.pmjay_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Extract similarity scores (distances converted to similarities)
            distances = results.get('distances', [[]])[0]
            similarities = [1 - d for d in distances] if distances else []
            
            search_result = {
                'documents': results.get('documents', [[]])[0],
                'metadatas': results.get('metadatas', [[]])[0],
                'ids': results.get('ids', [[]])[0],
                'similarities': similarities,
                'query_time_ms': query_time,
                'result_count': len(results.get('documents', [[]])[0])
            }
            
            logger.info(f"PM-JAY search completed in {query_time:.2f}ms, found {search_result['result_count']} results")
            return search_result
            
        except Exception as e:
            logger.error(f"Failed to search PM-JAY guidelines: {e}")
            raise
    
    def search_medical_codes(
        self, 
        query: str, 
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search medical codes using vector similarity."""
        try:
            if not self.medical_codes_collection:
                self.create_medical_codes_collection()
            
            start_time = time.time()
            
            results = self.medical_codes_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            query_time = (time.time() - start_time) * 1000
            
            # Extract similarity scores
            distances = results.get('distances', [[]])[0]
            similarities = [1 - d for d in distances] if distances else []
            
            search_result = {
                'documents': results.get('documents', [[]])[0],
                'metadatas': results.get('metadatas', [[]])[0],
                'ids': results.get('ids', [[]])[0],
                'similarities': similarities,
                'query_time_ms': query_time,
                'result_count': len(results.get('documents', [[]])[0])
            }
            
            logger.info(f"Medical codes search completed in {query_time:.2f}ms, found {search_result['result_count']} results")
            return search_result
            
        except Exception as e:
            logger.error(f"Failed to search medical codes: {e}")
            raise
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            
            return {
                'name': collection_name,
                'count': count,
                'metadata': collection.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info for {collection_name}: {e}")
            raise
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False
    
    def reset_database(self) -> bool:
        """Reset the entire ChromaDB (for testing purposes)."""
        try:
            self.client.reset()
            logger.info("ChromaDB reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check if ChromaDB is healthy."""
        try:
            # Try to list collections
            collections = self.client.list_collections()
            logger.info(f"ChromaDB health check passed. Found {len(collections)} collections")
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False


# Global ChromaDB service instance
chroma_service = ChromaService()
