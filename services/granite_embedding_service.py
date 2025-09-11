"""IBM Granite embedding service for vector generation and text processing."""

import logging
import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GraniteEmbeddingService:
    """Service for generating embeddings using IBM Granite models."""
    
    def __init__(self):
        """Initialize Granite embedding service."""
        self.hf_token = os.getenv("HF_TOKEN")
        self.granite_embedding_model = os.getenv("GRANITE_EMBEDDING_MODEL", "ibm-granite/granite-embedding-50m-english")
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.sentence_transformer = None
        
        # Configuration
        self.max_sequence_length = 512
        self.batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            # Try to load IBM Granite embedding model
            if self.hf_token:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.granite_embedding_model,
                        token=self.hf_token,
                        trust_remote_code=True
                    )
                    self.model = AutoModel.from_pretrained(
                        self.granite_embedding_model,
                        token=self.hf_token,
                        trust_remote_code=True
                    ).to(self.device)
                    
                    logger.info(f"Loaded IBM Granite embedding model: {self.granite_embedding_model}")
                    return
                    
                except Exception as e:
                    logger.warning(f"Failed to load IBM Granite model: {e}")
            
            # Fallback to sentence-transformers with a multilingual model
            try:
                self.sentence_transformer = SentenceTransformer(
                    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                )
                logger.info("Loaded fallback multilingual sentence transformer")
                
            except Exception as e:
                logger.error(f"Failed to load fallback model: {e}")
                # Use a basic model as last resort
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded basic sentence transformer as last resort")
                
        except Exception as e:
            logger.error(f"Failed to initialize any embedding model: {e}")
            raise
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of input texts
            normalize: Whether to normalize embeddings
            show_progress: Whether to show progress for large batches
            
        Returns:
            NumPy array of embeddings with shape (len(texts), embedding_dim)
        """
        try:
            if not texts:
                return np.array([])
            
            start_time = time.time()
            
            # Clean and validate texts
            cleaned_texts = [self._preprocess_text(text) for text in texts]
            
            if self.model and self.tokenizer:
                # Use IBM Granite model
                embeddings = self._generate_granite_embeddings(cleaned_texts, normalize)
            elif self.sentence_transformer:
                # Use sentence transformer
                embeddings = self._generate_sentence_transformer_embeddings(
                    cleaned_texts, normalize, show_progress
                )
            else:
                raise RuntimeError("No embedding model available")
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Generated {len(embeddings)} embeddings in {processing_time:.2f}ms")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def _generate_granite_embeddings(self, texts: List[str], normalize: bool) -> np.ndarray:
        """Generate embeddings using IBM Granite model."""
        try:
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_sequence_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                    # Use mean pooling of last hidden states
                    embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                    
                    if normalize:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                    all_embeddings.append(embeddings.cpu().numpy())
            
            return np.vstack(all_embeddings)
            
        except Exception as e:
            logger.error(f"Granite embedding generation failed: {e}")
            raise
    
    def _generate_sentence_transformer_embeddings(
        self, 
        texts: List[str], 
        normalize: bool,
        show_progress: bool
    ) -> np.ndarray:
        """Generate embeddings using sentence transformer."""
        try:
            embeddings = self.sentence_transformer.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                batch_size=self.batch_size
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Sentence transformer embedding generation failed: {e}")
            raise
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling to token embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding generation."""
        if not text:
            return ""
        
        # Clean whitespace
        cleaned = " ".join(text.split())
        
        # Truncate if too long (keep some buffer for tokenization)
        max_chars = self.max_sequence_length * 4  # Rough estimate
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars]
        
        return cleaned
    
    def generate_single_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate embedding for a single text."""
        embeddings = self.generate_embeddings([text], normalize=normalize)
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            # Normalize embeddings if not already normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            normalized1 = embedding1 / norm1
            normalized2 = embedding2 / norm2
            
            # Compute cosine similarity
            similarity = np.dot(normalized1, normalized2)
            
            # Ensure result is in valid range [-1, 1]
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def batch_similarity(self, query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Compute similarity between a query embedding and multiple embeddings."""
        try:
            if len(embeddings) == 0:
                return np.array([])
            
            # Normalize query embedding
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return np.zeros(len(embeddings))
            
            normalized_query = query_embedding / query_norm
            
            # Normalize all embeddings
            embedding_norms = np.linalg.norm(embeddings, axis=1)
            # Avoid division by zero
            embedding_norms = np.where(embedding_norms == 0, 1e-8, embedding_norms)
            normalized_embeddings = embeddings / embedding_norms[:, np.newaxis]
            
            # Compute similarities
            similarities = np.dot(normalized_embeddings, normalized_query)
            
            # Ensure results are in valid range
            similarities = np.clip(similarities, -1.0, 1.0)
            
            return similarities
            
        except Exception as e:
            logger.error(f"Failed to compute batch similarities: {e}")
            return np.zeros(len(embeddings))
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        try:
            if self.model:
                return self.model.config.hidden_size
            elif self.sentence_transformer:
                return self.sentence_transformer.get_sentence_embedding_dimension()
            else:
                return 384  # Default dimension for many models
                
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {e}")
            return 384
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the embedding service is healthy."""
        try:
            # Test with a simple text
            test_text = "This is a test sentence for embedding generation."
            embedding = self.generate_single_embedding(test_text)
            
            return {
                "status": "healthy",
                "model_type": "granite" if self.model else "sentence_transformer",
                "model_name": self.granite_embedding_model,
                "embedding_dimension": len(embedding),
                "device": self.device,
                "test_embedding_shape": embedding.shape
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": self.model is not None or self.sentence_transformer is not None
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "granite_model": self.granite_embedding_model,
            "has_granite_model": self.model is not None,
            "has_sentence_transformer": self.sentence_transformer is not None,
            "device": self.device,
            "max_sequence_length": self.max_sequence_length,
            "batch_size": self.batch_size,
            "embedding_dimension": self.get_embedding_dimension()
        }


# Global embedding service instance
granite_embedding_service = GraniteEmbeddingService()
