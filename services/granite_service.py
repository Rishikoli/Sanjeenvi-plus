"""IBM Granite models integration service for language detection and AI reasoning."""

import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GraniteLanguageDetectionService:
    """Service for language detection using IBM Granite models."""
    
    def __init__(self):
        """Initialize language detection service."""
        self.hf_token = os.getenv("HF_TOKEN")
        self.granite_model = os.getenv("GRANITE_TEXT_MODEL", "ibm-granite/granite-3.3-8b-instruct")
        
        # Initialize language detection pipeline
        self.language_detector = None
        self.supported_languages = ["hindi", "marathi", "english"]
        self.language_codes = {"hi": "hindi", "mr": "marathi", "en": "english"}
        
        self._initialize_language_detector()
    
    def _initialize_language_detector(self):
        """Initialize the language detection pipeline."""
        try:
            # Use a lightweight multilingual model for language detection
            # Since IBM Granite might not have a specific language detection model,
            # we'll use a proven multilingual model and integrate with Granite for text processing
            self.language_detector = pipeline(
                "text-classification",
                model="papluca/xlm-roberta-base-language-detection",
                tokenizer="papluca/xlm-roberta-base-language-detection"
            )
            
            logger.info("Language detection pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize language detection: {e}")
            # Fallback to rule-based detection
            self.language_detector = None
    
    def detect_language(self, text: str, confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Detect language of input text with confidence scoring.
        
        Args:
            text: Input text to analyze
            confidence_threshold: Minimum confidence required for detection
            
        Returns:
            Dictionary with detected language, confidence, and metadata
        """
        try:
            start_time = time.time()
            
            if not text or not text.strip():
                return {
                    "detected_language": "unknown",
                    "confidence": 0.0,
                    "supported": False,
                    "processing_time_ms": 0,
                    "error": "Empty text provided"
                }
            
            # Clean and prepare text
            cleaned_text = self._preprocess_text(text)
            
            if self.language_detector:
                # Use ML-based detection
                result = self._ml_language_detection(cleaned_text)
            else:
                # Fallback to rule-based detection
                result = self._rule_based_language_detection(cleaned_text)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Check if detected language is supported
            detected_lang = result.get("detected_language", "unknown")
            is_supported = detected_lang.lower() in self.supported_languages
            
            # Apply confidence threshold
            confidence = result.get("confidence", 0.0)
            if confidence < confidence_threshold:
                logger.warning(f"Language detection confidence {confidence:.2f} below threshold {confidence_threshold}")
            
            return {
                "detected_language": detected_lang,
                "confidence": confidence,
                "supported": is_supported,
                "processing_time_ms": processing_time,
                "all_predictions": result.get("all_predictions", []),
                "method": result.get("method", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {
                "detected_language": "unknown",
                "confidence": 0.0,
                "supported": False,
                "processing_time_ms": 0,
                "error": str(e)
            }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for language detection."""
        # Remove extra whitespace and normalize
        cleaned = " ".join(text.split())
        
        # Take a sample if text is too long (first 500 characters should be enough)
        if len(cleaned) > 500:
            cleaned = cleaned[:500]
        
        return cleaned
    
    def _ml_language_detection(self, text: str) -> Dict[str, Any]:
        """ML-based language detection using transformers."""
        try:
            predictions = self.language_detector(text)
            
            # Convert predictions to our format
            all_predictions = []
            top_prediction = None
            
            if isinstance(predictions, list) and len(predictions) > 0:
                for pred in predictions:
                    lang_code = pred.get("label", "unknown")
                    confidence = pred.get("score", 0.0)
                    
                    # Map language codes to full names
                    lang_name = self.language_codes.get(lang_code, lang_code)
                    
                    pred_info = {
                        "language": lang_name,
                        "confidence": confidence,
                        "code": lang_code
                    }
                    all_predictions.append(pred_info)
                    
                    if top_prediction is None:
                        top_prediction = pred_info
                
                # Sort by confidence
                all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
            
            if top_prediction:
                return {
                    "detected_language": top_prediction["language"],
                    "confidence": top_prediction["confidence"],
                    "all_predictions": all_predictions,
                    "method": "ml_transformer"
                }
            else:
                return {
                    "detected_language": "unknown",
                    "confidence": 0.0,
                    "all_predictions": [],
                    "method": "ml_transformer_failed"
                }
                
        except Exception as e:
            logger.error(f"ML language detection failed: {e}")
            return self._rule_based_language_detection(text)
    
    def _rule_based_language_detection(self, text: str) -> Dict[str, Any]:
        """Rule-based language detection as fallback."""
        try:
            # Simple heuristic-based detection
            text_lower = text.lower()
            
            # Hindi indicators (Devanagari script detection would be better)
            hindi_indicators = ["है", "की", "का", "के", "में", "से", "को", "पर", "और", "या"]
            hindi_score = sum(1 for indicator in hindi_indicators if indicator in text_lower)
            
            # Marathi indicators
            marathi_indicators = ["आहे", "च्या", "ला", "ने", "मध्ये", "आणि", "किंवा", "तर", "पण"]
            marathi_score = sum(1 for indicator in marathi_indicators if indicator in text_lower)
            
            # English indicators
            english_indicators = ["the", "and", "or", "is", "are", "was", "were", "in", "on", "at"]
            english_score = sum(1 for indicator in english_indicators if indicator in text_lower)
            
            # Determine language based on scores
            scores = {
                "hindi": hindi_score / len(hindi_indicators),
                "marathi": marathi_score / len(marathi_indicators),
                "english": english_score / len(english_indicators)
            }
            
            # Find the language with highest score
            detected_lang = max(scores.keys(), key=lambda k: scores[k])
            confidence = min(scores[detected_lang], 1.0)  # Cap at 1.0
            
            # If all scores are very low, default to English
            if confidence < 0.1:
                detected_lang = "english"
                confidence = 0.5  # Low confidence default
            
            all_predictions = [
                {"language": lang, "confidence": score, "code": lang[:2]}
                for lang, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ]
            
            return {
                "detected_language": detected_lang,
                "confidence": confidence,
                "all_predictions": all_predictions,
                "method": "rule_based"
            }
            
        except Exception as e:
            logger.error(f"Rule-based language detection failed: {e}")
            return {
                "detected_language": "english",
                "confidence": 0.3,
                "all_predictions": [],
                "method": "fallback_default"
            }
    
    def detect_multilingual_document(self, text: str) -> Dict[str, Any]:
        """Detect multiple languages in a single document."""
        try:
            # Split text into paragraphs or sections
            sections = self._split_text_sections(text)
            
            section_results = []
            language_counts = {}
            
            for i, section in enumerate(sections):
                if len(section.strip()) < 10:  # Skip very short sections
                    continue
                
                result = self.detect_language(section)
                section_info = {
                    "section_index": i,
                    "text_preview": section[:100] + "..." if len(section) > 100 else section,
                    "detected_language": result["detected_language"],
                    "confidence": result["confidence"]
                }
                section_results.append(section_info)
                
                # Count languages
                lang = result["detected_language"]
                language_counts[lang] = language_counts.get(lang, 0) + 1
            
            # Determine primary language
            primary_language = max(language_counts.keys(), key=lambda k: language_counts[k]) if language_counts else "unknown"
            
            # Check if document is multilingual
            is_multilingual = len([lang for lang, count in language_counts.items() if count > 0]) > 1
            
            return {
                "primary_language": primary_language,
                "is_multilingual": is_multilingual,
                "language_distribution": language_counts,
                "section_results": section_results,
                "total_sections": len(section_results)
            }
            
        except Exception as e:
            logger.error(f"Multilingual document detection failed: {e}")
            return {
                "primary_language": "unknown",
                "is_multilingual": False,
                "language_distribution": {},
                "section_results": [],
                "error": str(e)
            }
    
    def _split_text_sections(self, text: str) -> List[str]:
        """Split text into logical sections for multilingual analysis."""
        # Split by double newlines (paragraphs)
        sections = text.split('\n\n')
        
        # If no paragraph breaks, split by single newlines
        if len(sections) == 1:
            sections = text.split('\n')
        
        # If still one section and it's long, split by sentences
        if len(sections) == 1 and len(text) > 1000:
            # Simple sentence splitting (can be improved)
            import re
            sentences = re.split(r'[.!?]+', text)
            sections = [s.strip() for s in sentences if s.strip()]
        
        return [section.strip() for section in sections if section.strip()]
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.supported_languages.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the language detection service is healthy."""
        try:
            # Test with a simple English text
            test_result = self.detect_language("This is a test sentence in English.")
            
            return {
                "status": "healthy",
                "model_loaded": self.language_detector is not None,
                "supported_languages": self.supported_languages,
                "test_detection": test_result.get("detected_language"),
                "test_confidence": test_result.get("confidence")
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": False
            }


# Global language detection service instance
granite_language_service = GraniteLanguageDetectionService()
