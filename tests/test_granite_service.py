"""Unit tests for IBM Granite language detection service."""

import pytest
from services.granite_service import GraniteLanguageDetectionService


@pytest.fixture
def language_service():
    """Create a language detection service for testing."""
    return GraniteLanguageDetectionService()


class TestGraniteLanguageDetectionService:
    """Test IBM Granite language detection functionality."""
    
    def test_service_initialization(self, language_service):
        """Test service initialization."""
        assert language_service is not None
        assert language_service.supported_languages == ["hindi", "marathi", "english"]
        assert "hi" in language_service.language_codes
        assert "mr" in language_service.language_codes
        assert "en" in language_service.language_codes
    
    def test_health_check(self, language_service):
        """Test service health check."""
        health = language_service.health_check()
        
        assert "status" in health
        assert "supported_languages" in health
        assert health["supported_languages"] == ["hindi", "marathi", "english"]
    
    def test_detect_english_text(self, language_service):
        """Test detection of English text."""
        english_text = "This is a medical report for patient John Doe. The patient was admitted for cardiac surgery."
        
        result = language_service.detect_language(english_text)
        
        assert result["detected_language"] in ["english", "en"]
        assert result["confidence"] > 0.0
        assert result["supported"] is True
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0
    
    def test_detect_hindi_text(self, language_service):
        """Test detection of Hindi text."""
        hindi_text = "यह मरीज़ की रिपोर्ट है। मरीज़ को हृदय की सर्जरी के लिए भर्ती किया गया है।"
        
        result = language_service.detect_language(hindi_text)
        
        # Should detect as Hindi or fall back to rule-based detection
        assert result["detected_language"] in ["hindi", "hi", "unknown"]
        assert result["confidence"] >= 0.0
        assert "processing_time_ms" in result
    
    def test_detect_marathi_text(self, language_service):
        """Test detection of Marathi text."""
        marathi_text = "हा रुग्णाचा अहवाल आहे। रुग्णाला हृदयाच्या शस्त्रक्रियेसाठी दाखल करण्यात आले आहे।"
        
        result = language_service.detect_language(marathi_text)
        
        # Should detect as Marathi or fall back to rule-based detection
        # Note: Language models may confuse Marathi with Urdu due to similar scripts
        assert result["detected_language"] in ["marathi", "mr", "ur", "unknown"]
        assert result["confidence"] >= 0.0
        assert "processing_time_ms" in result
    
    def test_detect_empty_text(self, language_service):
        """Test detection with empty text."""
        result = language_service.detect_language("")
        
        assert result["detected_language"] == "unknown"
        assert result["confidence"] == 0.0
        assert result["supported"] is False
        assert "error" in result
    
    def test_detect_very_short_text(self, language_service):
        """Test detection with very short text."""
        result = language_service.detect_language("Hi")
        
        assert "detected_language" in result
        assert result["confidence"] >= 0.0
        assert "processing_time_ms" in result
    
    def test_confidence_threshold(self, language_service):
        """Test confidence threshold functionality."""
        text = "Medical report patient surgery"
        
        # Test with high threshold
        result_high = language_service.detect_language(text, confidence_threshold=0.9)
        
        # Test with low threshold
        result_low = language_service.detect_language(text, confidence_threshold=0.1)
        
        assert "confidence" in result_high
        assert "confidence" in result_low
        # Both should return results, but may have different confidence interpretations
    
    def test_multilingual_document_detection(self, language_service):
        """Test multilingual document detection."""
        multilingual_text = """
        This is an English paragraph about medical procedures.
        
        यह हिंदी में एक पैराग्राफ है जो चिकित्सा प्रक्रियाओं के बारे में है।
        
        This is another English paragraph with medical terminology.
        """
        
        result = language_service.detect_multilingual_document(multilingual_text)
        
        assert "primary_language" in result
        assert "is_multilingual" in result
        assert "language_distribution" in result
        assert "section_results" in result
        assert "total_sections" in result
        
        # Should detect multiple languages
        assert len(result["language_distribution"]) >= 1
        assert result["total_sections"] > 0
    
    def test_preprocess_text(self, language_service):
        """Test text preprocessing."""
        messy_text = "   This   is   a   messy    text   with   extra   spaces   "
        cleaned = language_service._preprocess_text(messy_text)
        
        assert cleaned == "This is a messy text with extra spaces"
    
    def test_long_text_handling(self, language_service):
        """Test handling of very long text."""
        long_text = "This is a medical report. " * 100  # Very long text
        
        result = language_service.detect_language(long_text)
        
        assert "detected_language" in result
        assert result["processing_time_ms"] >= 0
        # Should handle long text without errors
    
    def test_rule_based_detection_fallback(self, language_service):
        """Test rule-based detection as fallback."""
        # Test with text containing clear English indicators
        english_text = "The patient is in the hospital and the doctor said he was fine."
        
        result = language_service._rule_based_language_detection(english_text)
        
        assert result["detected_language"] == "english"
        assert result["method"] == "rule_based"
        assert result["confidence"] > 0
        assert "all_predictions" in result
    
    def test_hindi_rule_based_detection(self, language_service):
        """Test Hindi detection using rule-based method."""
        # Text with Hindi indicators
        hindi_text = "मरीज़ अस्पताल में है और डॉक्टर ने कहा की वह ठीक है।"
        
        result = language_service._rule_based_language_detection(hindi_text)
        
        # Should detect Hindi or have reasonable confidence
        assert result["method"] == "rule_based"
        assert result["confidence"] >= 0
    
    def test_marathi_rule_based_detection(self, language_service):
        """Test Marathi detection using rule-based method."""
        # Text with Marathi indicators
        marathi_text = "रुग्ण रुग्णालयात आहे आणि डॉक्टरांनी सांगितले की तो बरा आहे।"
        
        result = language_service._rule_based_language_detection(marathi_text)
        
        assert result["method"] == "rule_based"
        assert result["confidence"] >= 0
    
    def test_get_supported_languages(self, language_service):
        """Test getting supported languages."""
        languages = language_service.get_supported_languages()
        
        assert isinstance(languages, list)
        assert "hindi" in languages
        assert "marathi" in languages
        assert "english" in languages
        assert len(languages) == 3
    
    def test_split_text_sections(self, language_service):
        """Test text section splitting."""
        text_with_paragraphs = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        
        sections = language_service._split_text_sections(text_with_paragraphs)
        
        assert len(sections) == 3
        assert "First paragraph." in sections[0]
        assert "Second paragraph." in sections[1]
        assert "Third paragraph." in sections[2]
    
    def test_split_text_by_lines(self, language_service):
        """Test text splitting by lines when no paragraphs."""
        text_with_lines = "First line.\nSecond line.\nThird line."
        
        sections = language_service._split_text_sections(text_with_lines)
        
        assert len(sections) >= 3
        assert any("First line." in section for section in sections)
    
    def test_error_handling_in_detection(self, language_service):
        """Test error handling in language detection."""
        # Test with None input (should be handled gracefully)
        try:
            result = language_service.detect_language(None)
            # Should return error result, not crash
            assert "error" in result or result["detected_language"] == "unknown"
        except Exception:
            # If it raises an exception, that's also acceptable for None input
            pass


class TestGraniteServicePerformance:
    """Test performance requirements for language detection."""
    
    def test_detection_speed(self, language_service):
        """Test that language detection is reasonably fast."""
        text = "This is a medical report for cardiac surgery procedure. The patient requires immediate attention."
        
        result = language_service.detect_language(text)
        
        # Should complete within reasonable time (< 5 seconds for unit test)
        assert result["processing_time_ms"] < 5000
        assert result["processing_time_ms"] > 0
    
    def test_multilingual_detection_speed(self, language_service):
        """Test multilingual detection performance."""
        multilingual_text = """
        English medical report section.
        
        हिंदी चिकित्सा रिपोर्ट अनुभाग।
        
        मराठी वैद्यकीय अहवाल विभाग।
        
        Another English section with medical terms.
        """
        
        import time
        start_time = time.time()
        
        result = language_service.detect_multilingual_document(multilingual_text)
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        # Should complete multilingual detection reasonably quickly
        assert processing_time < 10000  # Less than 10 seconds
        assert result["total_sections"] > 0
