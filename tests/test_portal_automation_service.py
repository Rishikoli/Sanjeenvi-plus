"""Tests for Portal Automation Service components.

Tests authentication, form submission, status tracking, and retry mechanisms
with mocked portal responses to avoid external dependencies.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from decimal import Decimal

from services.portal_automation_service import (
    PortalAuthenticationService, PortalFormService, PortalStatusTracker,
    AuthenticationResult, SubmissionResult, StatusTrackingResult, PortalStatus
)
from services.portal_retry_service import (
    PortalRetryService, PortalErrorHandler, RetryConfig, RetryStrategy
)
from models.medical import PatientInfo, MedicalRecord, MedicalProcedure, Diagnosis


class TestPortalAuthenticationService:
    """Test suite for Portal Authentication Service."""
    
    @pytest.fixture
    def auth_service(self):
        """Create authentication service for testing."""
        service = PortalAuthenticationService()
        service.username = "test_user"
        service.password = "test_pass"
        service.client_id = "test_client"
        service.client_secret = "test_secret"
        return service
    
    @pytest.fixture
    def mock_auth_response(self):
        """Mock successful authentication response."""
        return {
            "access_token": "test_token_123",
            "refresh_token": "refresh_token_123",
            "session_id": "session_123",
            "token_type": "Bearer",
            "expires_in": 7200
        }
    
    @pytest.mark.asyncio
    async def test_successful_authentication(self, auth_service, mock_auth_response):
        """Test successful portal authentication."""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_auth_response
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await auth_service.authenticate()
            
            assert result.success is True
            assert result.session_token == "test_token_123"
            assert result.portal_session_id == "session_123"
            assert result.error_message is None
            assert auth_service.current_session is not None
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, auth_service):
        """Test authentication failure handling."""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock failed response
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Invalid credentials"
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await auth_service.authenticate()
            
            assert result.success is False
            assert result.session_token is None
            assert "401" in result.error_message
    
    @pytest.mark.asyncio
    async def test_session_refresh(self, auth_service, mock_auth_response):
        """Test session refresh functionality."""
        # Set up existing session
        auth_service.current_session = {
            "token": "old_token",
            "refresh_token": "refresh_123",
            "session_id": "session_123",
            "token_type": "Bearer"
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful refresh response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "access_token": "new_token_456",
                "refresh_token": "new_refresh_456",
                "expires_in": 7200
            }
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await auth_service.refresh_session()
            
            assert result.success is True
            assert result.session_token == "new_token_456"
            assert auth_service.current_session["token"] == "new_token_456"
    
    def test_session_validation(self, auth_service):
        """Test session validity checking."""
        # No session
        assert not auth_service._is_session_valid()
        
        # Valid session
        auth_service.current_session = {"token": "test"}
        auth_service.session_expires_at = datetime.now() + timedelta(hours=1)
        assert auth_service._is_session_valid()
        
        # Expired session
        auth_service.session_expires_at = datetime.now() - timedelta(hours=1)
        assert not auth_service._is_session_valid()
    
    def test_auth_headers(self, auth_service):
        """Test authentication header generation."""
        auth_service.current_session = {
            "token": "test_token",
            "token_type": "Bearer"
        }
        
        headers = auth_service.get_auth_headers()
        
        assert headers["Authorization"] == "Bearer test_token"
        assert headers["Content-Type"] == "application/json"


class TestPortalFormService:
    """Test suite for Portal Form Service."""
    
    @pytest.fixture
    def auth_service_mock(self):
        """Mock authentication service."""
        mock_auth = Mock(spec=PortalAuthenticationService)
        mock_auth.portal_base_url = "https://test-portal.gov.in/api"
        mock_auth.ensure_authenticated = AsyncMock(return_value=AuthenticationResult(
            success=True,
            session_token="test_token",
            expires_at=datetime.now() + timedelta(hours=1),
            error_message=None,
            portal_session_id="session_123"
        ))
        mock_auth.get_auth_headers.return_value = {
            "Authorization": "Bearer test_token",
            "Content-Type": "application/json"
        }
        return mock_auth
    
    @pytest.fixture
    def form_service(self, auth_service_mock):
        """Create form service for testing."""
        return PortalFormService(auth_service_mock)
    
    @pytest.fixture
    def sample_medical_record(self):
        """Sample medical record for testing."""
        patient = PatientInfo(
            patient_id="P12345",
            name="John Doe",
            age=45,
            gender="Male"
        )
        
        procedure = MedicalProcedure(
            procedure_name="Cardiac Surgery",
            procedure_code="CARD-001",
            procedure_date=datetime(2024, 1, 16)
        )
        
        diagnosis = Diagnosis(
            diagnosis_name="Heart Disease",
            diagnosis_code="I25.1",
            diagnosis_date=datetime(2024, 1, 15)
        )
        
        return MedicalRecord(
            patient_info=patient,
            hospital_id="H001",
            admission_date=datetime(2024, 1, 15),
            discharge_date=datetime(2024, 1, 20),
            procedures=[procedure],
            diagnoses=[diagnosis],
            total_amount=Decimal("150000.00"),
            document_confidence=0.92
        )
    
    @pytest.fixture
    def package_recommendation(self):
        """Sample package recommendation."""
        return {
            "package_code": "CARD-PKG-001",
            "package_name": "Cardiac Surgery Package",
            "confidence": 0.95
        }
    
    @pytest.mark.asyncio
    async def test_form_population(self, form_service, sample_medical_record, package_recommendation):
        """Test form data population from medical record."""
        form_data = await form_service.populate_claim_form(sample_medical_record, package_recommendation)
        
        assert form_data["patient_id"] == "P12345"
        assert form_data["patient_name"] == "John Doe"
        assert form_data["hospital_id"] == "H001"
        assert form_data["total_amount"] == "150000.00"
        assert form_data["recommended_package_code"] == "CARD-PKG-001"
        assert len(form_data["procedures"]) == 1
        assert len(form_data["diagnoses"]) == 1
    
    @pytest.mark.asyncio
    async def test_successful_form_submission(self, form_service):
        """Test successful form submission."""
        form_data = {"patient_id": "P123", "total_amount": "50000"}
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful submission response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "submission_id": "SUB123",
                "portal_reference": "REF456"
            }
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await form_service.submit_claim_form(form_data, "CLAIM123")
            
            assert result.success is True
            assert result.submission_id == "SUB123"
            assert result.portal_reference == "REF456"
            assert result.status == PortalStatus.FORM_SUBMITTED
    
    @pytest.mark.asyncio
    async def test_rate_limited_submission(self, form_service):
        """Test handling of rate-limited submission."""
        form_data = {"patient_id": "P123"}
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock rate limit response
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "300"}
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await form_service.submit_claim_form(form_data, "CLAIM123")
            
            assert result.success is False
            assert result.retry_after == 300
            assert result.status == PortalStatus.SUBMISSION_FAILED
    
    @pytest.mark.asyncio
    async def test_form_validation(self, form_service):
        """Test form data validation."""
        # Valid form data
        valid_data = {
            "patient_id": "P123",
            "patient_name": "John Doe",
            "patient_age": 45,
            "patient_gender": "Male",
            "hospital_id": "H001",
            "admission_date": "2024-01-15",
            "discharge_date": "2024-01-20",
            "total_amount": "50000"
        }
        
        result = await form_service.validate_form_data(valid_data)
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        
        # Invalid form data
        invalid_data = {
            "patient_age": "invalid_age",
            "total_amount": "-1000"
        }
        
        result = await form_service.validate_form_data(invalid_data)
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0


class TestPortalRetryService:
    """Test suite for Portal Retry Service."""
    
    @pytest.fixture
    def retry_config(self):
        """Create retry configuration for testing."""
        return RetryConfig(
            max_attempts=3,
            base_delay=0.1,  # Short delays for testing
            max_delay=1.0,
            backoff_multiplier=2.0
        )
    
    @pytest.fixture
    def retry_service(self, retry_config):
        """Create retry service for testing."""
        return PortalRetryService(retry_config)
    
    @pytest.mark.asyncio
    async def test_successful_operation_no_retry(self, retry_service):
        """Test successful operation on first attempt."""
        async def successful_operation():
            return "success"
        
        result = await retry_service.retry_async_operation(successful_operation, "test_op")
        
        assert result.success is True
        assert result.final_result == "success"
        assert result.total_attempts == 1
        assert len(result.attempts) == 1
        assert result.attempts[0].success is True
    
    @pytest.mark.asyncio
    async def test_operation_succeeds_after_retries(self, retry_service):
        """Test operation that succeeds after initial failures."""
        call_count = 0
        
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = await retry_service.retry_async_operation(flaky_operation, "flaky_op")
        
        assert result.success is True
        assert result.final_result == "success"
        assert result.total_attempts == 3
        assert len(result.attempts) == 3
        assert result.attempts[-1].success is True
    
    @pytest.mark.asyncio
    async def test_operation_fails_after_max_attempts(self, retry_service):
        """Test operation that fails after maximum attempts."""
        async def failing_operation():
            raise Exception("Persistent failure")
        
        result = await retry_service.retry_async_operation(failing_operation, "failing_op")
        
        assert result.success is False
        assert result.total_attempts == 3  # max_attempts
        assert len(result.attempts) == 3
        assert all(not attempt.success for attempt in result.attempts)
        assert "Persistent failure" in result.final_error
    
    def test_delay_calculation(self, retry_service):
        """Test delay calculation for different strategies."""
        # Exponential backoff
        delay1 = retry_service._calculate_delay(1)
        delay2 = retry_service._calculate_delay(2)
        delay3 = retry_service._calculate_delay(3)
        
        # Should increase exponentially (with some jitter tolerance)
        assert delay1 < delay2 < delay3
        
        # Should respect max delay
        large_delay = retry_service._calculate_delay(10)
        assert large_delay <= retry_service.config.max_delay
    
    def test_should_retry_logic(self, retry_service):
        """Test retry decision logic."""
        # Should retry on timeout
        timeout_error = Exception("Request timeout occurred")
        assert retry_service._should_retry(timeout_error, 1) is True
        
        # Should not retry after max attempts
        assert retry_service._should_retry(timeout_error, 3) is False
        
        # Should retry on server errors
        server_error = Exception("500 Internal Server Error")
        assert retry_service._should_retry(server_error, 1) is True


class TestPortalErrorHandler:
    """Test suite for Portal Error Handler."""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing."""
        retry_config = RetryConfig(max_attempts=2, base_delay=0.1)
        retry_service = PortalRetryService(retry_config)
        return PortalErrorHandler(retry_service)
    
    @pytest.mark.asyncio
    async def test_submission_with_validation_failure(self, error_handler):
        """Test submission handling with validation failure."""
        invalid_form_data = {}  # Missing required fields
        
        with patch('services.portal_automation_service.portal_form_service') as mock_form_service:
            mock_form_service.validate_form_data = AsyncMock(return_value={
                "is_valid": False,
                "errors": ["Missing required field: patient_id"]
            })
            
            result = await error_handler.handle_submission_with_retry(invalid_form_data, "CLAIM123")
            
            assert result.success is False
            assert "validation failed" in result.error_message.lower()
            assert result.status == PortalStatus.SUBMISSION_FAILED
    
    def test_retry_statistics(self, error_handler):
        """Test retry statistics reporting."""
        stats = error_handler.get_retry_statistics()
        
        assert "max_attempts" in stats
        assert "base_delay" in stats
        assert "strategy" in stats
        assert isinstance(stats["max_attempts"], int)
        assert isinstance(stats["base_delay"], float)


@pytest.mark.integration
class TestPortalIntegration:
    """Integration tests for portal automation components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_submission_flow(self):
        """Test complete submission flow with mocked portal."""
        # This would test the complete flow from authentication to submission
        # In a real scenario, this might use a test portal or comprehensive mocks
        pass
    
    @pytest.mark.asyncio
    async def test_concurrent_submissions(self):
        """Test handling of concurrent claim submissions."""
        # This would test the system's ability to handle multiple concurrent submissions
        pass
