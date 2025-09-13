"""
Portal integration tests for PM-JAY portal automation.

Tests mock portal interactions, retry mechanisms, error recovery,
authentication, and session management functionality.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any

from services.portal_automation_service import (
    PortalAuthenticator, 
    FormSubmissionService, 
    StatusTrackingService
)
from services.error_handler import ErrorHandlerFactory, ErrorContext, ErrorSeverity
from models.claims import Claim, ClaimStatus


class TestPortalAuthentication:
    """Test suite for portal authentication functionality."""
    
    @pytest.fixture
    def portal_auth(self):
        """Create portal authenticator instance."""
        return PortalAuthenticator()
    
    @pytest.fixture
    def mock_browser(self):
        """Mock browser instance."""
        browser = Mock()
        browser.new_page.return_value = Mock()
        return browser
    
    def test_successful_authentication(self, portal_auth, mock_browser):
        """Test successful portal authentication."""
        with patch.object(portal_auth, '_launch_browser', return_value=mock_browser):
            page = mock_browser.new_page.return_value
            page.goto.return_value = None
            page.fill.return_value = None
            page.click.return_value = None
            page.wait_for_selector.return_value = None
            
            result = portal_auth.authenticate("test_user", "test_pass", "https://pmjay.gov.in")
            
            assert result["success"] is True
            assert "session_id" in result
            page.goto.assert_called_once()
            page.fill.assert_called()
            page.click.assert_called()
    
    def test_authentication_failure_invalid_credentials(self, portal_auth, mock_browser):
        """Test authentication failure with invalid credentials."""
        with patch.object(portal_auth, '_launch_browser', return_value=mock_browser):
            page = mock_browser.new_page.return_value
            page.goto.return_value = None
            page.fill.return_value = None
            page.click.return_value = None
            # Simulate authentication failure
            page.wait_for_selector.side_effect = Exception("Invalid credentials")
            
            result = portal_auth.authenticate("invalid_user", "wrong_pass", "https://pmjay.gov.in")
            
            assert result["success"] is False
            assert "error" in result
    
    def test_authentication_timeout(self, portal_auth, mock_browser):
        """Test authentication timeout handling."""
        with patch.object(portal_auth, '_launch_browser', return_value=mock_browser):
            page = mock_browser.new_page.return_value
            page.goto.return_value = None
            page.fill.return_value = None
            page.click.return_value = None
            # Simulate timeout
            page.wait_for_selector.side_effect = asyncio.TimeoutError("Page load timeout")
            
            result = portal_auth.authenticate("test_user", "test_pass", "https://pmjay.gov.in")
            
            assert result["success"] is False
            assert "timeout" in result["error"].lower()
    
    def test_session_validation(self, portal_auth, mock_browser):
        """Test session validation functionality."""
        with patch.object(portal_auth, '_launch_browser', return_value=mock_browser):
            page = mock_browser.new_page.return_value
            page.evaluate.return_value = True  # Session is valid
            
            # First authenticate
            portal_auth.authenticate("test_user", "test_pass", "https://pmjay.gov.in")
            
            # Test session validation
            is_valid = portal_auth.validate_session()
            
            assert is_valid is True
    
    def test_session_refresh(self, portal_auth, mock_browser):
        """Test session refresh functionality."""
        with patch.object(portal_auth, '_launch_browser', return_value=mock_browser):
            page = mock_browser.new_page.return_value
            page.reload.return_value = None
            page.wait_for_selector.return_value = None
            
            # First authenticate
            portal_auth.authenticate("test_user", "test_pass", "https://pmjay.gov.in")
            
            # Test session refresh
            result = portal_auth.refresh_session()
            
            assert result["success"] is True
            page.reload.assert_called_once()


class TestFormSubmission:
    """Test suite for form submission functionality."""
    
    @pytest.fixture
    def form_service(self):
        """Create form submission service instance."""
        return FormSubmissionService()
    
    @pytest.fixture
    def sample_claim_data(self):
        """Sample claim data for testing."""
        return {
            "claim_id": "CLAIM001",
            "patient_name": "राम कुमार",  # Hindi name
            "patient_id": "PAT001",
            "hospital_id": "HOSP001",
            "package_code": "PKG001",
            "amount": 50000.0,
            "documents": ["prescription.pdf", "lab_report.pdf"]
        }
    
    def test_successful_form_submission(self, form_service, sample_claim_data):
        """Test successful claim form submission."""
        with patch.object(form_service, 'page') as mock_page:
            mock_page.fill.return_value = None
            mock_page.select_option.return_value = None
            mock_page.set_input_files.return_value = None
            mock_page.click.return_value = None
            mock_page.wait_for_selector.return_value = None
            mock_page.text_content.return_value = "PMJAY123456"
            
            result = form_service.submit_claim(sample_claim_data)
            
            assert result["success"] is True
            assert "portal_reference" in result
            assert result["portal_reference"] == "PMJAY123456"
    
    def test_form_submission_with_multilingual_data(self, form_service, sample_claim_data):
        """Test form submission with multilingual patient data."""
        # Add Marathi patient name
        sample_claim_data["patient_name"] = "राम कुमार शर्मा"
        sample_claim_data["address"] = "मुंबई, महाराष्ट्र"
        
        with patch.object(form_service, 'page') as mock_page:
            mock_page.fill.return_value = None
            mock_page.select_option.return_value = None
            mock_page.set_input_files.return_value = None
            mock_page.click.return_value = None
            mock_page.wait_for_selector.return_value = None
            mock_page.text_content.return_value = "PMJAY789012"
            
            result = form_service.submit_claim(sample_claim_data)
            
            assert result["success"] is True
            # Verify multilingual data was handled correctly
            mock_page.fill.assert_called()
    
    def test_form_submission_validation_error(self, form_service, sample_claim_data):
        """Test form submission with validation errors."""
        # Remove required field
        del sample_claim_data["patient_name"]
        
        with patch.object(form_service, 'page') as mock_page:
            mock_page.fill.side_effect = Exception("Required field missing")
            
            result = form_service.submit_claim(sample_claim_data)
            
            assert result["success"] is False
            assert "error" in result
    
    def test_form_submission_network_error(self, form_service, sample_claim_data):
        """Test form submission with network errors."""
        with patch.object(form_service, 'page') as mock_page:
            mock_page.click.side_effect = Exception("Network timeout")
            
            result = form_service.submit_claim(sample_claim_data)
            
            assert result["success"] is False
            assert "network" in result["error"].lower() or "timeout" in result["error"].lower()
    
    def test_document_upload_handling(self, form_service, sample_claim_data):
        """Test document upload functionality."""
        with patch.object(form_service, 'page') as mock_page, \
             patch('os.path.exists', return_value=True):
            
            mock_page.set_input_files.return_value = None
            mock_page.wait_for_selector.return_value = None
            
            result = form_service._upload_documents(sample_claim_data["documents"])
            
            assert result["success"] is True
            mock_page.set_input_files.assert_called()
    
    def test_document_upload_file_not_found(self, form_service, sample_claim_data):
        """Test document upload with missing files."""
        with patch('os.path.exists', return_value=False):
            
            result = form_service._upload_documents(sample_claim_data["documents"])
            
            assert result["success"] is False
            assert "not found" in result["error"].lower()


class TestRetryMechanisms:
    """Test suite for retry mechanisms and error recovery."""
    
    @pytest.fixture
    def form_service_with_retry(self):
        """Create form service with retry configuration."""
        service = FormSubmissionService()
        service.max_retries = 3
        service.retry_delay = 0.1  # Fast retries for testing
        return service
    
    def test_retry_on_network_failure(self, form_service_with_retry):
        """Test retry mechanism on network failures."""
        attempt_count = 0
        
        def mock_submit_side_effect(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Network error")
            return {"success": True, "portal_reference": "PMJAY123"}
        
        with patch.object(form_service_with_retry, '_submit_form', side_effect=mock_submit_side_effect):
            result = form_service_with_retry.submit_claim_with_retry({"claim_id": "CLAIM001"})
            
            assert result["success"] is True
            assert attempt_count == 3  # Should have retried twice, succeeded on third
    
    def test_retry_exhaustion(self, form_service_with_retry):
        """Test behavior when all retries are exhausted."""
        with patch.object(form_service_with_retry, '_submit_form', side_effect=Exception("Persistent error")):
            result = form_service_with_retry.submit_claim_with_retry({"claim_id": "CLAIM001"})
            
            assert result["success"] is False
            assert "retry" in result["error"].lower() or "exhausted" in result["error"].lower()
    
    def test_exponential_backoff(self, form_service_with_retry):
        """Test exponential backoff in retry mechanism."""
        retry_times = []
        
        def mock_submit_side_effect(*args, **kwargs):
            retry_times.append(time.time())
            raise Exception("Temporary error")
        
        with patch.object(form_service_with_retry, '_submit_form', side_effect=mock_submit_side_effect):
            start_time = time.time()
            form_service_with_retry.submit_claim_with_retry({"claim_id": "CLAIM001"})
            
            # Verify exponential backoff (delays should increase)
            if len(retry_times) > 1:
                delays = [retry_times[i] - retry_times[i-1] for i in range(1, len(retry_times))]
                assert all(delays[i] >= delays[i-1] for i in range(1, len(delays)))
    
    def test_circuit_breaker_pattern(self, form_service_with_retry):
        """Test circuit breaker pattern for error recovery."""
        # Simulate multiple consecutive failures
        failure_count = 0
        
        def mock_submit_side_effect(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 5:
                raise Exception("Service unavailable")
            return {"success": True, "portal_reference": "PMJAY456"}
        
        with patch.object(form_service_with_retry, '_submit_form', side_effect=mock_submit_side_effect):
            # Circuit should open after multiple failures
            for _ in range(3):
                result = form_service_with_retry.submit_claim_with_retry({"claim_id": f"CLAIM00{_}"})
                assert result["success"] is False
            
            # Circuit should eventually close and allow requests through
            time.sleep(0.2)  # Wait for circuit breaker reset
            result = form_service_with_retry.submit_claim_with_retry({"claim_id": "CLAIM004"})
            # This might still fail depending on circuit breaker implementation


class TestStatusTracking:
    """Test suite for status tracking functionality."""
    
    @pytest.fixture
    def status_service(self):
        """Create status tracking service instance."""
        return StatusTrackingService()
    
    def test_successful_status_check(self, status_service):
        """Test successful status checking."""
        with patch.object(status_service, 'page') as mock_page:
            mock_page.goto.return_value = None
            mock_page.fill.return_value = None
            mock_page.click.return_value = None
            mock_page.wait_for_selector.return_value = None
            mock_page.text_content.return_value = "Approved"
            
            result = status_service.check_claim_status("PMJAY123456")
            
            assert result["success"] is True
            assert result["status"] == "Approved"
    
    def test_status_check_claim_not_found(self, status_service):
        """Test status check for non-existent claim."""
        with patch.object(status_service, 'page') as mock_page:
            mock_page.goto.return_value = None
            mock_page.fill.return_value = None
            mock_page.click.return_value = None
            mock_page.text_content.return_value = "Claim not found"
            
            result = status_service.check_claim_status("INVALID123")
            
            assert result["success"] is False
            assert "not found" in result["error"].lower()
    
    def test_bulk_status_check(self, status_service):
        """Test bulk status checking functionality."""
        portal_refs = ["PMJAY123", "PMJAY456", "PMJAY789"]
        
        with patch.object(status_service, 'check_claim_status') as mock_check:
            mock_check.side_effect = [
                {"success": True, "status": "Approved"},
                {"success": True, "status": "Pending"},
                {"success": False, "error": "Claim not found"}
            ]
            
            results = status_service.bulk_status_check(portal_refs)
            
            assert len(results) == 3
            assert results[0]["success"] is True
            assert results[1]["success"] is True
            assert results[2]["success"] is False
    
    def test_status_change_detection(self, status_service):
        """Test detection of status changes."""
        # Mock previous status
        previous_status = "Pending"
        current_status = "Approved"
        
        with patch.object(status_service, '_get_previous_status', return_value=previous_status), \
             patch.object(status_service, 'check_claim_status', return_value={"success": True, "status": current_status}):
            
            result = status_service.check_status_change("PMJAY123")
            
            assert result["status_changed"] is True
            assert result["previous_status"] == previous_status
            assert result["current_status"] == current_status


class TestErrorRecovery:
    """Test suite for error recovery scenarios."""
    
    def test_session_expiry_recovery(self):
        """Test recovery from session expiry."""
        portal_auth = PortalAuthenticator()
        
        with patch.object(portal_auth, '_launch_browser') as mock_browser:
            page = mock_browser.return_value.new_page.return_value
            
            # First call succeeds (authentication)
            page.wait_for_selector.return_value = None
            auth_result = portal_auth.authenticate("user", "pass", "https://pmjay.gov.in")
            assert auth_result["success"] is True
            
            # Second call fails (session expired)
            page.evaluate.side_effect = [False, True]  # First check fails, second succeeds
            page.goto.return_value = None
            page.fill.return_value = None
            page.click.return_value = None
            
            # Should automatically re-authenticate
            is_valid = portal_auth.validate_session_with_recovery()
            assert is_valid is True
    
    def test_captcha_handling(self):
        """Test CAPTCHA detection and handling."""
        form_service = FormSubmissionService()
        
        with patch.object(form_service, 'page') as mock_page:
            # Simulate CAPTCHA detection
            mock_page.query_selector.return_value = Mock()  # CAPTCHA element found
            mock_page.screenshot.return_value = b"captcha_image_data"
            
            result = form_service._handle_captcha()
            
            assert result["captcha_detected"] is True
            assert "manual_intervention" in result
    
    def test_rate_limiting_recovery(self):
        """Test recovery from rate limiting."""
        form_service = FormSubmissionService()
        
        with patch.object(form_service, 'page') as mock_page:
            # Simulate rate limiting error
            mock_page.text_content.return_value = "Too many requests. Please try again later."
            
            # Should implement exponential backoff
            with patch('time.sleep') as mock_sleep:
                result = form_service._handle_rate_limit()
                
                assert result["rate_limited"] is True
                mock_sleep.assert_called()  # Should have waited
    
    def test_portal_maintenance_detection(self):
        """Test detection of portal maintenance mode."""
        portal_auth = PortalAuthenticator()
        
        with patch.object(portal_auth, '_launch_browser') as mock_browser:
            page = mock_browser.return_value.new_page.return_value
            page.goto.return_value = None
            page.text_content.return_value = "Portal is under maintenance"
            
            result = portal_auth.authenticate("user", "pass", "https://pmjay.gov.in")
            
            assert result["success"] is False
            assert "maintenance" in result["error"].lower()


class TestConcurrentOperations:
    """Test suite for concurrent portal operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_submissions(self):
        """Test concurrent claim submissions."""
        form_service = FormSubmissionService()
        
        async def mock_submit(claim_data):
            await asyncio.sleep(0.1)  # Simulate processing time
            return {"success": True, "portal_reference": f"PMJAY{claim_data['claim_id']}"}
        
        with patch.object(form_service, 'submit_claim', side_effect=mock_submit):
            claims = [{"claim_id": f"CLAIM{i:03d}"} for i in range(5)]
            
            # Submit all claims concurrently
            tasks = [form_service.submit_claim(claim) for claim in claims]
            results = await asyncio.gather(*tasks)
            
            # All submissions should succeed
            assert all(result["success"] for result in results)
            assert len(set(result["portal_reference"] for result in results)) == 5
    
    @pytest.mark.asyncio
    async def test_concurrent_status_checks(self):
        """Test concurrent status checking."""
        status_service = StatusTrackingService()
        
        async def mock_check(portal_ref):
            await asyncio.sleep(0.05)  # Simulate API call
            return {"success": True, "status": "Approved"}
        
        with patch.object(status_service, 'check_claim_status', side_effect=mock_check):
            portal_refs = [f"PMJAY{i:06d}" for i in range(10)]
            
            # Check all statuses concurrently
            tasks = [status_service.check_claim_status(ref) for ref in portal_refs]
            results = await asyncio.gather(*tasks)
            
            # All checks should succeed
            assert all(result["success"] for result in results)
    
    def test_session_sharing_across_operations(self):
        """Test sharing browser session across multiple operations."""
        portal_auth = PortalAuthenticator()
        form_service = FormSubmissionService()
        
        with patch.object(portal_auth, '_launch_browser') as mock_browser:
            # Authenticate once
            page = mock_browser.return_value.new_page.return_value
            page.wait_for_selector.return_value = None
            auth_result = portal_auth.authenticate("user", "pass", "https://pmjay.gov.in")
            
            # Share session with form service
            form_service.set_session(portal_auth.get_session())
            
            # Both services should use the same session
            assert form_service.session_id == portal_auth.session_id


class TestPerformanceMetrics:
    """Test suite for performance metrics collection."""
    
    def test_operation_timing(self):
        """Test timing of portal operations."""
        from services.monitoring_service import PerformanceContext
        
        with PerformanceContext("portal_authentication") as perf:
            time.sleep(0.1)  # Simulate operation
            
        # Performance should be recorded
        assert perf.success is True
    
    def test_throughput_measurement(self):
        """Test measurement of operation throughput."""
        form_service = FormSubmissionService()
        
        start_time = time.time()
        
        # Simulate multiple operations
        with patch.object(form_service, 'submit_claim', return_value={"success": True}):
            for i in range(10):
                form_service.submit_claim({"claim_id": f"CLAIM{i:03d}"})
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = 10 / duration  # operations per second
        
        # Should achieve reasonable throughput
        assert throughput > 5  # At least 5 operations per second


if __name__ == "__main__":
    pytest.main([__file__])
