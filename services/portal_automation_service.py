"""Portal automation service for PM-JAY government portal interactions.

This service handles authentication, form submission, and status tracking
for automated government portal submissions using WatsonX Orchestrate ADK.
"""

import logging
import os
import time
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import httpx
import json
from dotenv import load_dotenv
from services.watsonx_orchestrate_client import wxo_client, OrchestrateRun

load_dotenv()

logger = logging.getLogger(__name__)


class PortalStatus(Enum):
    """Portal interaction status enumeration."""
    AUTHENTICATED = "authenticated"
    SESSION_EXPIRED = "session_expired"
    AUTHENTICATION_FAILED = "authentication_failed"
    FORM_SUBMITTED = "form_submitted"
    SUBMISSION_FAILED = "submission_failed"
    STATUS_UPDATED = "status_updated"
    CONNECTION_ERROR = "connection_error"


@dataclass
class AuthenticationResult:
    """Result of portal authentication attempt."""
    success: bool
    session_token: Optional[str]
    expires_at: Optional[datetime]
    error_message: Optional[str]
    portal_session_id: Optional[str]


@dataclass
class SubmissionResult:
    """Result of form submission to portal."""
    success: bool
    submission_id: Optional[str]
    portal_reference: Optional[str]
    error_message: Optional[str]
    retry_after: Optional[int]
    status: PortalStatus


@dataclass
class StatusTrackingResult:
    """Result of status tracking query."""
    claim_status: str
    last_updated: datetime
    portal_messages: List[str]
    approval_status: Optional[str]
    rejection_reason: Optional[str]


class PortalAuthenticationService:
    """Handles PM-JAY portal authentication and session management."""
    
    def __init__(self):
        """Initialize portal authentication service."""
        self.portal_base_url = os.getenv("PMJAY_PORTAL_URL", "https://pmjay.gov.in/api")
        self.username = os.getenv("PMJAY_USERNAME")
        self.password = os.getenv("PMJAY_PASSWORD")
        self.client_id = os.getenv("PMJAY_CLIENT_ID")
        self.client_secret = os.getenv("PMJAY_CLIENT_SECRET")
        
        # Session management
        self.current_session: Optional[Dict[str, Any]] = None
        self.session_expires_at: Optional[datetime] = None
        self.max_session_duration = timedelta(hours=2)
        
        # HTTP client configuration
        self.timeout = httpx.Timeout(30.0)
        self.retry_attempts = 3
        
    async def authenticate(self) -> AuthenticationResult:
        """
        Authenticate with PM-JAY portal and establish session.
        
        Returns:
            AuthenticationResult with session details
        """
        try:
            # Check if current session is still valid
            if self._is_session_valid():
                logger.info("Using existing valid session")
                return AuthenticationResult(
                    success=True,
                    session_token=self.current_session["token"],
                    expires_at=self.session_expires_at,
                    error_message=None,
                    portal_session_id=self.current_session["session_id"]
                )
            
            # Perform new authentication
            auth_data = {
                "username": self.username,
                "password": self.password,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "password"
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.portal_base_url}/auth/token",
                    data=auth_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.status_code == 200:
                    auth_response = response.json()
                    
                    # Store session details
                    expires_in = auth_response.get("expires_in", 7200)  # Default 2 hours
                    self.session_expires_at = datetime.now() + timedelta(seconds=expires_in)
                    
                    self.current_session = {
                        "token": auth_response["access_token"],
                        "refresh_token": auth_response.get("refresh_token"),
                        "session_id": auth_response.get("session_id"),
                        "token_type": auth_response.get("token_type", "Bearer")
                    }
                    
                    logger.info(f"Successfully authenticated with PM-JAY portal. Session expires at: {self.session_expires_at}")
                    
                    return AuthenticationResult(
                        success=True,
                        session_token=self.current_session["token"],
                        expires_at=self.session_expires_at,
                        error_message=None,
                        portal_session_id=self.current_session["session_id"]
                    )
                else:
                    error_msg = f"Authentication failed: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    
                    return AuthenticationResult(
                        success=False,
                        session_token=None,
                        expires_at=None,
                        error_message=error_msg,
                        portal_session_id=None
                    )
                    
        except Exception as e:
            error_msg = f"Authentication error: {str(e)}"
            logger.error(error_msg)
            
            return AuthenticationResult(
                success=False,
                session_token=None,
                expires_at=None,
                error_message=error_msg,
                portal_session_id=None
            )
    
    async def refresh_session(self) -> AuthenticationResult:
        """
        Refresh current session using refresh token.
        
        Returns:
            AuthenticationResult with refreshed session details
        """
        try:
            if not self.current_session or not self.current_session.get("refresh_token"):
                logger.warning("No refresh token available, performing new authentication")
                return await self.authenticate()
            
            refresh_data = {
                "refresh_token": self.current_session["refresh_token"],
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "refresh_token"
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.portal_base_url}/auth/refresh",
                    data=refresh_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.status_code == 200:
                    auth_response = response.json()
                    
                    # Update session details
                    expires_in = auth_response.get("expires_in", 7200)
                    self.session_expires_at = datetime.now() + timedelta(seconds=expires_in)
                    
                    self.current_session.update({
                        "token": auth_response["access_token"],
                        "refresh_token": auth_response.get("refresh_token", self.current_session["refresh_token"])
                    })
                    
                    logger.info("Successfully refreshed session")
                    
                    return AuthenticationResult(
                        success=True,
                        session_token=self.current_session["token"],
                        expires_at=self.session_expires_at,
                        error_message=None,
                        portal_session_id=self.current_session["session_id"]
                    )
                else:
                    logger.warning("Session refresh failed, performing new authentication")
                    return await self.authenticate()
                    
        except Exception as e:
            logger.error(f"Session refresh error: {str(e)}")
            return await self.authenticate()
    
    def _is_session_valid(self) -> bool:
        """Check if current session is valid and not expired."""
        if not self.current_session or not self.session_expires_at:
            return False
        
        # Add 5-minute buffer before expiration
        buffer_time = timedelta(minutes=5)
        return datetime.now() < (self.session_expires_at - buffer_time)
    
    async def ensure_authenticated(self) -> AuthenticationResult:
        """
        Ensure we have a valid authenticated session.
        
        Returns:
            AuthenticationResult with current session status
        """
        if self._is_session_valid():
            return AuthenticationResult(
                success=True,
                session_token=self.current_session["token"],
                expires_at=self.session_expires_at,
                error_message=None,
                portal_session_id=self.current_session["session_id"]
            )
        
        # Try to refresh first, then authenticate if refresh fails
        if self.current_session and self.current_session.get("refresh_token"):
            result = await self.refresh_session()
            if result.success:
                return result
        
        return await self.authenticate()
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        if not self.current_session:
            raise ValueError("No active session. Please authenticate first.")
        
        return {
            "Authorization": f"{self.current_session['token_type']} {self.current_session['token']}",
            "Content-Type": "application/json"
        }
    
    async def logout(self) -> bool:
        """
        Logout and invalidate current session.
        
        Returns:
            True if logout successful, False otherwise
        """
        try:
            if not self.current_session:
                return True
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.portal_base_url}/auth/logout",
                    headers=self.get_auth_headers()
                )
                
                # Clear session regardless of response
                self.current_session = None
                self.session_expires_at = None
                
                logger.info("Session logged out and cleared")
                return response.status_code == 200
                
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            # Clear session even if logout request failed
            self.current_session = None
            self.session_expires_at = None
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Check authentication service health."""
        return {
            "status": "healthy",
            "has_credentials": all([self.username, self.password, self.client_id, self.client_secret]),
            "has_active_session": self._is_session_valid(),
            "session_expires_at": self.session_expires_at.isoformat() if self.session_expires_at else None,
            "portal_base_url": self.portal_base_url
        }


class PortalFormService:
    """Handles form population and submission to PM-JAY portal."""
    
    def __init__(self, auth_service: PortalAuthenticationService):
        """Initialize form service with authentication service."""
        self.auth_service = auth_service
        self.portal_base_url = auth_service.portal_base_url
        self.timeout = httpx.Timeout(60.0)  # Longer timeout for form submissions
        
    async def populate_claim_form(self, medical_record: 'MedicalRecord', package_recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Populate PM-JAY claim form with medical record data.
        
        Args:
            medical_record: Structured medical record data
            package_recommendation: Recommended package details
            
        Returns:
            Populated form data dictionary
        """
        try:
            # Ensure authentication
            auth_result = await self.auth_service.ensure_authenticated()
            if not auth_result.success:
                raise ValueError(f"Authentication failed: {auth_result.error_message}")
            
            # Map medical record to PM-JAY form fields
            form_data = {
                # Patient Information
                "patient_id": medical_record.patient_info.patient_id,
                "patient_name": medical_record.patient_info.name,
                "patient_age": medical_record.patient_info.age,
                "patient_gender": medical_record.patient_info.gender,
                
                # Hospital Information
                "hospital_id": medical_record.hospital_id,
                "admission_date": medical_record.admission_date.strftime("%Y-%m-%d"),
                "discharge_date": medical_record.discharge_date.strftime("%Y-%m-%d"),
                
                # Medical Information
                "procedures": [
                    {
                        "procedure_code": proc.procedure_code,
                        "procedure_name": proc.procedure_name,
                        "procedure_date": proc.procedure_date.strftime("%Y-%m-%d") if proc.procedure_date else None
                    }
                    for proc in medical_record.procedures
                ],
                "diagnoses": [
                    {
                        "diagnosis_code": diag.diagnosis_code,
                        "diagnosis_name": diag.diagnosis_name,
                        "diagnosis_date": diag.diagnosis_date.strftime("%Y-%m-%d") if diag.diagnosis_date else None
                    }
                    for diag in medical_record.diagnoses
                ],
                
                # Financial Information
                "total_amount": str(medical_record.total_amount),
                "document_confidence": medical_record.document_confidence,
                
                # Package Recommendation
                "recommended_package_code": package_recommendation.get("package_code"),
                "recommended_package_name": package_recommendation.get("package_name"),
                "package_confidence": package_recommendation.get("confidence", 0.0),
                
                # Metadata
                "submission_timestamp": datetime.now().isoformat(),
                "system_version": "sanjeevni_plus_1.0"
            }
            
            logger.info(f"Populated form data for patient: {medical_record.patient_info.patient_id}")
            return form_data
            
        except Exception as e:
            logger.error(f"Form population failed: {str(e)}")
            raise
    
    async def submit_claim_form(self, form_data: Dict[str, Any], claim_id: str) -> SubmissionResult:
        """
        Submit populated claim form to PM-JAY portal.
        
        Args:
            form_data: Populated form data
            claim_id: Internal claim identifier
            
        Returns:
            SubmissionResult with submission details
        """
        try:
            # Ensure authentication
            auth_result = await self.auth_service.ensure_authenticated()
            if not auth_result.success:
                return SubmissionResult(
                    success=False,
                    submission_id=None,
                    portal_reference=None,
                    error_message=f"Authentication failed: {auth_result.error_message}",
                    retry_after=None,
                    status=PortalStatus.AUTHENTICATION_FAILED
                )
            
            # Add claim ID to form data
            submission_data = {
                **form_data,
                "internal_claim_id": claim_id,
                "submission_method": "automated"
            }

            # If WatsonX Orchestrate is configured, use it to submit the claim
            wxo_skill_submit = os.getenv("WXO_SKILL_SUBMIT_CLAIM")
            if wxo_client.api_key and wxo_skill_submit:
                try:
                    orchestrate_inputs = {
                        "claim_id": claim_id,
                        "form_data": submission_data
                    }
                    run: OrchestrateRun = await wxo_client.run_and_wait(wxo_skill_submit, orchestrate_inputs, timeout_seconds=180)
                    if run.success and run.status == "completed" and run.output:
                        out = run.output
                        return SubmissionResult(
                            success=bool(out.get("success", True)),
                            submission_id=out.get("submission_id"),
                            portal_reference=out.get("portal_reference"),
                            error_message=out.get("error"),
                            retry_after=out.get("retry_after"),
                            status=PortalStatus.FORM_SUBMITTED if out.get("success", True) else PortalStatus.SUBMISSION_FAILED
                        )
                    else:
                        logger.warning(f"WXO submit run failed or incomplete: {run.status} - {run.error}")
                        # Fallback to direct API call below
                except Exception as ex:
                    logger.error(f"WXO submit claim error, falling back to direct API: {ex}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.portal_base_url}/claims/submit",
                    json=submission_data,
                    headers=self.auth_service.get_auth_headers()
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    logger.info(f"Successfully submitted claim {claim_id}. Portal reference: {response_data.get('portal_reference')}")
                    
                    return SubmissionResult(
                        success=True,
                        submission_id=response_data.get("submission_id"),
                        portal_reference=response_data.get("portal_reference"),
                        error_message=None,
                        retry_after=None,
                        status=PortalStatus.FORM_SUBMITTED
                    )
                
                elif response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 300))
                    error_msg = f"Rate limited. Retry after {retry_after} seconds"
                    
                    logger.warning(error_msg)
                    
                    return SubmissionResult(
                        success=False,
                        submission_id=None,
                        portal_reference=None,
                        error_message=error_msg,
                        retry_after=retry_after,
                        status=PortalStatus.SUBMISSION_FAILED
                    )
                
                else:
                    error_msg = f"Submission failed: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    
                    return SubmissionResult(
                        success=False,
                        submission_id=None,
                        portal_reference=None,
                        error_message=error_msg,
                        retry_after=None,
                        status=PortalStatus.SUBMISSION_FAILED
                    )
                    
        except httpx.TimeoutException:
            error_msg = "Submission timeout - portal may be slow"
            logger.error(error_msg)
            
            return SubmissionResult(
                success=False,
                submission_id=None,
                portal_reference=None,
                error_message=error_msg,
                retry_after=60,  # Retry after 1 minute
                status=PortalStatus.CONNECTION_ERROR
            )
            
        except Exception as e:
            error_msg = f"Submission error: {str(e)}"
            logger.error(error_msg)
            
            return SubmissionResult(
                success=False,
                submission_id=None,
                portal_reference=None,
                error_message=error_msg,
                retry_after=None,
                status=PortalStatus.SUBMISSION_FAILED
            )
    
    async def validate_form_data(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate form data before submission.
        
        Args:
            form_data: Form data to validate
            
        Returns:
            Validation result with errors if any
        """
        validation_errors = []
        
        # Required fields validation
        required_fields = [
            "patient_id", "patient_name", "patient_age", "patient_gender",
            "hospital_id", "admission_date", "discharge_date", "total_amount"
        ]
        
        for field in required_fields:
            if not form_data.get(field):
                validation_errors.append(f"Missing required field: {field}")
        
        # Data format validation
        try:
            if form_data.get("patient_age"):
                age = int(form_data["patient_age"])
                if age < 0 or age > 150:
                    validation_errors.append("Invalid patient age")
        except (ValueError, TypeError):
            validation_errors.append("Patient age must be a valid number")
        
        try:
            if form_data.get("total_amount"):
                amount = float(form_data["total_amount"])
                if amount < 0:
                    validation_errors.append("Total amount cannot be negative")
        except (ValueError, TypeError):
            validation_errors.append("Total amount must be a valid number")
        
        # Date validation
        date_fields = ["admission_date", "discharge_date"]
        for field in date_fields:
            if form_data.get(field):
                try:
                    datetime.strptime(form_data[field], "%Y-%m-%d")
                except ValueError:
                    validation_errors.append(f"Invalid date format for {field}")
        
        return {
            "is_valid": len(validation_errors) == 0,
            "errors": validation_errors,
            "validated_at": datetime.now().isoformat()
        }


class PortalStatusTracker:
    """Handles status tracking for submitted claims."""
    
    def __init__(self, auth_service: PortalAuthenticationService):
        """Initialize status tracker with authentication service."""
        self.auth_service = auth_service
        self.portal_base_url = auth_service.portal_base_url
        self.timeout = httpx.Timeout(30.0)
        
    async def track_claim_status(self, portal_reference: str) -> StatusTrackingResult:
        """
        Track status of submitted claim.
        
        Args:
            portal_reference: Portal reference number
            
        Returns:
            StatusTrackingResult with current status
        """
        try:
            # Ensure authentication
            auth_result = await self.auth_service.ensure_authenticated()
            if not auth_result.success:
                raise ValueError(f"Authentication failed: {auth_result.error_message}")
            
            # If WatsonX Orchestrate is configured, use it to check status
            wxo_skill_status = os.getenv("WXO_SKILL_CHECK_STATUS")
            if wxo_client.api_key and wxo_skill_status:
                try:
                    inputs = {"portal_reference": portal_reference}
                    run: OrchestrateRun = await wxo_client.run_and_wait(wxo_skill_status, inputs, timeout_seconds=120)
                    if run.success and run.status == "completed" and run.output:
                        out = run.output
                        return StatusTrackingResult(
                            claim_status=out.get("status", "unknown"),
                            last_updated=datetime.fromisoformat(out.get("last_updated", datetime.now().isoformat())),
                            portal_messages=out.get("messages", []),
                            approval_status=out.get("approval_status"),
                            rejection_reason=out.get("rejection_reason")
                        )
                    else:
                        logger.warning(f"WXO status run failed or incomplete: {run.status} - {run.error}")
                        # Fallback to direct API call below
                except Exception as ex:
                    logger.error(f"WXO check status error, falling back to direct API: {ex}")

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.portal_base_url}/claims/{portal_reference}/status",
                    headers=self.auth_service.get_auth_headers()
                )
                
                if response.status_code == 200:
                    status_data = response.json()
                    
                    return StatusTrackingResult(
                        claim_status=status_data.get("status", "unknown"),
                        last_updated=datetime.fromisoformat(status_data.get("last_updated", datetime.now().isoformat())),
                        portal_messages=status_data.get("messages", []),
                        approval_status=status_data.get("approval_status"),
                        rejection_reason=status_data.get("rejection_reason")
                    )
                else:
                    raise ValueError(f"Status tracking failed: {response.status_code} - {response.text}")
                    
        except Exception as e:
            logger.error(f"Status tracking error for {portal_reference}: {str(e)}")
            raise
    
    async def get_bulk_status(self, portal_references: List[str]) -> Dict[str, StatusTrackingResult]:
        """
        Get status for multiple claims in bulk.
        
        Args:
            portal_references: List of portal reference numbers
            
        Returns:
            Dictionary mapping portal references to status results
        """
        results = {}
        
        # Process in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(portal_references), batch_size):
            batch = portal_references[i:i + batch_size]
            
            # Create concurrent tasks for batch
            tasks = [self.track_claim_status(ref) for ref in batch]
            
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for ref, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to get status for {ref}: {str(result)}")
                    else:
                        results[ref] = result
                        
            except Exception as e:
                logger.error(f"Batch status tracking failed: {str(e)}")
        
        return results


# Global service instances
portal_auth_service = PortalAuthenticationService()
portal_form_service = PortalFormService(portal_auth_service)
portal_status_tracker = PortalStatusTracker(portal_auth_service)
