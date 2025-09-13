"""
Portal Automation API Endpoints.

Handles interactions with the PM-JAY government portal including form submission
and status tracking.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, HttpUrl

from services.portal_automation_service import (
    portal_auth_service,
    portal_form_service,
    portal_status_tracker,
    PortalStatus,
    AuthenticationResult,
    SubmissionResult,
    StatusTrackingResult
)
from models.claims import Claim, ClaimStatus
from database.repository import ClaimsRepository

# Configure logging
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/portal", tags=["portal"])

# --- Request/Response Models ---

class PortalAuthRequest(BaseModel):
    """Request model for portal authentication."""
    username: str = Field(..., description="PM-JAY portal username")
    password: str = Field(..., description="PM-JAY portal password")
    client_id: str = Field(..., description="OAuth client ID")
    client_secret: str = Field(..., description="OAuth client secret")


class PortalAuthResponse(BaseModel):
    """Response model for authentication status."""
    authenticated: bool
    expires_at: Optional[datetime] = None
    portal_session_id: Optional[str] = None
    error: Optional[str] = None


class PortalFormData(BaseModel):
    """Model for form data submission to portal."""
    claim_id: str
    form_data: Dict[str, Any]
    force_submit: bool = False


class PortalSubmissionResponse(BaseModel):
    """Response model for form submission."""
    success: bool
    submission_id: Optional[str] = None
    portal_reference: Optional[str] = None
    status: str
    error: Optional[str] = None
    retry_after: Optional[int] = None


class StatusCheckResponse(BaseModel):
    """Response model for claim status check."""
    claim_id: str
    portal_reference: str
    status: str
    last_updated: datetime
    portal_messages: List[str] = []
    approval_status: Optional[str] = None
    rejection_reason: Optional[str] = None


# --- Helper Functions ---

def map_portal_status(portal_status: str) -> ClaimStatus:
    """Map portal status to internal claim status."""
    status_map = {
        "SUBMITTED": ClaimStatus.SUBMITTED,
        "IN_REVIEW": ClaimStatus.UNDER_REVIEW,
        "APPROVED": ClaimStatus.APPROVED,
        "REJECTED": ClaimStatus.REJECTED,
        "PAID": ClaimStatus.PAID,
        "PENDING_DOCUMENTS": ClaimStatus.PENDING_DOCUMENTS,
    }
    return status_map.get(portal_status.upper(), ClaimStatus.SUBMISSION_ERROR)


# --- API Endpoints ---

@router.post("/auth", response_model=PortalAuthResponse)
async def authenticate_portal(credentials: PortalAuthRequest):
    """
    Authenticate with the PM-JAY portal.
    
    This endpoint establishes a session with the PM-JAY portal using the provided
    credentials. The session will be used for subsequent API calls.
    """
    try:
        # Update service credentials
        portal_auth_service.username = credentials.username
        portal_auth_service.password = credentials.password
        portal_auth_service.client_id = credentials.client_id
        portal_auth_service.client_secret = credentials.client_secret
        
        # Authenticate
        result = await portal_auth_service.authenticate()
        
        return PortalAuthResponse(
            authenticated=result.success,
            expires_at=result.expires_at,
            portal_session_id=result.portal_session_id,
            error=result.error_message
        )
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}"
        )


@router.post("/submit-claim", response_model=PortalSubmissionResponse)
async def submit_claim_to_portal(
    form_data: PortalFormData,
    claims_repo: ClaimsRepository = Depends()
):
    """
    Submit a claim to the PM-JAY portal.
    
    This endpoint takes the populated form data and submits it to the PM-JAY portal.
    It also updates the claim status in our system based on the submission result.
    """
    try:
        # Get the claim from database
        claim = claims_repo.get_claim_by_id(form_data.claim_id)
        if not claim:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Claim {form_data.claim_id} not found"
            )
        
        # Submit to portal
        result = await portal_form_service.submit_claim_form(
            form_data=form_data.form_data,
            claim_id=form_data.claim_id
        )
        
        # Update claim status based on submission result
        if result.success:
            claim.portal_reference = result.portal_reference
            claim.status = ClaimStatus.SUBMITTED
            claim.submission_date = datetime.utcnow()
        else:
            claim.status = ClaimStatus.SUBMISSION_ERROR
            claim.error_message = result.error_message
        
        # Save updated claim
        claims_repo.update_claim(claim.id, claim.dict())
        
        return PortalSubmissionResponse(
            success=result.success,
            submission_id=result.submission_id,
            portal_reference=result.portal_reference,
            status=result.status.value,
            error=result.error_message,
            retry_after=result.retry_after
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Claim submission failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit claim: {str(e)}"
        )


@router.get("/status/{claim_id}", response_model=StatusCheckResponse)
async def check_claim_status(
    claim_id: str,
    claims_repo: ClaimsRepository = Depends()
):
    """
    Check the status of a submitted claim.
    
    This endpoint queries the PM-JAY portal for the latest status of a claim
    and updates our system accordingly.
    """
    try:
        # Get the claim from database
        claim = claims_repo.get_claim_by_id(claim_id)
        if not claim:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Claim {claim_id} not found"
            )
        
        if not claim.portal_reference:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Claim {claim_id} has not been submitted to the portal"
            )
        
        # Check status with portal
        status_result = await portal_status_tracker.track_claim_status(
            portal_reference=claim.portal_reference
        )
        
        # Update claim status based on portal status
        claim.status = map_portal_status(status_result.claim_status)
        claim.last_status_check = datetime.utcnow()
        
        # Update additional fields if available
        if hasattr(claim, 'approval_status'):
            claim.approval_status = status_result.approval_status
        if hasattr(claim, 'rejection_reason') and status_result.rejection_reason:
            claim.rejection_reason = status_result.rejection_reason
        
        # Save updated claim
        claims_repo.update_claim(claim.id, claim.dict())
        
        return StatusCheckResponse(
            claim_id=claim.id,
            portal_reference=claim.portal_reference,
            status=status_result.claim_status,
            last_updated=status_result.last_updated,
            portal_messages=status_result.portal_messages,
            approval_status=status_result.approval_status,
            rejection_reason=status_result.rejection_reason
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check claim status: {str(e)}"
        )


@router.get("/bulk-status", response_model=Dict[str, StatusCheckResponse])
async def check_bulk_status(
    portal_references: List[str] = Query(..., description="List of portal reference numbers")
):
    """
    Check status for multiple claims in a single request.
    
    This endpoint is more efficient than checking status one by one when
    dealing with multiple claims.
    """
    try:
        if not portal_references:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one portal reference is required"
            )
        
        # Get status for all references
        results = await portal_status_tracker.get_bulk_status(portal_references)
        
        # Convert to response model
        return {
            ref: StatusCheckResponse(
                claim_id="",  # Will be filled by the client
                portal_reference=result.portal_reference,
                status=result.claim_status,
                last_updated=result.last_updated,
                portal_messages=result.portal_messages,
                approval_status=result.approval_status,
                rejection_reason=result.rejection_reason
            )
            for ref, result in results.items()
        }
        
    except Exception as e:
        logger.error(f"Bulk status check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check bulk status: {str(e)}"
        )
