"""
Claims management API endpoints.

Handles claim creation, retrieval, updates, and management operations.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status, Path, Query, Body
from pydantic import BaseModel, Field

from database.repository import ClaimsRepository, ProcessingStepsRepository
from models.claims import SubmissionStatus, ProcessingStatus, Claim as ClaimModel

# Configure logging
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/claims", tags=["claims"])

# --- Request/Response Models ---

class ClaimBase(BaseModel):
    hospital_id: str = Field(..., description="Hospital identifier")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    priority: str = Field("normal", description="Processing priority (low, normal, high)")

class ClaimCreate(ClaimBase):
    pass

class ClaimUpdate(BaseModel):
    status: Optional[SubmissionStatus] = None
    priority: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ClaimResponse(ClaimBase):
    claim_id: str
    status: SubmissionStatus
    created_at: datetime
    updated_at: datetime
    documents: List[Dict[str, Any]] = []

class ClaimListResponse(BaseModel):
    claims: List[ClaimResponse]
    total: int
    limit: int
    offset: int

# --- Helper Functions ---

def _map_claim_to_response(claim: ClaimModel) -> Dict[str, Any]:
    return {
        "claim_id": str(claim.claim_id),
        "hospital_id": claim.hospital_id,
        "patient_id": claim.patient_id,
        "status": claim.status,
        "priority": claim.priority,
        "created_at": claim.created_at,
        "updated_at": claim.updated_at,
        "documents": [doc.dict() for doc in getattr(claim, 'documents', [])]
    }

# --- API Endpoints ---

@router.post("", response_model=ClaimResponse, status_code=status.HTTP_201_CREATED)
async def create_claim(claim_data: ClaimCreate, claims_repo: ClaimsRepository = Depends()):
    """Create a new claim."""
    try:
        claim = ClaimModel(
            claim_id=f"clm_{uuid4().hex}",
            hospital_id=claim_data.hospital_id,
            patient_id=claim_data.patient_id,
            status=SubmissionStatus.DRAFT,
            priority=claim_data.priority,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        created_claim = claims_repo.create_claim(claim)
        return _map_claim_to_response(created_claim)
    except Exception as e:
        logger.exception(f"Error creating claim: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create claim: {str(e)}")

@router.get("/{claim_id}", response_model=ClaimResponse)
async def get_claim(claim_id: str = Path(...), claims_repo: ClaimsRepository = Depends()):
    """Get a claim by ID."""
    claim = claims_repo.get_claim(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")
    return _map_claim_to_response(claim)

@router.patch("/{claim_id}", response_model=ClaimResponse)
async def update_claim(
    claim_id: str = Path(...),
    update_data: ClaimUpdate = Body(...),
    claims_repo: ClaimsRepository = Depends()
):
    """Update a claim."""
    claim = claims_repo.get_claim(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")
    
    update_dict = update_data.dict(exclude_unset=True)
    updated_claim = claims_repo.update_claim(claim_id, update_dict)
    
    if not updated_claim:
        raise HTTPException(status_code=500, detail="Failed to update claim")
    
    return _map_claim_to_response(updated_claim)

@router.get("", response_model=ClaimListResponse)
async def list_claims(
    hospital_id: Optional[str] = None,
    status: Optional[SubmissionStatus] = None,
    patient_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    claims_repo: ClaimsRepository = Depends()
):
    """List claims with optional filtering."""
    try:
        filters = {}
        if hospital_id:
            filters["hospital_id"] = hospital_id
        if status:
            filters["status"] = status
        if patient_id:
            filters["patient_id"] = patient_id
            
        claims, total = claims_repo.list_claims(
            filters=filters,
            limit=limit,
            offset=offset
        )
        
        return ClaimListResponse(
            claims=[_map_claim_to_response(claim) for claim in claims],
            total=total,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.exception(f"Error listing claims: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list claims: {str(e)}")
