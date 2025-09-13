from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from .medical import MedicalRecord


class ComplianceStatus(str, Enum):
    """Enumeration for compliance status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_REVIEW = "requires_review"


class SubmissionStatus(str, Enum):
    """Enumeration for claim submission status."""
    DRAFT = "draft"
    PENDING = "pending"
    PROCESSING = "processing"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"
    UNDER_REVIEW = "under_review"


class PackageRecommendation(BaseModel):
    """Represents a recommended PM-JAY package for a claim."""
    package_code: str = Field(..., description="PM-JAY package code.")
    package_name: str = Field(..., description="Name of the PM-JAY package.")
    confidence_score: float = Field(
        ..., ge=0, le=1, description="Confidence score for this recommendation."
    )
    estimated_amount: Decimal = Field(..., gt=0, description="Estimated reimbursement amount.")
    approval_probability: float = Field(
        ..., ge=0, le=1, description="Probability of approval for this package."
    )
    risk_factors: List[str] = Field(
        default_factory=list, description="List of identified risk factors."
    )
    compliance_status: ComplianceStatus = Field(
        ..., description="Compliance status with PM-JAY guidelines."
    )
    chroma_similarity_scores: List[float] = Field(
        default_factory=list, description="Vector similarity scores from ChromaDB search."
    )


class RiskAssessment(BaseModel):
    """Represents a risk assessment for a claim."""
    overall_risk_score: float = Field(..., ge=0, le=1, description="Overall risk score.")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, or HIGH.")
    risk_factors: List[str] = Field(
        default_factory=list, description="Identified risk factors."
    )
    fraud_indicators: List[str] = Field(
        default_factory=list, description="Potential fraud indicators."
    )
    recommendation: str = Field(..., description="Processing recommendation: AUTO_APPROVE or MANUAL_REVIEW.")


class ClaimSubmission(BaseModel):
    """Represents a complete claim submission."""
    claim_id: str = Field(..., description="Unique identifier for the claim.")
    hospital_id: str = Field(..., description="Identifier for the submitting hospital.")
    patient_data: MedicalRecord = Field(..., description="Medical record data.")
    recommended_package: Optional[PackageRecommendation] = Field(
        None, description="Recommended PM-JAY package."
    )
    risk_assessment: Optional[RiskAssessment] = Field(
        None, description="Risk assessment for the claim."
    )
    submission_status: SubmissionStatus = Field(
        default=SubmissionStatus.PENDING, description="Current submission status."
    )
    portal_reference: Optional[str] = Field(
        None, description="Reference number from government portal."
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    retry_count: int = Field(default=0, ge=0, description="Number of submission retry attempts.")
