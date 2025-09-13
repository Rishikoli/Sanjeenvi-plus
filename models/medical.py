from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    """Represents the patient's demographic and identification information."""
    patient_id: str = Field(..., description="Unique identifier for the patient.")
    name: str = Field(..., description="Full name of the patient.")
    age: int = Field(..., gt=0, description="Age of the patient in years.")
    gender: str = Field(..., description="Gender of the patient.")


class MedicalProcedure(BaseModel):
    """Represents a single medical procedure performed."""
    procedure_code: str = Field(..., description="Procedure code (e.g., ICD-10-PCS).")
    procedure_name: str = Field(..., description="Name/description of the medical procedure.")
    procedure_date: Optional[datetime] = Field(None, description="Date of the procedure if available.")


class Diagnosis(BaseModel):
    """Represents a single diagnosis."""
    diagnosis_code: str = Field(..., description="Diagnosis code (e.g., ICD-10).")
    diagnosis_name: str = Field(..., description="Name/description of the diagnosis.")
    diagnosis_date: Optional[datetime] = Field(None, description="Date of diagnosis if available.")


class MedicalRecord(BaseModel):
    """Represents a comprehensive, structured medical record extracted from a document."""
    patient_info: PatientInfo
    hospital_id: str = Field(..., description="Identifier for the hospital.")
    admission_date: datetime
    discharge_date: datetime
    procedures: List[MedicalProcedure]
    diagnoses: List[Diagnosis]
    total_amount: Decimal = Field(..., ge=0, description="Total amount on the claim.")
    document_confidence: float = Field(
        ..., ge=0, le=1, description="OCR confidence score for the source document."
    )
