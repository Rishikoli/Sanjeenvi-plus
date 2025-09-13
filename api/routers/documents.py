"""
Document processing API endpoints.

Handles document uploads, processing, and management for healthcare claims.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl

from services.document_processing_service import document_processing_service, DocumentProcessingResult
from services.claims_intelligence_service import ClaimsIntelligenceService
from database.repository import ClaimsRepository, ProcessingStepsRepository
from models.claims import SubmissionStatus, ProcessingStep, ProcessingStatus
from models.medical import MedicalRecord

# Configure logging
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

# --- Request/Response Models ---

class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    hospital_id: str = Field(..., description="Unique identifier for the hospital")
    claim_id: Optional[str] = Field(None, description="Optional existing claim ID")
    document_type: str = Field(..., description="Type of document (prescription, lab_report, discharge_summary, etc.)")
    priority: str = Field("normal", description="Processing priority (low, normal, high)")

class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str = Field(..., description="Unique identifier for the uploaded document")
    claim_id: str = Field(..., description="Claim ID (new or existing)")
    processing_status: str = Field(..., description="Current processing status")
    document_url: Optional[HttpUrl] = Field(None, description="URL to access the document")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated time for processing completion")

class DocumentStatusResponse(BaseModel):
    """Response model for document status check."""
    document_id: str
    claim_id: str
    status: str
    processing_steps: List[Dict[str, Any]]
    extracted_data: Optional[Dict[str, Any]] = None
    medical_record: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int

# --- Helper Functions ---

def _save_uploaded_file(upload_file: UploadFile, upload_dir: Path) -> Path:
    """Save uploaded file to disk and return the path."""
    try:
        # Create upload directory if it doesn't exist
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_ext = Path(upload_file.filename).suffix.lower()
        filename = f"{uuid4().hex}{file_ext}"
        file_path = upload_dir / filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            buffer.write(upload_file.file.read())
            
        return file_path
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

# --- API Endpoints ---

@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(..., description="Medical document to process"),
    request: DocumentUploadRequest = Depends(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    claims_service: ClaimsIntelligenceService = Depends()
):
    """
    Upload a medical document for processing.
    
    This endpoint accepts medical documents in various formats (PDF, images) and starts
    an asynchronous processing pipeline to extract relevant information.
    
    Returns immediately with a processing ID that can be used to check status.
    """
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_ext}. Supported types: .pdf, .jpg, .jpeg, .png, .tiff"
            )
        
        # Generate document and claim IDs if not provided
        document_id = f"doc_{uuid4().hex}"
        claim_id = request.claim_id or f"clm_{uuid4().hex}"
        
        # Save uploaded file to temporary location
        upload_dir = Path("data/uploads") / claim_id
        file_path = _save_uploaded_file(file, upload_dir)
        
        # Log the upload
        logger.info(f"Received document upload: {file.filename} for claim {claim_id}")
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            file_path=file_path,
            document_id=document_id,
            claim_id=claim_id,
            hospital_id=request.hospital_id,
            document_type=request.document_type,
            priority=request.priority,
            claims_service=claims_service
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            claim_id=claim_id,
            processing_status="queued",
            document_url=f"/api/documents/{document_id}/content"  # TODO: Implement actual URL
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing document upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )

@router.get("/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: str,
    claims_repo: ClaimsRepository = Depends()
):
    """
    Get the status of a processed document.
    
    Returns detailed status information including processing steps, extracted data,
    and any errors that occurred during processing.
    """
    try:
        # Get document and processing steps from database
        document = claims_repo.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
            
        processing_steps = claims_repo.get_processing_steps(document_id)
        
        # Format response
        response = DocumentStatusResponse(
            document_id=document_id,
            claim_id=document["claim_id"],
            status=document["status"].value,
            processing_steps=[step.dict() for step in processing_steps],
            created_at=document["created_at"],
            updated_at=document["updated_at"]
        )
        
        # Add extracted data if available
        if "extracted_data" in document and document["extracted_data"]:
            response.extracted_data = document["extracted_data"]
            
        # Add medical record if available
        if "medical_record" in document and document["medical_record"]:
            response.medical_record = document["medical_record"]
            
        # Add error if present
        if document["error"]:
            response.error = document["error"]
            
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving document status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document status: {str(e)}"
        )

@router.get("/{document_id}/content")
async def get_document_content(
    document_id: str,
    claims_repo: ClaimsRepository = Depends()
):
    """
    Download the original document content.
    
    Returns the original uploaded file with appropriate content type.
    """
    try:
        # Get document metadata
        document = claims_repo.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
            
        # Check if file exists
        file_path = Path(document["file_path"])
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document file not found"
            )
            
        # Determine content type
        content_type = "application/octet-stream"
        if file_path.suffix.lower() == ".pdf":
            content_type = "application/pdf"
        elif file_path.suffix.lower() in [".jpg", ".jpeg"]:
            content_type = "image/jpeg"
        elif file_path.suffix.lower() == ".png":
            content_type = "image/png"
        elif file_path.suffix.lower() in [".tiff", ".tif"]:
            content_type = "image/tiff"
            
        # Stream file content
        from fastapi.responses import FileResponse
        return FileResponse(
            file_path,
            media_type=content_type,
            filename=file_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving document content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document content: {str(e)}"
        )

@router.get("/claim/{claim_id}", response_model=DocumentListResponse)
async def list_claim_documents(
    claim_id: str,
    limit: int = 10,
    offset: int = 0,
    claims_repo: ClaimsRepository = Depends()
):
    """
    List all documents associated with a claim.
    
    Returns paginated list of documents with their status and metadata.
    """
    try:
        documents, total = claims_repo.list_documents_by_claim(
            claim_id=claim_id,
            limit=limit,
            offset=offset
        )
        
        return DocumentListResponse(
            documents=[doc.dict() for doc in documents],
            total=total,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        logger.exception(f"Error listing claim documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list claim documents: {str(e)}"
        )

# --- Background Tasks ---

async def process_document_background(
    file_path: Path,
    document_id: str,
    claim_id: str,
    hospital_id: str,
    document_type: str,
    priority: str,
    claims_service: ClaimsIntelligenceService
):
    """
    Background task to process an uploaded document.
    
    This runs asynchronously after the initial upload response is returned.
    """
    from datetime import datetime
    
    # Initialize repositories
    claims_repo = ClaimsRepository()
    steps_repo = ProcessingStepsRepository()
    
    try:
        # Create initial document record
        document = {
            "document_id": document_id,
            "claim_id": claim_id,
            "hospital_id": hospital_id,
            "document_type": document_type,
            "file_path": str(file_path),
            "status": ProcessingStatus.PROCESSING,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "error": None
        }
        claims_repo.create_document(document)
        
        # Log processing start
        steps_repo.add_step(
            ProcessingStep(
                document_id=document_id,
                step_name="document_upload",
                status=ProcessingStatus.COMPLETED,
                details={"message": "Document uploaded successfully"},
                timestamp=datetime.utcnow()
            )
        )
        
        # Process the document
        steps_repo.add_step(
            ProcessingStep(
                document_id=document_id,
                step_name="document_processing",
                status=ProcessingStatus.PROCESSING,
                details={"message": "Starting document processing"},
                timestamp=datetime.utcnow()
            )
        )
        
        # Call document processing service
        result = document_processing_service.process_document(str(file_path))
        
        # Update document with extracted data
        extracted_data = {
            "text": result.ocr_result.text,
            "language": result.ocr_result.language,
            "confidence": result.ocr_result.confidence,
            "page_count": result.ocr_result.page_count,
            "processing_time_ms": result.ocr_result.processing_time_ms
        }
        
        # Create or update claim with extracted data
        claim = claims_repo.get_claim(claim_id) or {}
        claim.update({
            "claim_id": claim_id,
            "hospital_id": hospital_id,
            "status": SubmissionStatus.PROCESSING,
            "priority": priority,
            "extracted_data": {**claim.get("extracted_data", {}), **extracted_data},
            "updated_at": datetime.utcnow()
        })
        
        if not claim.get("created_at"):
            claim["created_at"] = datetime.utcnow()
            claims_repo.create_claim(claim)
        else:
            claims_repo.update_claim(claim_id, claim)
        
        # Save medical record if available
        if result.medical_record:
            claims_repo.update_document(
                document_id,
                {
                    "medical_record": result.medical_record.dict(),
                    "status": ProcessingStatus.COMPLETED,
                    "updated_at": datetime.utcnow()
                }
            )
            
            # Update claim with medical record reference
            claims_repo.update_claim(
                claim_id,
                {
                    "medical_record_id": document_id,
                    "updated_at": datetime.utcnow()
                }
            )
        
        # Log completion
        steps_repo.update_step(
            document_id=document_id,
            step_name="document_processing",
            status=ProcessingStatus.COMPLETED,
            details={
                "message": "Document processing completed",
                "processing_time_ms": result.ocr_result.processing_time_ms,
                "page_count": result.ocr_result.page_count,
                "confidence": result.ocr_result.confidence
            },
            timestamp=datetime.utcnow()
        )
        
        # Trigger claims intelligence pipeline
        await claims_service.process_claim(claim_id)
        
    except Exception as e:
        logger.exception(f"Error in background document processing: {str(e)}")
        
        # Update document with error
        claims_repo.update_document(
            document_id,
            {
                "status": ProcessingStatus.FAILED,
                "error": str(e),
                "updated_at": datetime.utcnow()
            }
        )
        
        # Log error step
        steps_repo.add_step(
            ProcessingStep(
                document_id=document_id,
                step_name="document_processing",
                status=ProcessingStatus.FAILED,
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
        )
        
        # Update claim status
        claims_repo.update_claim(
            claim_id,
            {
                "status": SubmissionStatus.ERROR,
                "error": f"Document processing failed: {str(e)}",
                "updated_at": datetime.utcnow()
            }
        )
