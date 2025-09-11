"""
Sanjeevni Plus REST API

FastAPI application providing endpoints for healthcare claims processing
using AI-powered document analysis, vector search, and intelligent recommendations.
"""

import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from services.claims_intelligence_service import ClaimsIntelligenceService
from services.document_processing_service import DocumentProcessingService
from services.chroma_service import ChromaService
from services.granite_service import GraniteLanguageDetectionService
from services.granite_embedding_service import GraniteEmbeddingService
from services.knowledge_base_service import KnowledgeBaseService
from database.connection import DatabaseConnection
from database.schema import DatabaseSchema
from database.repository import ClaimsRepository, ProcessingStepsRepository, VectorQueriesRepository
from models.claims import SubmissionStatus, ComplianceStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app configuration
app = FastAPI(
    title="Sanjeevni Plus Healthcare Claims API",
    description="AI-powered healthcare claims processing system for PM-JAY automation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances (will be initialized on startup)
claims_intelligence_service: Optional[ClaimsIntelligenceService] = None
knowledge_base_service: Optional[KnowledgeBaseService] = None


# Pydantic models for API requests/responses
class ClaimProcessingRequest(BaseModel):
    """Request model for claim processing."""
    hospital_id: str = Field(..., description="Hospital identifier")
    claim_id: Optional[str] = Field(None, description="Optional existing claim ID")
    priority: str = Field("normal", description="Processing priority (low, normal, high)")


class ClaimProcessingResponse(BaseModel):
    """Response model for claim processing."""
    claim_id: str
    processing_status: str
    processing_time_ms: float
    document_analysis: Dict[str, Any]
    package_recommendations: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    compliance_status: str
    message: str


class ClaimAnalysisResponse(BaseModel):
    """Response model for claim analysis retrieval."""
    claim: Dict[str, Any]
    processing_steps: List[Dict[str, Any]]
    vector_queries: List[Dict[str, Any]]


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    service: str
    status: str
    timestamp: str
    dependencies: Dict[str, str]


class KnowledgeBaseLoadRequest(BaseModel):
    """Request model for knowledge base loading."""
    data_type: str = Field(..., description="Type of data to load (packages, guidelines, medical_codes)")
    force_reload: bool = Field(False, description="Force reload even if data exists")


# Dependency injection for services
async def get_claims_service() -> ClaimsIntelligenceService:
    """Get claims intelligence service instance."""
    if claims_intelligence_service is None:
        raise HTTPException(status_code=503, detail="Claims service not initialized")
    return claims_intelligence_service


async def get_knowledge_service() -> KnowledgeBaseService:
    """Get knowledge base service instance."""
    if knowledge_base_service is None:
        raise HTTPException(status_code=503, detail="Knowledge base service not initialized")
    return knowledge_base_service


@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    global claims_intelligence_service, knowledge_base_service
    
    try:
        logger.info("Initializing Sanjeevni Plus API services...")
        
        # Initialize database connection
        db_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./sanjeevni.db")
        db_connection = DatabaseConnection(db_url)
        
        # Create database schema
        schema = DatabaseSchema(db_connection)
        await schema.create_all_tables()
        
        # Initialize repositories
        claims_repository = ClaimsRepository(db_connection)
        steps_repository = ProcessingStepsRepository(db_connection)
        vector_repository = VectorQueriesRepository(db_connection)
        
        # Initialize services
        document_service = DocumentProcessingService()
        chroma_service = ChromaService()
        language_service = GraniteLanguageDetectionService()
        embedding_service = GraniteEmbeddingService()
        
        # Initialize claims intelligence service
        claims_intelligence_service = ClaimsIntelligenceService(
            document_service=document_service,
            chroma_service=chroma_service,
            language_service=language_service,
            embedding_service=embedding_service,
            claims_repository=claims_repository,
            steps_repository=steps_repository,
            vector_repository=vector_repository
        )
        
        # Initialize knowledge base service
        knowledge_base_service = KnowledgeBaseService(chroma_service)
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Sanjeevni Plus API...")
    # Add cleanup logic here if needed


@app.get("/health", response_model=HealthCheckResponse)
async def health_check(service: ClaimsIntelligenceService = Depends(get_claims_service)):
    """
    Comprehensive health check for all services.
    """
    try:
        health_result = await service.health_check()
        return HealthCheckResponse(**health_result)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            service="sanjeevni_plus_api",
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            dependencies={"error": str(e)}
        )


@app.post("/claims/process", response_model=ClaimProcessingResponse)
async def process_claim_document(
    file: UploadFile = File(..., description="Medical document to process"),
    request: ClaimProcessingRequest = Depends(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    service: ClaimsIntelligenceService = Depends(get_claims_service)
):
    """
    Process a medical document and generate intelligent claims recommendations.
    
    This endpoint:
    1. Accepts uploaded medical documents (PDF, images)
    2. Performs OCR and data extraction
    3. Conducts vector search for relevant guidelines
    4. Generates package recommendations
    5. Performs risk assessment and compliance checking
    """
    try:
        # Validate file type
        allowed_types = ["application/pdf", "image/jpeg", "image/png", "image/tiff"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. Allowed: {allowed_types}"
            )
        
        # Save uploaded file temporarily
        upload_dir = Path("./temp_uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"{datetime.now().timestamp()}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the document
        logger.info(f"Processing claim document: {file.filename} for hospital: {request.hospital_id}")
        
        result = await service.process_claim_document(
            document_path=str(file_path),
            hospital_id=request.hospital_id,
            claim_id=request.claim_id
        )
        
        # Schedule file cleanup
        background_tasks.add_task(cleanup_temp_file, file_path)
        
        # Prepare response
        response = ClaimProcessingResponse(
            claim_id=result["claim_id"],
            processing_status=result["processing_status"],
            processing_time_ms=result["processing_time_ms"],
            document_analysis=result["document_analysis"],
            package_recommendations=result["package_recommendations"],
            risk_assessment=result["risk_assessment"],
            compliance_status=result["compliance_status"],
            message="Document processed successfully"
        )
        
        logger.info(f"Successfully processed claim: {result['claim_id']}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to process claim document: {e}")
        # Cleanup file on error
        if 'file_path' in locals():
            background_tasks.add_task(cleanup_temp_file, file_path)
        
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/claims/{claim_id}", response_model=ClaimAnalysisResponse)
async def get_claim_analysis(
    claim_id: str,
    service: ClaimsIntelligenceService = Depends(get_claims_service)
):
    """
    Retrieve complete analysis for an existing claim.
    """
    try:
        result = await service.get_claim_analysis(claim_id)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")
        
        return ClaimAnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve claim analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@app.post("/claims/{claim_id}/reprocess")
async def reprocess_claim(
    claim_id: str,
    service: ClaimsIntelligenceService = Depends(get_claims_service)
):
    """
    Reprocess an existing claim with updated logic.
    """
    try:
        result = await service.reprocess_claim(claim_id)
        return {"message": "Claim reprocessed successfully", "result": result}
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to reprocess claim: {e}")
        raise HTTPException(status_code=500, detail=f"Reprocessing failed: {str(e)}")


@app.get("/claims")
async def list_claims(
    hospital_id: Optional[str] = None,
    status: Optional[SubmissionStatus] = None,
    limit: int = 50,
    offset: int = 0,
    service: ClaimsIntelligenceService = Depends(get_claims_service)
):
    """
    List claims with optional filtering.
    """
    try:
        # This would require additional repository methods
        # For now, return a placeholder response
        return {
            "claims": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "message": "Claims listing endpoint - implementation pending"
        }
        
    except Exception as e:
        logger.error(f"Failed to list claims: {e}")
        raise HTTPException(status_code=500, detail=f"Listing failed: {str(e)}")


@app.post("/knowledge-base/load")
async def load_knowledge_base(
    request: KnowledgeBaseLoadRequest,
    background_tasks: BackgroundTasks,
    service: KnowledgeBaseService = Depends(get_knowledge_service)
):
    """
    Load data into the knowledge base (PM-JAY packages, guidelines, medical codes).
    """
    try:
        # Schedule knowledge base loading as background task
        if request.data_type == "packages":
            background_tasks.add_task(service.load_pmjay_packages, request.force_reload)
        elif request.data_type == "guidelines":
            background_tasks.add_task(service.load_pmjay_guidelines, request.force_reload)
        elif request.data_type == "medical_codes":
            background_tasks.add_task(service.load_medical_codes, request.force_reload)
        elif request.data_type == "all":
            background_tasks.add_task(service.load_all_data, request.force_reload)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid data_type: {request.data_type}. Allowed: packages, guidelines, medical_codes, all"
            )
        
        return {
            "message": f"Knowledge base loading started for: {request.data_type}",
            "status": "in_progress"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start knowledge base loading: {e}")
        raise HTTPException(status_code=500, detail=f"Loading failed: {str(e)}")


@app.get("/knowledge-base/status")
async def get_knowledge_base_status(service: KnowledgeBaseService = Depends(get_knowledge_service)):
    """
    Get status of the knowledge base collections.
    """
    try:
        # This would require additional service methods
        return {
            "collections": {
                "pmjay_guidelines": {"status": "loaded", "document_count": 0},
                "medical_codes": {"status": "loaded", "document_count": 0}
            },
            "last_updated": datetime.now().isoformat(),
            "message": "Knowledge base status endpoint - implementation pending"
        }
        
    except Exception as e:
        logger.error(f"Failed to get knowledge base status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@app.get("/search/guidelines")
async def search_guidelines(
    query: str,
    limit: int = 10,
    service: ClaimsIntelligenceService = Depends(get_claims_service)
):
    """
    Search PM-JAY guidelines using vector similarity.
    """
    try:
        results = service.chroma_service.search_pmjay_guidelines(query, n_results=limit)
        
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Failed to search guidelines: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/search/medical-codes")
async def search_medical_codes(
    query: str,
    limit: int = 10,
    service: ClaimsIntelligenceService = Depends(get_claims_service)
):
    """
    Search medical codes using vector similarity.
    """
    try:
        results = service.chroma_service.search_medical_codes(query, n_results=limit)
        
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Failed to search medical codes: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/analytics/dashboard")
async def get_analytics_dashboard():
    """
    Get analytics dashboard data.
    """
    try:
        # Placeholder for analytics implementation
        return {
            "total_claims": 0,
            "claims_by_status": {},
            "processing_metrics": {
                "avg_processing_time_ms": 0,
                "success_rate": 0.0
            },
            "risk_distribution": {},
            "compliance_metrics": {},
            "message": "Analytics dashboard - implementation pending"
        }
        
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


# Utility functions
async def cleanup_temp_file(file_path: Path):
    """Clean up temporary uploaded file."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found", "path": str(request.url)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "path": str(request.url)}
    )


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


