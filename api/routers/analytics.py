"""
Analytics and reporting API endpoints.

Provides insights and metrics about claims processing.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import csv
import io

from database.repository import ClaimsRepository, get_db
from models.claims import Claim, ClaimStatus

# Configure logging
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])

# --- Response Models ---

class StatusMetrics(BaseModel):
    """Metrics about claim statuses."""
    total: int
    draft: int
    processing: int
    processed: int
    submitted: int
    approved: int
    rejected: int
    error: int

class TimeSeriesPoint(BaseModel):
    """A single data point in a time series."""
    date: str
    count: int

class TimeSeriesMetrics(BaseModel):
    """Time series metrics."""
    data: List[TimeSeriesPoint]
    period: str
    total: int

class HospitalMetrics(BaseModel):
    """Metrics grouped by hospital."""
    hospital_id: str
    claims_count: int
    avg_processing_time_hours: Optional[float]
    approval_rate: float
    avg_claim_amount: Optional[float]

class AnalyticsResponse(BaseModel):
    """Comprehensive analytics response."""
    status_metrics: StatusMetrics
    claims_trend: TimeSeriesMetrics
    hospital_metrics: List[HospitalMetrics]
    top_packages: List[Dict[str, Any]]
    processing_times: Dict[str, float]

# --- Helper Functions ---

def _get_date_range(days: int = 30) -> Tuple[datetime, datetime]:
    """Get a date range for analytics queries."""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    return start_date, end_date

# --- API Endpoints ---

@router.get("/status-metrics", response_model=StatusMetrics)
async def get_status_metrics(
    hospital_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    claims_repo: ClaimsRepository = Depends()
):
    """
    Get metrics about claim statuses.
    
    Returns counts of claims in each status category.
    """
    try:
        if not start_date or not end_date:
            start_date, end_date = _get_date_range()
            
        filters = {"created_at__gte": start_date, "created_at__lte": end_date}
        if hospital_id:
            filters["hospital_id"] = hospital_id
            
        claims = claims_repo.list_claims(filters=filters, limit=0)[0]
        
        status_counts = {
            "draft": 0,
            "processing": 0,
            "processed": 0,
            "submitted": 0,
            "approved": 0,
            "rejected": 0,
            "error": 0
        }
        
        for claim in claims:
            status = claim.status.value.lower()
            if status in status_counts:
                status_counts[status] += 1
        
        return StatusMetrics(
            total=len(claims),
            **status_counts
        )
        
    except Exception as e:
        logger.exception(f"Error getting status metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status metrics: {str(e)}")

@router.get("/claims-trend", response_model=TimeSeriesMetrics)
async def get_claims_trend(
    period: str = Query("day", regex="^(day|week|month)$"),
    days: int = Query(30, ge=1, le=365),
    hospital_id: Optional[str] = None,
    claims_repo: ClaimsRepository = Depends()
):
    """
    Get time series data about claims over time.
    
    Returns the number of claims created in each time period.
    """
    try:
        start_date, end_date = _get_date_range(days)
        
        # This would be implemented in the repository
        # For now, return a placeholder
        return TimeSeriesMetrics(
            data=[
                TimeSeriesPoint(date="2023-01-01", count=10),
                TimeSeriesPoint(date="2023-01-02", count=15),
            ],
            period=period,
            total=25
        )
        
    except Exception as e:
        logger.exception(f"Error getting claims trend: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get claims trend: {str(e)}")

@router.get("/hospital-metrics", response_model=List[HospitalMetrics])
async def get_hospital_metrics(
    limit: int = Query(10, ge=1, le=100),
    claims_repo: ClaimsRepository = Depends()
):
    """
    Get metrics grouped by hospital.
    
    Returns a list of hospitals with their claim metrics.
    """
    try:
        # This would be implemented in the repository
        # For now, return a placeholder
        return [
            HospitalMetrics(
                hospital_id="hosp_123",
                claims_count=100,
                avg_processing_time_hours=12.5,
                approval_rate=0.85,
                avg_claim_amount=15000.0
            )
        ]
        
    except Exception as e:
        logger.exception(f"Error getting hospital metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get hospital metrics: {str(e)}")

@router.get("/dashboard", response_model=AnalyticsResponse)
async def get_dashboard(
    hospital_id: Optional[str] = None,
    days: int = Query(30, ge=1, le=365),
    claims_repo: ClaimsRepository = Depends()
):
    """
    Get a comprehensive analytics dashboard.
    
    Returns all key metrics in a single response.
    """
    try:
        # Get status metrics
        status_metrics = await get_status_metrics(hospital_id=hospital_id, claims_repo=claims_repo)
        
        # Get claims trend
        claims_trend = await get_claims_trend(
            period="day",
            days=days,
            hospital_id=hospital_id,
            claims_repo=claims_repo
        )
        
        # Get hospital metrics
        hospital_metrics = await get_hospital_metrics(claims_repo=claims_repo)
        
        # Placeholder for top packages
        top_packages = [
            {"package_id": "pkg_1", "name": "Cardiac Surgery", "count": 50},
            {"package_id": "pkg_2", "name": "Orthopedic Surgery", "count": 45}
        ]
        
        # Placeholder for processing times
        processing_times = {
            "document_processing_avg_seconds": 45.2,
            "claims_processing_avg_seconds": 8.7,
            "total_processing_avg_seconds": 53.9
        }
        
        return AnalyticsResponse(
            status_metrics=status_metrics,
            claims_trend=claims_trend,
            hospital_metrics=hospital_metrics,
            top_packages=top_packages,
            processing_times=processing_times
        )
        
    except Exception as e:
        logger.exception(f"Error getting dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")


# --- CSV Export Endpoints ---

@router.get("/export/claims")
async def export_claims_csv(
    hospital_id: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    claims_repo: ClaimsRepository = Depends()
):
    """
    Export claims data as CSV file.
    
    Filters:
    - hospital_id: Filter by specific hospital
    - status: Filter by claim status
    - start_date: Filter claims from this date (YYYY-MM-DD)
    - end_date: Filter claims until this date (YYYY-MM-DD)
    """
    try:
        # Get claims data based on filters
        claims = claims_repo.get_claims_filtered(
            hospital_id=hospital_id,
            status=status,
            start_date=start_date,
            end_date=end_date
        )
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Claim ID', 'Hospital ID', 'Patient ID', 'Status', 
            'Package Code', 'Amount', 'Created Date', 'Last Updated',
            'Processing Time (seconds)', 'Error Message'
        ])
        
        # Write data rows
        for claim in claims:
            processing_time = ""
            if hasattr(claim, 'created_at') and hasattr(claim, 'last_updated'):
                if claim.created_at and claim.last_updated:
                    delta = claim.last_updated - claim.created_at
                    processing_time = str(delta.total_seconds())
            
            writer.writerow([
                getattr(claim, 'id', ''),
                getattr(claim, 'hospital_id', ''),
                getattr(claim, 'patient_id', ''),
                getattr(claim, 'status', ''),
                getattr(claim, 'package_code', ''),
                getattr(claim, 'amount', ''),
                getattr(claim, 'created_at', ''),
                getattr(claim, 'last_updated', ''),
                processing_time,
                getattr(claim, 'error_message', '')
            ])
        
        # Prepare response
        output.seek(0)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"claims_export_{timestamp}.csv"
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.exception(f"Error exporting claims CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export claims: {str(e)}")


@router.get("/export/analytics")
async def export_analytics_csv(
    hospital_id: Optional[str] = None,
    days: int = Query(30, ge=1, le=365),
    claims_repo: ClaimsRepository = Depends()
):
    """
    Export analytics summary as CSV file.
    
    Includes status metrics, trends, and hospital performance data.
    """
    try:
        # Get analytics data
        status_metrics = await get_status_metrics(hospital_id=hospital_id, claims_repo=claims_repo)
        claims_trend = await get_claims_trend(
            period="day", days=days, hospital_id=hospital_id, claims_repo=claims_repo
        )
        hospital_metrics = await get_hospital_metrics(claims_repo=claims_repo)
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write status metrics section
        writer.writerow(['=== STATUS METRICS ==='])
        writer.writerow(['Status', 'Count'])
        writer.writerow(['Total', status_metrics.total])
        writer.writerow(['Draft', status_metrics.draft])
        writer.writerow(['Processing', status_metrics.processing])
        writer.writerow(['Processed', status_metrics.processed])
        writer.writerow(['Submitted', status_metrics.submitted])
        writer.writerow(['Approved', status_metrics.approved])
        writer.writerow(['Rejected', status_metrics.rejected])
        writer.writerow(['Error', status_metrics.error])
        writer.writerow([])  # Empty row
        
        # Write claims trend section
        writer.writerow(['=== CLAIMS TREND ==='])
        writer.writerow(['Date', 'Count'])
        for point in claims_trend.data:
            writer.writerow([point.date, point.count])
        writer.writerow([])  # Empty row
        
        # Write hospital metrics section
        writer.writerow(['=== HOSPITAL METRICS ==='])
        writer.writerow(['Hospital ID', 'Total Claims', 'Approval Rate', 'Avg Processing Time'])
        for hospital in hospital_metrics.hospitals:
            writer.writerow([
                hospital.hospital_id,
                hospital.total_claims,
                f"{hospital.approval_rate:.2f}%",
                f"{hospital.avg_processing_time:.1f}s"
            ])
        
        # Prepare response
        output.seek(0)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analytics_export_{timestamp}.csv"
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.exception(f"Error exporting analytics CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export analytics: {str(e)}")


@router.get("/export/hospital-performance")
async def export_hospital_performance_csv(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    claims_repo: ClaimsRepository = Depends()
):
    """
    Export detailed hospital performance metrics as CSV.
    
    Includes per-hospital breakdown of claims, approval rates, and processing times.
    """
    try:
        # Get hospital performance data
        hospital_metrics = await get_hospital_metrics(claims_repo=claims_repo)
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Hospital ID', 'Hospital Name', 'Total Claims', 'Approved Claims',
            'Rejected Claims', 'Pending Claims', 'Approval Rate (%)',
            'Rejection Rate (%)', 'Average Processing Time (seconds)',
            'Fastest Processing Time (seconds)', 'Slowest Processing Time (seconds)',
            'Total Amount Claimed', 'Total Amount Approved'
        ])
        
        # Write data rows
        for hospital in hospital_metrics.hospitals:
            writer.writerow([
                hospital.hospital_id,
                getattr(hospital, 'hospital_name', 'Unknown'),
                hospital.total_claims,
                getattr(hospital, 'approved_claims', 0),
                getattr(hospital, 'rejected_claims', 0),
                getattr(hospital, 'pending_claims', 0),
                f"{hospital.approval_rate:.2f}",
                f"{getattr(hospital, 'rejection_rate', 0):.2f}",
                f"{hospital.avg_processing_time:.1f}",
                getattr(hospital, 'min_processing_time', 0),
                getattr(hospital, 'max_processing_time', 0),
                getattr(hospital, 'total_amount_claimed', 0),
                getattr(hospital, 'total_amount_approved', 0)
            ])
        
        # Prepare response
        output.seek(0)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hospital_performance_{timestamp}.csv"
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.exception(f"Error exporting hospital performance CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export hospital performance: {str(e)}")
