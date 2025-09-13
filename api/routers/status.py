"""
Status Tracking API Endpoints.

Provides endpoints for tracking claim statuses and receiving notifications.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from services.status_tracking_service import StatusTrackingService, StatusCheckResult, StatusUpdateAction
from services.notification_service import NotificationService, NotificationChannel, NotificationPriority
from database.repository import ClaimsRepository, get_db
from models.claims import Claim, ClaimStatus, ClaimStatusUpdate

# Configure logging
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/status", tags=["status"])

# Initialize services
db_url = "sqlite:///./sanjeevni.db"  # Should come from config
status_service = StatusTrackingService(db_url=db_url)
notification_service = NotificationService()

# --- Request/Response Models ---

class StatusCheckRequest(BaseModel):
    """Request model for checking claim status."""
    claim_ids: List[str] = Field(..., description="List of claim IDs to check")
    force_update: bool = Field(False, description="Force a status update even if recently checked")


class StatusCheckResponse(BaseModel):
    """Response model for status check results."""
    claim_id: str
    status: str
    status_changed: bool
    previous_status: Optional[str] = None
    last_updated: datetime
    actions_taken: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StatusHistoryResponse(BaseModel):
    """Response model for claim status history."""
    claim_id: str
    history: List[Dict[str, Any]] = Field(default_factory=list)


class NotificationPreference(BaseModel):
    """Model for notification preferences."""
    email: bool = True
    sms: bool = False
    in_app: bool = True
    slack: bool = False
    whatsapp: bool = False


class StatusUpdateNotification(BaseModel):
    """Model for status update notifications."""
    claim_id: str
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SubscriptionRequest(BaseModel):
    """Request model for subscribing to status updates."""
    claim_ids: List[str] = Field(..., description="List of claim IDs to subscribe to")
    callback_url: Optional[str] = Field(None, description="Webhook URL to receive updates")
    email: Optional[str] = Field(None, description="Email address for notifications")
    phone: Optional[str] = Field(None, description="Phone number for SMS notifications")
    preferences: NotificationPreference = Field(default_factory=NotificationPreference)


class SubscriptionResponse(BaseModel):
    """Response model for subscription operations."""
    success: bool
    message: str
    subscription_id: Optional[str] = None
    claim_ids: List[str] = Field(default_factory=list)
    failed_claim_ids: List[str] = Field(default_factory=list)


# --- Helper Functions ---

def map_claim_status(status: ClaimStatus) -> str:
    """Map internal claim status to a human-readable string."""
    return status.value.upper().replace("_", " ")


def map_notification_channels(prefs: NotificationPreference) -> List[NotificationChannel]:
    """Convert notification preferences to a list of channels."""
    channels = []
    if prefs.email:
        channels.append(NotificationChannel.EMAIL)
    if prefs.sms:
        channels.append(NotificationChannel.SMS)
    if prefs.in_app:
        channels.append(NotificationChannel.IN_APP)
    if prefs.slack:
        channels.append(NotificationChannel.SLACK)
    if prefs.whatsapp:
        channels.append(NotificationChannel.WHATSAPP)
    return channels


# --- API Endpoints ---

@router.post("/check", response_model=List[StatusCheckResponse])
async def check_claim_status(
    request: StatusCheckRequest,
    background_tasks: BackgroundTasks,
    db: ClaimsRepository = Depends(get_db)
):
    """
    Check the status of one or more claims.
    
    This endpoint will check the current status of the specified claims and
    return the results. If the status has changed, appropriate actions will
    be triggered (e.g., notifications).
    """
    results = []
    
    for claim_id in request.claim_ids:
        try:
            # Get the claim from the database
            claim = db.get_claim_by_id(claim_id)
            if not claim:
                logger.warning(f"Claim {claim_id} not found")
                results.append(StatusCheckResponse(
                    claim_id=claim_id,
                    status="NOT_FOUND",
                    status_changed=False,
                    last_updated=datetime.utcnow(),
                    metadata={"error": "Claim not found"}
                ))
                continue
            
            # Check if we should force an update
            force_update = request.force_update
            if not force_update and claim.last_status_check:
                # Only check if it's been more than 1 hour since the last check
                time_since_last_check = datetime.utcnow() - claim.last_status_check
                force_update = time_since_last_check > timedelta(hours=1)
            
            if not force_update:
                # Return the current status without checking
                results.append(StatusCheckResponse(
                    claim_id=claim_id,
                    status=map_claim_status(claim.status),
                    status_changed=False,
                    last_updated=claim.last_status_update or datetime.utcnow(),
                    metadata={"cached": True}
                ))
                continue
            
            # Check the status (this will trigger any necessary actions)
            result = await status_service.check_claim_status(claim_id)
            
            # Map the result to our response model
            response = StatusCheckResponse(
                claim_id=claim_id,
                status=map_claim_status(result.current_status),
                status_changed=result.status_changed,
                previous_status=map_claim_status(result.previous_status) if result.previous_status else None,
                last_updated=result.timestamp,
                actions_taken=[action.name for action in result.actions_taken],
                metadata=result.metadata
            )
            
            results.append(response)
            
            # If status changed, send notifications in the background
            if result.status_changed:
                background_tasks.add_task(
                    _send_status_notification,
                    claim_id=claim_id,
                    previous_status=result.previous_status,
                    new_status=result.current_status,
                    metadata=result.metadata
                )
            
        except Exception as e:
            logger.error(f"Error checking status for claim {claim_id}: {str(e)}")
            results.append(StatusCheckResponse(
                claim_id=claim_id,
                status="ERROR",
                status_changed=False,
                last_updated=datetime.utcnow(),
                metadata={"error": str(e)}
            ))
    
    return results


@router.get("/history/{claim_id}", response_model=StatusHistoryResponse)
async def get_status_history(claim_id: str, db: ClaimsRepository = Depends(get_db)):
    """
    Get the status history for a claim.
    
    Returns a chronological list of status changes for the specified claim.
    """
    try:
        # In a real implementation, this would query a status history table
        # For now, we'll return a mock response
        history = status_service.get_claim_status_history(claim_id)
        
        # Convert to list of dicts for the response
        history_list = [
            {
                "status": map_claim_status(status),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "system"
            }
            for status in history
        ]
        
        return StatusHistoryResponse(
            claim_id=claim_id,
            history=history_list
        )
        
    except Exception as e:
        logger.error(f"Error getting status history for claim {claim_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status history: {str(e)}"
        )


@router.post("/subscribe", response_model=SubscriptionResponse)
async def subscribe_to_updates(request: SubscriptionRequest):
    """
    Subscribe to status updates for one or more claims.
    
    This will start tracking the specified claims and send notifications
    when their status changes.
    """
    success_claim_ids = []
    failed_claim_ids = []
    
    for claim_id in request.claim_ids:
        try:
            # Start tracking the claim
            await status_service.start_tracking_claim(claim_id)
            success_claim_ids.append(claim_id)
            
            # If a callback URL was provided, register it (implementation not shown)
            if request.callback_url:
                # In a real implementation, this would store the callback URL
                # and use it to send webhook notifications
                pass
                
            # If email or phone was provided, register for notifications
            if request.email or request.phone:
                # In a real implementation, this would store the contact info
                # and preferences in a database
                pass
                
        except Exception as e:
            logger.error(f"Failed to subscribe to updates for claim {claim_id}: {str(e)}")
            failed_claim_ids.append(claim_id)
    
    return SubscriptionResponse(
        success=len(success_claim_ids) > 0,
        message=f"Subscribed to updates for {len(success_claim_ids)} claims. "
               f"Failed to subscribe to {len(failed_claim_ids)} claims.",
        subscription_id=f"sub_{len(success_claim_ids)}_{datetime.utcnow().timestamp()}",
        claim_ids=success_claim_ids,
        failed_claim_ids=failed_claim_ids
    )


@router.post("/unsubscribe", response_model=SubscriptionResponse)
async def unsubscribe_from_updates(claim_ids: List[str]):
    """
    Unsubscribe from status updates for one or more claims.
    
    This will stop tracking the specified claims and cancel any scheduled
    status checks.
    """
    success_claim_ids = []
    failed_claim_ids = []
    
    for claim_id in claim_ids:
        try:
            await status_service.stop_tracking_claim(claim_id)
            success_claim_ids.append(claim_id)
        except Exception as e:
            logger.error(f"Failed to unsubscribe from updates for claim {claim_id}: {str(e)}")
            failed_claim_ids.append(claim_id)
    
    return SubscriptionResponse(
        success=len(success_claim_ids) > 0,
        message=f"Unsubscribed from updates for {len(success_claim_ids)} claims. "
               f"Failed to unsubscribe from {len(failed_claim_ids)} claims.",
        claim_ids=success_claim_ids,
        failed_claim_ids=failed_claim_ids
    )


@router.get("/active-subscriptions", response_model=List[str])
async def get_active_subscriptions():
    """
    Get a list of claim IDs that are currently being tracked.
    """
    return status_service.get_active_trackers()


# --- Background Tasks ---

async def _send_status_notification(
    claim_id: str,
    previous_status: ClaimStatus,
    new_status: ClaimStatus,
    metadata: Dict[str, Any]
) -> None:
    """
    Send a notification about a status change.
    
    This is called in the background when a claim's status changes.
    """
    try:
        # In a real implementation, this would look up the claim and its
        # notification preferences from the database
        
        # For now, we'll just log the notification
        logger.info(
            f"Status changed for claim {claim_id}: "
            f"{previous_status.value} -> {new_status.value}"
        )
        
        # Example: Send an email notification
        await notification_service.send_notification(
            recipient={
                "id": claim_id,
                "email": "admin@example.com",  # Would come from user preferences
                "name": "Claim Administrator"
            },
            subject=f"Claim Status Update: {claim_id}",
            message=(
                f"The status of claim {claim_id} has changed from "
                f"{previous_status.value.upper()} to {new_status.value.upper()}.\n\n"
                f"Additional information: {metadata.get('message', 'No additional information')}"
            ),
            channel=NotificationChannel.EMAIL,
            priority=NotificationPriority.HIGH if new_status in [
                ClaimStatus.REJECTED, 
                ClaimStatus.ESCALATED
            ] else NotificationPriority.NORMAL,
            metadata={
                "claim_id": claim_id,
                "previous_status": previous_status.value,
                "new_status": new_status.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to send status notification for claim {claim_id}: {str(e)}")


# --- Startup/Shutdown Handlers ---

@router.on_event("startup")
async def startup_event():
    """Initialize the status tracking service when the app starts."""
    try:
        await status_service.start()
        logger.info("Status tracking service started")
    except Exception as e:
        logger.error(f"Failed to start status tracking service: {str(e)}")
        raise


@router.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the app shuts down."""
    try:
        await status_service.stop()
        logger.info("Status tracking service stopped")
    except Exception as e:
        logger.error(f"Error stopping status tracking service: {str(e)}")
