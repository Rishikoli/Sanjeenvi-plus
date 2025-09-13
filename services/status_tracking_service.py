"""
Status Tracking Service

Monitors claim statuses and sends notifications when statuses change.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any
from enum import Enum, auto
import json

from pydantic import BaseModel, Field
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from database.repository import ClaimsRepository
from models.claims import Claim, ClaimStatus, ClaimStatusUpdate
from services.notification_service import NotificationService

logger = logging.getLogger(__name__)

class StatusUpdateAction(Enum):
    """Actions that can be taken when a status update is detected."""
    NOTIFY = auto()
    ESCALATE = auto()
    UPDATE_RECORD = auto()
    TRIGGER_WORKFLOW = auto()

class StatusCheckResult(BaseModel):
    """Result of a status check operation."""
    claim_id: str
    previous_status: ClaimStatus
    current_status: ClaimStatus
    status_changed: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)
    actions_taken: List[StatusUpdateAction] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusTrackingRule(BaseModel):
    """Rule for handling status updates."""
    name: str
    description: str
    condition: Callable[[Claim, ClaimStatus, Dict[str, Any]], bool]
    actions: List[StatusUpdateAction]
    priority: int = 0
    active: bool = True

class StatusTrackingService:
    """Service for tracking claim statuses and triggering actions on changes."""
    
    def __init__(self, db_url: str):
        """Initialize the status tracking service."""
        self.db_url = db_url
        self.repository = ClaimsRepository(db_url)
        self.notification_service = NotificationService()
        self.scheduler = AsyncIOScheduler()
        self.tracking_rules: List[StatusTrackingRule] = []
        self.active_trackers: Dict[str, asyncio.Task] = {}
        self.status_history: Dict[str, List[ClaimStatus]] = {}
        
        # Register default tracking rules
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """Register default status tracking rules."""
        # Rule for notifying when a claim is approved
        self.add_tracking_rule(
            name="notify_on_approval",
            description="Notify when a claim is approved",
            condition=lambda claim, new_status, _: (
                new_status == ClaimStatus.APPROVED and 
                claim.status != ClaimStatus.APPROVED
            ),
            actions=[
                StatusUpdateAction.NOTIFY,
                StatusUpdateAction.UPDATE_RECORD
            ]
        )
        
        # Rule for escalating claims stuck in review
        self.add_tracking_rule(
            name="escalate_stale_reviews",
            description="Escalate claims stuck in review for more than 7 days",
            condition=lambda claim, new_status, _: (
                new_status == ClaimStatus.UNDER_REVIEW and
                claim.status == ClaimStatus.UNDER_REVIEW and
                claim.last_status_update and
                (datetime.utcnow() - claim.last_status_update).days > 7
            ),
            actions=[
                StatusUpdateAction.ESCALATE,
                StatusUpdateAction.NOTIFY
            ]
        )
        
        # Rule for handling rejected claims
        self.add_tracking_rule(
            name="handle_rejection",
            description="Handle claim rejection with appropriate notifications",
            condition=lambda claim, new_status, _: (
                new_status == ClaimStatus.REJECTED and
                claim.status != ClaimStatus.REJECTED
            ),
            actions=[
                StatusUpdateAction.NOTIFY,
                StatusUpdateAction.UPDATE_RECORD,
                StatusUpdateAction.TRIGGER_WORKFLOW
            ]
        )
    
    def add_tracking_rule(
        self,
        name: str,
        description: str,
        condition: Callable[[Claim, ClaimStatus, Dict[str, Any]], bool],
        actions: List[StatusUpdateAction],
        priority: int = 0
    ) -> None:
        """Add a new status tracking rule.
        
        Args:
            name: Unique name for the rule
            description: Description of what the rule does
            condition: Callable that takes (claim, new_status, metadata) and returns a boolean
            actions: List of actions to take when the condition is met
            priority: Priority of the rule (higher numbers are evaluated first)
        """
        rule = StatusTrackingRule(
            name=name,
            description=description,
            condition=condition,
            actions=actions,
            priority=priority
        )
        self.tracking_rules.append(rule)
        # Sort rules by priority (highest first)
        self.tracking_rules.sort(key=lambda r: r.priority, reverse=True)
    
    async def start_tracking_claim(self, claim_id: str, check_interval: int = 3600) -> None:
        """Start tracking a claim's status.
        
        Args:
            claim_id: ID of the claim to track
            check_interval: How often to check for status updates (in seconds)
        """
        if claim_id in self.active_trackers:
            logger.warning(f"Already tracking claim {claim_id}")
            return
        
        async def track_claim():
            """Background task to track a claim's status."""
            try:
                while claim_id in self.active_trackers:
                    try:
                        await self.check_claim_status(claim_id)
                    except Exception as e:
                        logger.error(f"Error tracking claim {claim_id}: {e}")
                    
                    # Wait for the next check interval
                    await asyncio.sleep(check_interval)
            except asyncio.CancelledError:
                logger.info(f"Stopped tracking claim {claim_id}")
            except Exception as e:
                logger.error(f"Unexpected error in tracking task for claim {claim_id}: {e}")
                raise
        
        # Start the tracking task
        self.active_trackers[claim_id] = asyncio.create_task(track_claim())
        logger.info(f"Started tracking claim {claim_id}")
    
    async def stop_tracking_claim(self, claim_id: str) -> None:
        """Stop tracking a claim's status.
        
        Args:
            claim_id: ID of the claim to stop tracking
        """
        if claim_id not in self.active_trackers:
            return
        
        # Cancel the tracking task
        task = self.active_trackers[claim_id]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Remove from active trackers
        del self.active_trackers[claim_id]
        logger.info(f"Stopped tracking claim {claim_id}")
    
    async def check_claim_status(self, claim_id: str) -> StatusCheckResult:
        """Check the current status of a claim and process any updates.
        
        Args:
            claim_id: ID of the claim to check
            
        Returns:
            StatusCheckResult with details of the status check
        """
        try:
            # Get the current claim from the database
            claim = self.repository.get_claim_by_id(claim_id)
            if not claim:
                raise ValueError(f"Claim {claim_id} not found")
            
            # Get the previous status from history or current status
            previous_status = claim.status
            
            # In a real implementation, we would check the external system here
            # For now, we'll just use the current status from the database
            current_status = claim.status
            
            # Check if status has changed
            status_changed = (current_status != previous_status)
            
            # Create the result object
            result = StatusCheckResult(
                claim_id=claim_id,
                previous_status=previous_status,
                current_status=current_status,
                status_changed=status_changed,
                metadata={
                    "checked_at": datetime.utcnow().isoformat(),
                    "claim_type": claim.claim_type.value if hasattr(claim, 'claim_type') else "unknown"
                }
            )
            
            # If status changed, process rules
            if status_changed:
                await self._process_status_update(claim, current_status, result)
            
            # Update status history
            if claim_id not in self.status_history:
                self.status_history[claim_id] = []
            self.status_history[claim_id].append(current_status)
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking status for claim {claim_id}: {e}")
            raise
    
    async def _process_status_update(
        self, 
        claim: Claim, 
        new_status: ClaimStatus,
        result: StatusCheckResult
    ) -> None:
        """Process a status update by applying all matching rules.
        
        Args:
            claim: The claim being updated
            new_status: The new status
            result: The status check result to update with actions taken
        """
        try:
            # Get metadata for rule processing
            metadata = {
                "previous_status": claim.status,
                "new_status": new_status,
                "claim_id": claim.id,
                "patient_id": getattr(claim, 'patient_id', None),
                "hospital_id": getattr(claim, 'hospital_id', None),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Apply all matching rules
            for rule in self.tracking_rules:
                if not rule.active:
                    continue
                    
                try:
                    if rule.condition(claim, new_status, metadata):
                        logger.info(f"Rule '{rule.name}' matched for claim {claim.id}")
                        
                        # Execute all actions for this rule
                        for action in rule.actions:
                            try:
                                await self._execute_action(action, claim, new_status, metadata)
                                result.actions_taken.append(action)
                            except Exception as e:
                                logger.error(
                                    f"Error executing action {action} for rule '{rule.name}' "
                                    f"on claim {claim.id}: {e}"
                                )
                                
                except Exception as e:
                    logger.error(
                        f"Error evaluating rule '{rule.name}' for claim {claim.id}: {e}"
                    )
            
            # Update the claim status in the database
            claim.status = new_status
            claim.last_status_update = datetime.utcnow()
            self.repository.update_claim(claim.id, claim.dict())
            
        except Exception as e:
            logger.error(f"Error processing status update for claim {claim.id}: {e}")
            raise
    
    async def _execute_action(
        self,
        action: StatusUpdateAction,
        claim: Claim,
        new_status: ClaimStatus,
        metadata: Dict[str, Any]
    ) -> None:
        """Execute a single status update action.
        
        Args:
            action: The action to execute
            claim: The claim being updated
            new_status: The new status
            metadata: Additional metadata for the update
        """
        if action == StatusUpdateAction.NOTIFY:
            await self._send_status_notification(claim, new_status, metadata)
            
        elif action == StatusUpdateAction.ESCALATE:
            await self._escalate_claim(claim, new_status, metadata)
            
        elif action == StatusUpdateAction.UPDATE_RECORD:
            # Already handled in _process_status_update
            pass
            
        elif action == StatusUpdateAction.TRIGGER_WORKFLOW:
            await self._trigger_workflow(claim, new_status, metadata)
    
    async def _send_status_notification(
        self,
        claim: Claim,
        new_status: ClaimStatus,
        metadata: Dict[str, Any]
    ) -> None:
        """Send a notification about a status update.
        
        Args:
            claim: The claim being updated
            new_status: The new status
            metadata: Additional metadata for the notification
        """
        try:
            # Get recipient information from claim or metadata
            recipient = getattr(claim, 'submitted_by', None) or metadata.get('submitted_by')
            if not recipient:
                logger.warning(f"No recipient found for claim {claim.id} notification")
                return
            
            # Create notification message
            message = (
                f"Claim {claim.id} status updated to {new_status.value}. "
                f"Previous status: {metadata.get('previous_status', 'N/A')}."
            )
            
            # Add any additional context
            if 'rejection_reason' in metadata:
                message += f" Reason: {metadata['rejection_reason']}"
            
            # Send the notification
            await self.notification_service.send_notification(
                recipient=recipient,
                subject=f"Claim Status Update: {claim.id}",
                message=message,
                metadata={
                    "claim_id": claim.id,
                    "previous_status": metadata.get('previous_status'),
                    "new_status": new_status.value,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Sent status notification for claim {claim.id} to {recipient}")
            
        except Exception as e:
            logger.error(f"Error sending status notification for claim {claim.id}: {e}")
            raise
    
    async def _escalate_claim(
        self,
        claim: Claim,
        new_status: ClaimStatus,
        metadata: Dict[str, Any]
    ) -> None:
        """Escalate a claim to a supervisor or support team.
        
        Args:
            claim: The claim to escalate
            new_status: The new status
            metadata: Additional metadata for the escalation
        """
        try:
            # In a real implementation, this would:
            # 1. Identify the appropriate escalation path
            # 2. Create an escalation ticket
            # 3. Notify the relevant team
            
            logger.info(
                f"Escalating claim {claim.id} in status {new_status.value} "
                f"(previously {metadata.get('previous_status', 'N/A')})"
            )
            
            # Example: Create an escalation record in the database
            escalation = {
                "claim_id": claim.id,
                "escalated_at": datetime.utcnow(),
                "reason": f"Claim in status {new_status.value} requires attention",
                "assigned_to": "support_team",
                "status": "open"
            }
            
            # In a real implementation, save this to a database
            # self.repository.create_escalation(escalation)
            
        except Exception as e:
            logger.error(f"Error escalating claim {claim.id}: {e}")
            raise
    
    async def _trigger_workflow(
        self,
        claim: Claim,
        new_status: ClaimStatus,
        metadata: Dict[str, Any]
    ) -> None:
        """Trigger a workflow based on the status update.
        
        Args:
            claim: The claim that was updated
            new_status: The new status
            metadata: Additional metadata for the workflow
        """
        try:
            # In a real implementation, this would trigger a workflow
            # based on the status change (e.g., approval workflow, rejection workflow)
            
            workflow_type = f"{metadata.get('previous_status', 'NONE')}_TO_{new_status.value}"
            logger.info(
                f"Triggering workflow {workflow_type} for claim {claim.id}"
            )
            
            # Example: Log the workflow event
            workflow_event = {
                "claim_id": claim.id,
                "workflow_type": workflow_type,
                "triggered_at": datetime.utcnow(),
                "status": "pending",
                "metadata": metadata
            }
            
            # In a real implementation, save this to a workflow queue or database
            # self.repository.create_workflow_event(workflow_event)
            
        except Exception as e:
            logger.error(f"Error triggering workflow for claim {claim.id}: {e}")
            raise
    
    async def start(self) -> None:
        """Start the status tracking service."""
        if self.scheduler.running:
            logger.warning("Status tracking service is already running")
            return
        
        # Start the scheduler
        self.scheduler.start()
        
        # Add a periodic job to check for stale claims
        self.scheduler.add_job(
            self._check_stale_claims,
            trigger=IntervalTrigger(hours=1),
            id="stale_claims_check",
            name="Check for stale claims",
            replace_existing=True
        )
        
        logger.info("Status tracking service started")
    
    async def stop(self) -> None:
        """Stop the status tracking service."""
        if not self.scheduler.running:
            return
        
        # Stop all tracking tasks
        for claim_id in list(self.active_trackers.keys()):
            await self.stop_tracking_claim(claim_id)
        
        # Shut down the scheduler
        self.scheduler.shutdown(wait=False)
        
        logger.info("Status tracking service stopped")
    
    async def _check_stale_claims(self) -> None:
        """Periodically check for claims that haven't been updated recently."""
        try:
            # In a real implementation, this would query the database for claims
            # that haven't been updated in a while and might need attention
            
            # Example: Get claims that haven't been updated in 7 days
            # stale_cutoff = datetime.utcnow() - timedelta(days=7)
            # stale_claims = self.repository.get_claims_updated_before(stale_cutoff)
            
            # For each stale claim, check if it needs attention
            # for claim in stale_claims:
            #     await self.check_claim_status(claim.id)
            
            pass
            
        except Exception as e:
            logger.error(f"Error checking for stale claims: {e}")
            raise
    
    def get_claim_status_history(self, claim_id: str) -> List[ClaimStatus]:
        """Get the status history for a claim.
        
        Args:
            claim_id: ID of the claim
            
        Returns:
            List of statuses in chronological order (oldest first)
        """
        return self.status_history.get(claim_id, [])
    
    def get_active_trackers(self) -> List[str]:
        """Get a list of claim IDs that are currently being tracked.
        
        Returns:
            List of claim IDs
        """
        return list(self.active_trackers.keys())
