"""
Notification Service

Handles sending notifications through various channels (email, SMS, in-app, etc.).
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, EmailStr, Field
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class NotificationChannel(str):
    """Supported notification channels."""
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"
    SLACK = "slack"
    WHATSAPP = "whatsapp"

class NotificationPriority(str):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class NotificationStatus(str):
    """Notification delivery status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"

class NotificationRecipient(BaseModel):
    """Recipient information for a notification."""
    id: str
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    user_id: Optional[str] = None
    preferred_channels: List[NotificationChannel] = Field(
        default_factory=lambda: [NotificationChannel.EMAIL, NotificationChannel.IN_APP]
    )

class NotificationPayload(BaseModel):
    """Payload for sending a notification."""
    subject: str
    message: str
    html_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: NotificationPriority = NotificationPriority.NORMAL
    category: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

class NotificationResult(BaseModel):
    """Result of a notification attempt."""
    notification_id: str
    status: NotificationStatus
    channel: NotificationChannel
    recipient: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NotificationService:
    """Service for sending notifications through various channels."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the notification service with configuration."""
        self.config = config or {}
        self._configure_email()
        self._configure_sms()
        self._configure_slack()
        self._configure_whatsapp()
    
    def _configure_email(self) -> None:
        """Configure email notification settings."""
        self.smtp_server = os.getenv("SMTP_SERVER", self.config.get("smtp_server", "smtp.gmail.com"))
        self.smtp_port = int(os.getenv("SMTP_PORT", self.config.get("smtp_port", 587)))
        self.smtp_username = os.getenv("SMTP_USERNAME", self.config.get("smtp_username"))
        self.smtp_password = os.getenv("SMTP_PASSWORD", self.config.get("smtp_password"))
        self.email_from = os.getenv("EMAIL_FROM", self.config.get("email_from", "noreply@sanjeevni.plus"))
        self.email_from_name = os.getenv("EMAIL_FROM_NAME", self.config.get("email_from_name", "Sanjeevni Plus"))
    
    def _configure_sms(self) -> None:
        """Configure SMS notification settings."""
        self.sms_provider = os.getenv("SMS_PROVIDER", self.config.get("sms_provider", "twilio"))
        self.sms_from = os.getenv("SMS_FROM", self.config.get("sms_from", ""))
        self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", self.config.get("twilio_account_sid"))
        self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", self.config.get("twilio_auth_token"))
    
    def _configure_slack(self) -> None:
        """Configure Slack notification settings."""
        self.slack_bot_token = os.getenv("SLACK_BOT_TOKEN", self.config.get("slack_bot_token"))
        self.slack_default_channel = os.getenv(
            "SLACK_DEFAULT_CHANNEL", 
            self.config.get("slack_default_channel", "#general")
        )
    
    def _configure_whatsapp(self) -> None:
        """Configure WhatsApp notification settings."""
        self.whatsapp_provider = os.getenv("WHATSAPP_PROVIDER", self.config.get("whatsapp_provider", "twilio"))
        self.whatsapp_from = os.getenv("WHATSAPP_FROM", self.config.get("whatsapp_from", ""))
    
    async def send_notification(
        self,
        recipient: Union[str, NotificationRecipient, Dict[str, Any]],
        subject: str,
        message: str,
        channel: Optional[NotificationChannel] = None,
        **kwargs
    ) -> NotificationResult:
        """
        Send a notification to a recipient through the specified channel.
        
        Args:
            recipient: Recipient information (can be email, phone, or NotificationRecipient object)
            subject: Notification subject
            message: Notification message content
            channel: Preferred notification channel
            **kwargs: Additional notification parameters
            
        Returns:
            NotificationResult with the result of the notification attempt
        """
        # Parse recipient
        recipient_obj = self._parse_recipient(recipient)
        
        # Determine the best channel if not specified
        if not channel and recipient_obj.preferred_channels:
            channel = recipient_obj.preferred_channels[0]
        
        if not channel:
            channel = NotificationChannel.EMAIL  # Default fallback
        
        # Create notification payload
        payload = NotificationPayload(
            subject=subject,
            message=message,
            **{k: v for k, v in kwargs.items() if k in NotificationPayload.__annotations__}
        )
        
        # Send notification through the appropriate channel
        try:
            if channel == NotificationChannel.EMAIL:
                if not recipient_obj.email:
                    raise ValueError("Email address is required for email notifications")
                result = await self._send_email(recipient_obj, payload)
                
            elif channel == NotificationChannel.SMS:
                if not recipient_obj.phone:
                    raise ValueError("Phone number is required for SMS notifications")
                result = await self._send_sms(recipient_obj, payload)
                
            elif channel == NotificationChannel.SLACK:
                result = await self._send_slack(recipient_obj, payload)
                
            elif channel == NotificationChannel.WHATSAPP:
                if not recipient_obj.phone:
                    raise ValueError("Phone number is required for WhatsApp notifications")
                result = await self._send_whatsapp(recipient_obj, payload)
                
            elif channel == NotificationChannel.IN_APP:
                result = await self._send_in_app(recipient_obj, payload)
                
            else:
                raise ValueError(f"Unsupported notification channel: {channel}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to send {channel} notification: {str(e)}")
            return NotificationResult(
                notification_id=self._generate_notification_id(),
                status=NotificationStatus.FAILED,
                channel=channel,
                recipient=recipient_obj.email or recipient_obj.phone or recipient_obj.id,
                error=str(e),
                metadata={
                    "subject": subject,
                    "channel": channel,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    def _parse_recipient(self, recipient: Union[str, NotificationRecipient, Dict[str, Any]]) -> NotificationRecipient:
        """Parse recipient information into a NotificationRecipient object."""
        if isinstance(recipient, NotificationRecipient):
            return recipient
            
        if isinstance(recipient, str):
            # If it's an email
            if "@" in recipient:
                return NotificationRecipient(
                    id=recipient,
                    name=recipient.split("@")[0],
                    email=recipient
                )
            # If it's a phone number (very basic check)
            elif recipient.replace("+", "").replace(" ", "").isdigit():
                return NotificationRecipient(
                    id=recipient,
                    name=recipient,
                    phone=recipient
                )
            # Otherwise treat as a user ID
            else:
                return NotificationRecipient(
                    id=recipient,
                    name=recipient,
                    user_id=recipient
                )
                
        elif isinstance(recipient, dict):
            return NotificationRecipient(**recipient)
            
        else:
            raise ValueError(f"Invalid recipient type: {type(recipient)}")
    
    async def _send_email(
        self, 
        recipient: NotificationRecipient, 
        payload: NotificationPayload
    ) -> NotificationResult:
        """Send an email notification."""
        if not recipient.email:
            raise ValueError("Email address is required for email notifications")
        
        notification_id = self._generate_notification_id()
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = f"{self.email_from_name} <{self.email_from}>"
            msg['To'] = recipient.email
            msg['Subject'] = payload.subject
            
            # Add body
            msg.attach(MIMEText(payload.message, 'plain'))
            
            if payload.html_message:
                msg.attach(MIMEText(payload.html_message, 'html'))
            
            # Connect to SMTP server and send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {recipient.email} with subject: {payload.subject}")
            
            return NotificationResult(
                notification_id=notification_id,
                status=NotificationStatus.SENT,
                channel=NotificationChannel.EMAIL,
                recipient=recipient.email,
                metadata={
                    "subject": payload.subject,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send email to {recipient.email}: {str(e)}")
            return NotificationResult(
                notification_id=notification_id,
                status=NotificationStatus.FAILED,
                channel=NotificationChannel.EMAIL,
                recipient=recipient.email,
                error=str(e),
                metadata={
                    "subject": payload.subject,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    async def _send_sms(
        self, 
        recipient: NotificationRecipient, 
        payload: NotificationPayload
    ) -> NotificationResult:
        """Send an SMS notification."""
        if not recipient.phone:
            raise ValueError("Phone number is required for SMS notifications")
            
        notification_id = self._generate_notification_id()
        
        try:
            # In a real implementation, this would use an SMS provider API
            # For now, we'll just log the SMS
            logger.info(
                f"SMS sent to {recipient.phone}: "
                f"{payload.subject}: {payload.message[:50]}..."
            )
            
            return NotificationResult(
                notification_id=notification_id,
                status=NotificationStatus.SENT,
                channel=NotificationChannel.SMS,
                recipient=recipient.phone,
                metadata={
                    "subject": payload.subject,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send SMS to {recipient.phone}: {str(e)}")
            return NotificationResult(
                notification_id=notification_id,
                status=NotificationStatus.FAILED,
                channel=NotificationChannel.SMS,
                recipient=recipient.phone,
                error=str(e),
                metadata={
                    "subject": payload.subject,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    async def _send_slack(
        self, 
        recipient: NotificationRecipient, 
        payload: NotificationPayload
    ) -> NotificationResult:
        """Send a Slack notification."""
        notification_id = self._generate_notification_id()
        
        try:
            # In a real implementation, this would use the Slack API
            # For now, we'll just log the Slack message
            logger.info(
                f"Slack message sent to {recipient.id or 'default channel'}: "
                f"{payload.subject}: {payload.message[:50]}..."
            )
            
            return NotificationResult(
                notification_id=notification_id,
                status=NotificationStatus.SENT,
                channel=NotificationChannel.SLACK,
                recipient=recipient.id,
                metadata={
                    "subject": payload.subject,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send Slack message: {str(e)}")
            return NotificationResult(
                notification_id=notification_id,
                status=NotificationStatus.FAILED,
                channel=NotificationChannel.SLACK,
                recipient=recipient.id,
                error=str(e),
                metadata={
                    "subject": payload.subject,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    async def _send_whatsapp(
        self, 
        recipient: NotificationRecipient, 
        payload: NotificationPayload
    ) -> NotificationResult:
        """Send a WhatsApp notification."""
        if not recipient.phone:
            raise ValueError("Phone number is required for WhatsApp notifications")
            
        notification_id = self._generate_notification_id()
        
        try:
            # In a real implementation, this would use the WhatsApp Business API
            # or a service like Twilio
            logger.info(
                f"WhatsApp message sent to {recipient.phone}: "
                f"{payload.subject}: {payload.message[:50]}..."
            )
            
            return NotificationResult(
                notification_id=notification_id,
                status=NotificationStatus.SENT,
                channel=NotificationChannel.WHATSAPP,
                recipient=recipient.phone,
                metadata={
                    "subject": payload.subject,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message to {recipient.phone}: {str(e)}")
            return NotificationResult(
                notification_id=notification_id,
                status=NotificationStatus.FAILED,
                channel=NotificationChannel.WHATSAPP,
                recipient=recipient.phone,
                error=str(e),
                metadata={
                    "subject": payload.subject,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    async def _send_in_app(
        self, 
        recipient: NotificationRecipient, 
        payload: NotificationPayload
    ) -> NotificationResult:
        """Send an in-app notification."""
        notification_id = self._generate_notification_id()
        
        try:
            # In a real implementation, this would store the notification in a database
            # and push it to the user's connected clients via WebSocket
            logger.info(
                f"In-app notification sent to user {recipient.id}: "
                f"{payload.subject}: {payload.message[:50]}..."
            )
            
            return NotificationResult(
                notification_id=notification_id,
                status=NotificationStatus.SENT,
                channel=NotificationChannel.IN_APP,
                recipient=recipient.id,
                metadata={
                    "subject": payload.subject,
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_id": recipient.user_id,
                    "category": payload.category
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send in-app notification to user {recipient.id}: {str(e)}")
            return NotificationResult(
                notification_id=notification_id,
                status=NotificationStatus.FAILED,
                channel=NotificationChannel.IN_APP,
                recipient=recipient.id,
                error=str(e),
                metadata={
                    "subject": payload.subject,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    def _generate_notification_id(self) -> str:
        """Generate a unique notification ID."""
        import uuid
        return f"notif_{uuid.uuid4().hex}"


# Global notification service instance
notification_service = NotificationService()
