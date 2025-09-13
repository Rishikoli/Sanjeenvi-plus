"""
Centralized error handling and monitoring service.

Provides error handler classes for OCR, portal, and database failures
with logging and alerting capabilities.
"""

import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dataclasses import dataclass

from pydantic import BaseModel


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification."""
    OCR_ERROR = "ocr_error"
    PORTAL_ERROR = "portal_error"
    DATABASE_ERROR = "database_error"
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    PROCESSING_ERROR = "processing_error"
    SYSTEM_ERROR = "system_error"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    user_id: Optional[str] = None
    claim_id: Optional[str] = None
    document_id: Optional[str] = None
    hospital_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class ErrorRecord(BaseModel):
    """Model for error records."""
    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    resolved: bool = False
    resolution_notes: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class BaseErrorHandler:
    """Base class for all error handlers."""
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
        self.error_records: List[ErrorRecord] = []
        self.alert_handlers: List[Callable] = []
    
    def handle_error(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: Optional[ErrorContext] = None,
        custom_message: Optional[str] = None
    ) -> ErrorRecord:
        """Handle an error with logging and alerting."""
        error_id = f"{category.value}_{datetime.utcnow().timestamp()}"
        
        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=datetime.utcnow(),
            category=category,
            severity=severity,
            message=custom_message or str(error),
            details=getattr(error, 'details', None),
            stack_trace=traceback.format_exc(),
            context=context.__dict__ if context else None
        )
        
        # Log the error
        self._log_error(error_record)
        
        # Store error record
        self.error_records.append(error_record)
        
        # Send alerts if severity is high or critical
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_alerts(error_record)
        
        return error_record
    
    def _log_error(self, error_record: ErrorRecord) -> None:
        """Log error to appropriate logging level."""
        log_message = f"[{error_record.error_id}] {error_record.message}"
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, extra=error_record.dict())
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, extra=error_record.dict())
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message, extra=error_record.dict())
        else:
            self.logger.info(log_message, extra=error_record.dict())
    
    def _send_alerts(self, error_record: ErrorRecord) -> None:
        """Send alerts for high severity errors."""
        for handler in self.alert_handlers:
            try:
                handler(error_record)
            except Exception as e:
                self.logger.error(f"Failed to send alert: {e}")
    
    def add_alert_handler(self, handler: Callable[[ErrorRecord], None]) -> None:
        """Add an alert handler."""
        self.alert_handlers.append(handler)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        total_errors = len(self.error_records)
        if total_errors == 0:
            return {"total_errors": 0}
        
        by_category = {}
        by_severity = {}
        resolved_count = 0
        
        for record in self.error_records:
            # Count by category
            category = record.category.value
            by_category[category] = by_category.get(category, 0) + 1
            
            # Count by severity
            severity = record.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
            
            # Count resolved
            if record.resolved:
                resolved_count += 1
        
        return {
            "total_errors": total_errors,
            "resolved_errors": resolved_count,
            "unresolved_errors": total_errors - resolved_count,
            "by_category": by_category,
            "by_severity": by_severity,
            "resolution_rate": (resolved_count / total_errors) * 100 if total_errors > 0 else 0
        }


class OCRErrorHandler(BaseErrorHandler):
    """Error handler for OCR-related failures."""
    
    def __init__(self):
        super().__init__("ocr_error_handler")
    
    def handle_ocr_failure(
        self,
        error: Exception,
        document_path: str,
        context: Optional[ErrorContext] = None
    ) -> ErrorRecord:
        """Handle OCR processing failures."""
        severity = ErrorSeverity.MEDIUM
        
        # Determine severity based on error type
        if "timeout" in str(error).lower():
            severity = ErrorSeverity.HIGH
        elif "file not found" in str(error).lower():
            severity = ErrorSeverity.LOW
        elif "memory" in str(error).lower():
            severity = ErrorSeverity.CRITICAL
        
        custom_message = f"OCR failed for document: {document_path}"
        
        return self.handle_error(
            error=error,
            category=ErrorCategory.OCR_ERROR,
            severity=severity,
            context=context,
            custom_message=custom_message
        )
    
    def handle_confidence_low(
        self,
        confidence_score: float,
        threshold: float,
        document_path: str,
        context: Optional[ErrorContext] = None
    ) -> ErrorRecord:
        """Handle low confidence OCR results."""
        error = ValueError(f"OCR confidence {confidence_score} below threshold {threshold}")
        custom_message = f"Low OCR confidence for {document_path}: {confidence_score}"
        
        return self.handle_error(
            error=error,
            category=ErrorCategory.OCR_ERROR,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            custom_message=custom_message
        )


class PortalErrorHandler(BaseErrorHandler):
    """Error handler for portal automation failures."""
    
    def __init__(self):
        super().__init__("portal_error_handler")
        self.retry_strategies = {
            "authentication": 3,
            "submission": 5,
            "status_check": 2
        }
    
    def handle_authentication_failure(
        self,
        error: Exception,
        portal_url: str,
        username: str,
        context: Optional[ErrorContext] = None
    ) -> ErrorRecord:
        """Handle portal authentication failures."""
        custom_message = f"Portal authentication failed for {username} at {portal_url}"
        
        return self.handle_error(
            error=error,
            category=ErrorCategory.AUTHENTICATION_ERROR,
            severity=ErrorSeverity.HIGH,
            context=context,
            custom_message=custom_message
        )
    
    def handle_submission_failure(
        self,
        error: Exception,
        claim_id: str,
        portal_reference: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ) -> ErrorRecord:
        """Handle claim submission failures."""
        severity = ErrorSeverity.HIGH
        
        # Check if it's a temporary network issue
        if "timeout" in str(error).lower() or "connection" in str(error).lower():
            severity = ErrorSeverity.MEDIUM
        
        custom_message = f"Claim submission failed for {claim_id}"
        if portal_reference:
            custom_message += f" (Portal ref: {portal_reference})"
        
        return self.handle_error(
            error=error,
            category=ErrorCategory.PORTAL_ERROR,
            severity=severity,
            context=context,
            custom_message=custom_message
        )
    
    def handle_status_check_failure(
        self,
        error: Exception,
        claim_id: str,
        portal_reference: str,
        context: Optional[ErrorContext] = None
    ) -> ErrorRecord:
        """Handle status check failures."""
        custom_message = f"Status check failed for claim {claim_id} (Portal ref: {portal_reference})"
        
        return self.handle_error(
            error=error,
            category=ErrorCategory.PORTAL_ERROR,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            custom_message=custom_message
        )
    
    def should_retry(self, error_record: ErrorRecord, operation_type: str) -> bool:
        """Determine if an operation should be retried."""
        max_retries = self.retry_strategies.get(operation_type, 3)
        return error_record.retry_count < max_retries


class DatabaseErrorHandler(BaseErrorHandler):
    """Error handler for database-related failures."""
    
    def __init__(self):
        super().__init__("database_error_handler")
    
    def handle_connection_failure(
        self,
        error: Exception,
        database_url: str,
        context: Optional[ErrorContext] = None
    ) -> ErrorRecord:
        """Handle database connection failures."""
        custom_message = f"Database connection failed: {database_url}"
        
        return self.handle_error(
            error=error,
            category=ErrorCategory.DATABASE_ERROR,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            custom_message=custom_message
        )
    
    def handle_query_failure(
        self,
        error: Exception,
        query: str,
        context: Optional[ErrorContext] = None
    ) -> ErrorRecord:
        """Handle database query failures."""
        severity = ErrorSeverity.HIGH
        
        # Check for specific database errors
        error_str = str(error).lower()
        if "syntax error" in error_str:
            severity = ErrorSeverity.MEDIUM
        elif "deadlock" in error_str:
            severity = ErrorSeverity.HIGH
        elif "disk full" in error_str:
            severity = ErrorSeverity.CRITICAL
        
        custom_message = f"Database query failed: {query[:100]}..."
        
        return self.handle_error(
            error=error,
            category=ErrorCategory.DATABASE_ERROR,
            severity=severity,
            context=context,
            custom_message=custom_message
        )
    
    def handle_migration_failure(
        self,
        error: Exception,
        migration_name: str,
        context: Optional[ErrorContext] = None
    ) -> ErrorRecord:
        """Handle database migration failures."""
        custom_message = f"Database migration failed: {migration_name}"
        
        return self.handle_error(
            error=error,
            category=ErrorCategory.DATABASE_ERROR,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            custom_message=custom_message
        )


class AlertManager:
    """Manages alert notifications for errors."""
    
    def __init__(self):
        self.email_config = {
            "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "username": os.getenv("SMTP_USERNAME"),
            "password": os.getenv("SMTP_PASSWORD"),
            "from_email": os.getenv("ALERT_FROM_EMAIL", "alerts@sanjeevni.plus"),
            "to_emails": os.getenv("ALERT_TO_EMAILS", "").split(",")
        }
        self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    
    def send_email_alert(self, error_record: ErrorRecord) -> None:
        """Send email alert for error."""
        if not self.email_config["username"] or not self.email_config["to_emails"]:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config["from_email"]
            msg['To'] = ", ".join(self.email_config["to_emails"])
            msg['Subject'] = f"[{error_record.severity.value.upper()}] Sanjeevni Plus Alert: {error_record.category.value}"
            
            body = f"""
            Error Alert - Sanjeevni Plus
            
            Error ID: {error_record.error_id}
            Timestamp: {error_record.timestamp}
            Category: {error_record.category.value}
            Severity: {error_record.severity.value}
            
            Message: {error_record.message}
            
            Details: {error_record.details or 'N/A'}
            
            Context: {json.dumps(error_record.context, indent=2) if error_record.context else 'N/A'}
            
            Stack Trace:
            {error_record.stack_trace or 'N/A'}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"]) as server:
                server.starttls()
                server.login(self.email_config["username"], self.email_config["password"])
                server.send_message(msg)
                
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
    
    def send_slack_alert(self, error_record: ErrorRecord) -> None:
        """Send Slack alert for error."""
        if not self.slack_webhook:
            return
        
        try:
            import requests
            
            color = {
                ErrorSeverity.LOW: "#36a64f",
                ErrorSeverity.MEDIUM: "#ff9500",
                ErrorSeverity.HIGH: "#ff0000",
                ErrorSeverity.CRITICAL: "#8b0000"
            }.get(error_record.severity, "#36a64f")
            
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"Sanjeevni Plus Alert: {error_record.category.value}",
                        "fields": [
                            {"title": "Error ID", "value": error_record.error_id, "short": True},
                            {"title": "Severity", "value": error_record.severity.value.upper(), "short": True},
                            {"title": "Message", "value": error_record.message, "short": False},
                            {"title": "Timestamp", "value": str(error_record.timestamp), "short": True}
                        ]
                    }
                ]
            }
            
            requests.post(self.slack_webhook, json=payload)
            
        except Exception as e:
            logging.error(f"Failed to send Slack alert: {e}")


class ErrorHandlerFactory:
    """Factory for creating error handlers."""
    
    _handlers = {}
    _alert_manager = AlertManager()
    
    @classmethod
    def get_ocr_handler(cls) -> OCRErrorHandler:
        """Get OCR error handler instance."""
        if "ocr" not in cls._handlers:
            handler = OCRErrorHandler()
            handler.add_alert_handler(cls._alert_manager.send_email_alert)
            handler.add_alert_handler(cls._alert_manager.send_slack_alert)
            cls._handlers["ocr"] = handler
        return cls._handlers["ocr"]
    
    @classmethod
    def get_portal_handler(cls) -> PortalErrorHandler:
        """Get portal error handler instance."""
        if "portal" not in cls._handlers:
            handler = PortalErrorHandler()
            handler.add_alert_handler(cls._alert_manager.send_email_alert)
            handler.add_alert_handler(cls._alert_manager.send_slack_alert)
            cls._handlers["portal"] = handler
        return cls._handlers["portal"]
    
    @classmethod
    def get_database_handler(cls) -> DatabaseErrorHandler:
        """Get database error handler instance."""
        if "database" not in cls._handlers:
            handler = DatabaseErrorHandler()
            handler.add_alert_handler(cls._alert_manager.send_email_alert)
            handler.add_alert_handler(cls._alert_manager.send_slack_alert)
            cls._handlers["database"] = handler
        return cls._handlers["database"]
    
    @classmethod
    def get_all_error_stats(cls) -> Dict[str, Any]:
        """Get error statistics from all handlers."""
        stats = {}
        for handler_name, handler in cls._handlers.items():
            stats[handler_name] = handler.get_error_stats()
        return stats


# Global error handlers
ocr_error_handler = ErrorHandlerFactory.get_ocr_handler()
portal_error_handler = ErrorHandlerFactory.get_portal_handler()
database_error_handler = ErrorHandlerFactory.get_database_handler()
