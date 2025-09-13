"""
Monitoring and logging service for system health and performance tracking.

Provides comprehensive monitoring capabilities including metrics collection,
health checks, and performance monitoring.
"""

import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import json
import os
from pathlib import Path

from pydantic import BaseModel


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: datetime
    active_claims: int
    processed_claims_today: int
    failed_claims_today: int
    average_processing_time: float
    ocr_success_rate: float
    portal_success_rate: float
    api_requests_per_minute: int
    database_connections: int
    cache_hit_rate: float


class PerformanceTracker:
    """Tracks performance metrics for operations."""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.operation_errors: Dict[str, int] = defaultdict(int)
    
    def record_operation(self, operation_name: str, duration: float, success: bool = True):
        """Record an operation's performance."""
        self.operation_times[operation_name].append(duration)
        self.operation_counts[operation_name] += 1
        
        if not success:
            self.operation_errors[operation_name] += 1
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        times = list(self.operation_times[operation_name])
        if not times:
            return {
                "operation": operation_name,
                "count": 0,
                "avg_time": 0,
                "min_time": 0,
                "max_time": 0,
                "success_rate": 100.0
            }
        
        total_count = self.operation_counts[operation_name]
        error_count = self.operation_errors[operation_name]
        success_rate = ((total_count - error_count) / total_count * 100) if total_count > 0 else 100.0
        
        return {
            "operation": operation_name,
            "count": total_count,
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "success_rate": success_rate,
            "error_count": error_count
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all operations."""
        return {op: self.get_operation_stats(op) for op in self.operation_times.keys()}


class HealthChecker:
    """Performs health checks on system components."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self.last_check_results: Dict[str, Dict[str, Any]] = {}
    
    def register_health_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """Register a health check function."""
        self.health_checks[name] = check_func
    
    def run_health_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.health_checks:
            return {"status": "error", "message": f"Health check '{name}' not found"}
        
        try:
            result = self.health_checks[name]()
            result["timestamp"] = datetime.utcnow().isoformat()
            result["check_name"] = name
            self.last_check_results[name] = result
            return result
        except Exception as e:
            error_result = {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "check_name": name
            }
            self.last_check_results[name] = error_result
            return error_result
    
    def run_all_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks."""
        results = {}
        for name in self.health_checks.keys():
            results[name] = self.run_health_check(name)
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        results = self.run_all_health_checks()
        
        total_checks = len(results)
        healthy_checks = sum(1 for r in results.values() if r.get("status") == "healthy")
        warning_checks = sum(1 for r in results.values() if r.get("status") == "warning")
        error_checks = sum(1 for r in results.values() if r.get("status") == "error")
        
        if error_checks > 0:
            overall_status = "unhealthy"
        elif warning_checks > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return {
            "overall_status": overall_status,
            "total_checks": total_checks,
            "healthy_checks": healthy_checks,
            "warning_checks": warning_checks,
            "error_checks": error_checks,
            "timestamp": datetime.utcnow().isoformat(),
            "details": results
        }


class MonitoringService:
    """Main monitoring service that coordinates all monitoring activities."""
    
    def __init__(self, metrics_retention_hours: int = 24):
        self.metrics_retention_hours = metrics_retention_hours
        self.system_metrics: deque = deque(maxlen=metrics_retention_hours * 60)  # 1 per minute
        self.app_metrics: deque = deque(maxlen=metrics_retention_hours * 60)
        self.performance_tracker = PerformanceTracker()
        self.health_checker = HealthChecker()
        self.logger = logging.getLogger("monitoring_service")
        
        # Monitoring thread control
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _register_default_health_checks(self):
        """Register default health checks."""
        self.health_checker.register_health_check("database", self._check_database_health)
        self.health_checker.register_health_check("disk_space", self._check_disk_space)
        self.health_checker.register_health_check("memory", self._check_memory_usage)
        self.health_checker.register_health_check("api_endpoints", self._check_api_endpoints)
    
    def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            # In a real implementation, this would test database connection
            # For now, we'll simulate a health check
            response_time = 0.05  # Simulated response time
            
            if response_time > 1.0:
                return {"status": "error", "message": "Database response time too high", "response_time": response_time}
            elif response_time > 0.5:
                return {"status": "warning", "message": "Database response time elevated", "response_time": response_time}
            else:
                return {"status": "healthy", "message": "Database responding normally", "response_time": response_time}
                
        except Exception as e:
            return {"status": "error", "message": f"Database connection failed: {str(e)}"}
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            disk_usage = psutil.disk_usage('/')
            free_percent = (disk_usage.free / disk_usage.total) * 100
            
            if free_percent < 5:
                return {"status": "error", "message": "Critical disk space", "free_percent": free_percent}
            elif free_percent < 15:
                return {"status": "warning", "message": "Low disk space", "free_percent": free_percent}
            else:
                return {"status": "healthy", "message": "Sufficient disk space", "free_percent": free_percent}
                
        except Exception as e:
            return {"status": "error", "message": f"Disk check failed: {str(e)}"}
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                return {"status": "error", "message": "Critical memory usage", "memory_percent": memory.percent}
            elif memory.percent > 80:
                return {"status": "warning", "message": "High memory usage", "memory_percent": memory.percent}
            else:
                return {"status": "healthy", "message": "Normal memory usage", "memory_percent": memory.percent}
                
        except Exception as e:
            return {"status": "error", "message": f"Memory check failed: {str(e)}"}
    
    def _check_api_endpoints(self) -> Dict[str, Any]:
        """Check API endpoint health."""
        try:
            # In a real implementation, this would test API endpoints
            # For now, we'll simulate based on recent performance data
            api_stats = self.performance_tracker.get_operation_stats("api_request")
            
            if api_stats["count"] == 0:
                return {"status": "warning", "message": "No recent API activity"}
            
            if api_stats["success_rate"] < 95:
                return {"status": "error", "message": "Low API success rate", "success_rate": api_stats["success_rate"]}
            elif api_stats["avg_time"] > 2.0:
                return {"status": "warning", "message": "High API response time", "avg_time": api_stats["avg_time"]}
            else:
                return {"status": "healthy", "message": "API endpoints healthy", "success_rate": api_stats["success_rate"]}
                
        except Exception as e:
            return {"status": "error", "message": f"API check failed: {str(e)}"}
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Network connections
            connections = len(psutil.net_connections())
            
            metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_percent=(disk.used / disk.total) * 100,
                disk_used_gb=disk.used / (1024 * 1024 * 1024),
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                active_connections=connections
            )
            
            self.system_metrics.append(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            raise
    
    def collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics."""
        try:
            # In a real implementation, these would query actual application state
            # For now, we'll use simulated data based on performance tracker
            
            ocr_stats = self.performance_tracker.get_operation_stats("ocr_processing")
            portal_stats = self.performance_tracker.get_operation_stats("portal_submission")
            api_stats = self.performance_tracker.get_operation_stats("api_request")
            
            metrics = ApplicationMetrics(
                timestamp=datetime.utcnow(),
                active_claims=50,  # Simulated
                processed_claims_today=120,  # Simulated
                failed_claims_today=5,  # Simulated
                average_processing_time=ocr_stats.get("avg_time", 0),
                ocr_success_rate=ocr_stats.get("success_rate", 100.0),
                portal_success_rate=portal_stats.get("success_rate", 100.0),
                api_requests_per_minute=api_stats.get("count", 0),
                database_connections=10,  # Simulated
                cache_hit_rate=85.5  # Simulated
            )
            
            self.app_metrics.append(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect application metrics: {e}")
            raise
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Collect metrics
                    self.collect_system_metrics()
                    self.collect_application_metrics()
                    
                    # Run health checks every 5 minutes
                    if len(self.system_metrics) % 5 == 0:
                        health_results = self.health_checker.run_all_health_checks()
                        self.logger.info(f"Health check results: {health_results}")
                    
                    # Clean up old metrics
                    self._cleanup_old_metrics()
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                
                time.sleep(interval_seconds)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Monitoring service started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Monitoring service stopped")
    
    def _cleanup_old_metrics(self):
        """Clean up metrics older than retention period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.metrics_retention_hours)
        
        # Clean system metrics
        while self.system_metrics and self.system_metrics[0].timestamp < cutoff_time:
            self.system_metrics.popleft()
        
        # Clean application metrics
        while self.app_metrics and self.app_metrics[0].timestamp < cutoff_time:
            self.app_metrics.popleft()
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Filter recent metrics
        recent_system = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
        recent_app = [m for m in self.app_metrics if m.timestamp >= cutoff_time]
        
        summary = {
            "time_period_hours": hours,
            "system_metrics": {
                "sample_count": len(recent_system),
                "avg_cpu_percent": sum(m.cpu_percent for m in recent_system) / len(recent_system) if recent_system else 0,
                "avg_memory_percent": sum(m.memory_percent for m in recent_system) / len(recent_system) if recent_system else 0,
                "avg_disk_percent": sum(m.disk_percent for m in recent_system) / len(recent_system) if recent_system else 0,
            },
            "application_metrics": {
                "sample_count": len(recent_app),
                "avg_processing_time": sum(m.average_processing_time for m in recent_app) / len(recent_app) if recent_app else 0,
                "avg_ocr_success_rate": sum(m.ocr_success_rate for m in recent_app) / len(recent_app) if recent_app else 0,
                "avg_portal_success_rate": sum(m.portal_success_rate for m in recent_app) / len(recent_app) if recent_app else 0,
            },
            "performance_stats": self.performance_tracker.get_all_stats(),
            "health_status": self.health_checker.get_overall_health()
        }
        
        return summary
    
    def export_metrics(self, filepath: str, hours: int = 24) -> None:
        """Export metrics to a JSON file."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Filter metrics
            recent_system = [asdict(m) for m in self.system_metrics if m.timestamp >= cutoff_time]
            recent_app = [asdict(m) for m in self.app_metrics if m.timestamp >= cutoff_time]
            
            # Convert datetime objects to strings
            for metrics_list in [recent_system, recent_app]:
                for metric in metrics_list:
                    if 'timestamp' in metric:
                        metric['timestamp'] = metric['timestamp'].isoformat()
            
            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "time_period_hours": hours,
                "system_metrics": recent_system,
                "application_metrics": recent_app,
                "performance_stats": self.performance_tracker.get_all_stats(),
                "health_status": self.health_checker.get_overall_health()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            raise


# Global monitoring service instance
monitoring_service = MonitoringService()


# Context manager for performance tracking
class PerformanceContext:
    """Context manager for tracking operation performance."""
    
    def __init__(self, operation_name: str, tracker: PerformanceTracker = None):
        self.operation_name = operation_name
        self.tracker = tracker or monitoring_service.performance_tracker
        self.start_time = None
        self.success = True
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.success = exc_type is None
            self.tracker.record_operation(self.operation_name, duration, self.success)
    
    def mark_error(self):
        """Mark the operation as failed."""
        self.success = False


# Decorator for automatic performance tracking
def track_performance(operation_name: str):
    """Decorator to automatically track function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceContext(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
