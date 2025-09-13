"""
Unit tests for REST API endpoints.

Tests for claims processing, analytics, status tracking, and portal automation APIs.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import FastAPI
import tempfile
import io

from api.main import app
from api.routers import claims, analytics, status, portal, documents
from database.repository import ClaimsRepository
from models.claims import Claim, ClaimStatus
from services.chroma_service import ChromaService


class TestClaimsAPI:
    """Test suite for Claims API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_claims_repo(self):
        """Mock claims repository."""
        return Mock(spec=ClaimsRepository)
    
    @pytest.fixture
    def sample_claim_data(self):
        """Sample claim data for testing."""
        return {
            "patient_id": "PAT001",
            "hospital_id": "HOSP001",
            "package_code": "PKG001",
            "amount": 50000.0,
            "documents": ["doc1.pdf", "doc2.pdf"]
        }
    
    def test_create_claim_success(self, client, sample_claim_data):
        """Test successful claim creation."""
        with patch('api.routers.claims.get_db') as mock_get_db:
            mock_repo = Mock()
            mock_repo.create_claim.return_value = "CLAIM001"
            mock_get_db.return_value = mock_repo
            
            response = client.post("/claims/", json=sample_claim_data)
            
            assert response.status_code == 201
            data = response.json()
            assert "claim_id" in data
            assert data["status"] == "success"
    
    def test_create_claim_validation_error(self, client):
        """Test claim creation with validation errors."""
        invalid_data = {
            "patient_id": "",  # Invalid empty string
            "hospital_id": "HOSP001",
            "amount": -1000  # Invalid negative amount
        }
        
        response = client.post("/claims/", json=invalid_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_get_claim_success(self, client):
        """Test successful claim retrieval."""
        with patch('api.routers.claims.get_db') as mock_get_db:
            mock_repo = Mock()
            mock_claim = Mock()
            mock_claim.id = "CLAIM001"
            mock_claim.status = ClaimStatus.DRAFT
            mock_claim.patient_id = "PAT001"
            mock_repo.get_claim_by_id.return_value = mock_claim
            mock_get_db.return_value = mock_repo
            
            response = client.get("/claims/CLAIM001")
            
            assert response.status_code == 200
            data = response.json()
            assert data["claim_id"] == "CLAIM001"
    
    def test_get_claim_not_found(self, client):
        """Test claim retrieval when claim doesn't exist."""
        with patch('api.routers.claims.get_db') as mock_get_db:
            mock_repo = Mock()
            mock_repo.get_claim_by_id.return_value = None
            mock_get_db.return_value = mock_repo
            
            response = client.get("/claims/NONEXISTENT")
            
            assert response.status_code == 404
    
    def test_update_claim_success(self, client):
        """Test successful claim update."""
        update_data = {"status": "processing", "notes": "Updated claim"}
        
        with patch('api.routers.claims.get_db') as mock_get_db:
            mock_repo = Mock()
            mock_repo.get_claim_by_id.return_value = Mock()
            mock_repo.update_claim.return_value = True
            mock_get_db.return_value = mock_repo
            
            response = client.put("/claims/CLAIM001", json=update_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
    
    def test_list_claims_with_filters(self, client):
        """Test listing claims with various filters."""
        with patch('api.routers.claims.get_db') as mock_get_db:
            mock_repo = Mock()
            mock_claims = [Mock(id="CLAIM001"), Mock(id="CLAIM002")]
            mock_repo.get_claims_by_hospital.return_value = mock_claims
            mock_get_db.return_value = mock_repo
            
            response = client.get("/claims/?hospital_id=HOSP001&status=draft&limit=10")
            
            assert response.status_code == 200
            data = response.json()
            assert "claims" in data
            assert len(data["claims"]) == 2


class TestAnalyticsAPI:
    """Test suite for Analytics API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_get_status_metrics(self, client):
        """Test status metrics endpoint."""
        with patch('api.routers.analytics.get_db') as mock_get_db:
            mock_repo = Mock()
            mock_get_db.return_value = mock_repo
            
            response = client.get("/analytics/status-metrics")
            
            assert response.status_code == 200
            data = response.json()
            assert "total" in data
            assert "draft" in data
            assert "approved" in data
    
    def test_get_claims_trend(self, client):
        """Test claims trend endpoint."""
        with patch('api.routers.analytics.get_db') as mock_get_db:
            mock_repo = Mock()
            mock_get_db.return_value = mock_repo
            
            response = client.get("/analytics/claims-trend?period=day&days=7")
            
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert "period" in data
            assert "total" in data
    
    def test_get_hospital_metrics(self, client):
        """Test hospital metrics endpoint."""
        with patch('api.routers.analytics.get_db') as mock_get_db:
            mock_repo = Mock()
            mock_get_db.return_value = mock_repo
            
            response = client.get("/analytics/hospital-metrics")
            
            assert response.status_code == 200
            data = response.json()
            assert "hospitals" in data
            assert "total_hospitals" in data
    
    def test_get_dashboard(self, client):
        """Test comprehensive dashboard endpoint."""
        with patch('api.routers.analytics.get_db') as mock_get_db:
            mock_repo = Mock()
            mock_get_db.return_value = mock_repo
            
            response = client.get("/analytics/dashboard?days=30")
            
            assert response.status_code == 200
            data = response.json()
            assert "status_metrics" in data
            assert "claims_trend" in data
            assert "hospital_metrics" in data
            assert "processing_times" in data
    
    def test_export_claims_csv(self, client):
        """Test CSV export for claims data."""
        with patch('api.routers.analytics.get_db') as mock_get_db:
            mock_repo = Mock()
            mock_claims = [
                Mock(
                    id="CLAIM001",
                    hospital_id="HOSP001",
                    patient_id="PAT001",
                    status="approved",
                    package_code="PKG001",
                    amount=50000,
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    error_message=""
                )
            ]
            mock_repo.get_claims_filtered.return_value = mock_claims
            mock_get_db.return_value = mock_repo
            
            response = client.get("/analytics/export/claims")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/csv; charset=utf-8"
            assert "attachment" in response.headers["content-disposition"]
            
            # Check CSV content
            csv_content = response.content.decode('utf-8')
            assert "Claim ID" in csv_content
            assert "CLAIM001" in csv_content
    
    def test_export_analytics_csv(self, client):
        """Test CSV export for analytics data."""
        with patch('api.routers.analytics.get_db') as mock_get_db:
            mock_repo = Mock()
            mock_get_db.return_value = mock_repo
            
            response = client.get("/analytics/export/analytics?days=30")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/csv; charset=utf-8"
            
            # Check CSV content
            csv_content = response.content.decode('utf-8')
            assert "STATUS METRICS" in csv_content
            assert "CLAIMS TREND" in csv_content
    
    def test_export_hospital_performance_csv(self, client):
        """Test CSV export for hospital performance data."""
        with patch('api.routers.analytics.get_db') as mock_get_db:
            mock_repo = Mock()
            mock_get_db.return_value = mock_repo
            
            response = client.get("/analytics/export/hospital-performance")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/csv; charset=utf-8"
            
            # Check CSV content
            csv_content = response.content.decode('utf-8')
            assert "Hospital ID" in csv_content
            assert "Total Claims" in csv_content


class TestStatusAPI:
    """Test suite for Status Tracking API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_check_claim_status(self, client):
        """Test claim status checking endpoint."""
        request_data = {
            "claim_ids": ["CLAIM001", "CLAIM002"],
            "force_update": False
        }
        
        with patch('api.routers.status.get_db') as mock_get_db:
            mock_repo = Mock()
            mock_claim = Mock()
            mock_claim.id = "CLAIM001"
            mock_claim.status = ClaimStatus.APPROVED
            mock_claim.last_status_check = None
            mock_claim.last_status_update = datetime.now()
            mock_repo.get_claim_by_id.return_value = mock_claim
            mock_get_db.return_value = mock_repo
            
            response = client.post("/status/check", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["claim_id"] == "CLAIM001"
    
    def test_get_status_history(self, client):
        """Test status history retrieval."""
        with patch('api.routers.status.get_db') as mock_get_db:
            mock_repo = Mock()
            mock_get_db.return_value = mock_repo
            
            response = client.get("/status/history/CLAIM001")
            
            assert response.status_code == 200
            data = response.json()
            assert "claim_id" in data
            assert "history" in data
    
    def test_subscribe_to_updates(self, client):
        """Test subscription to status updates."""
        request_data = {
            "claim_ids": ["CLAIM001", "CLAIM002"],
            "email": "test@example.com",
            "preferences": {
                "email": True,
                "sms": False,
                "in_app": True
            }
        }
        
        response = client.post("/status/subscribe", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "subscription_id" in data
    
    def test_unsubscribe_from_updates(self, client):
        """Test unsubscribing from status updates."""
        claim_ids = ["CLAIM001", "CLAIM002"]
        
        response = client.post("/status/unsubscribe", json=claim_ids)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_get_active_subscriptions(self, client):
        """Test getting active subscriptions."""
        response = client.get("/status/active-subscriptions")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestPortalAPI:
    """Test suite for Portal Automation API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_authenticate_portal(self, client):
        """Test portal authentication endpoint."""
        auth_data = {
            "username": "test_user",
            "password": "test_password",
            "portal_url": "https://pmjay.gov.in"
        }
        
        with patch('api.routers.portal.portal_service') as mock_service:
            mock_service.authenticate.return_value = {"success": True, "session_id": "sess123"}
            
            response = client.post("/portal/authenticate", json=auth_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "session_id" in data
    
    def test_submit_claim_to_portal(self, client):
        """Test claim submission to portal."""
        submit_data = {
            "claim_id": "CLAIM001",
            "portal_credentials": {
                "username": "test_user",
                "password": "test_password"
            }
        }
        
        with patch('api.routers.portal.get_db') as mock_get_db, \
             patch('api.routers.portal.portal_service') as mock_service:
            
            mock_repo = Mock()
            mock_claim = Mock(id="CLAIM001", status=ClaimStatus.PROCESSED)
            mock_repo.get_claim_by_id.return_value = mock_claim
            mock_get_db.return_value = mock_repo
            
            mock_service.submit_claim.return_value = {
                "success": True,
                "portal_reference": "PMJAY123"
            }
            
            response = client.post("/portal/submit", json=submit_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "portal_reference" in data
    
    def test_check_portal_status(self, client):
        """Test checking claim status on portal."""
        with patch('api.routers.portal.get_db') as mock_get_db, \
             patch('api.routers.portal.portal_service') as mock_service:
            
            mock_repo = Mock()
            mock_claim = Mock(portal_reference="PMJAY123")
            mock_repo.get_claim_by_id.return_value = mock_claim
            mock_get_db.return_value = mock_repo
            
            mock_service.check_status.return_value = {
                "status": "approved",
                "last_updated": datetime.now().isoformat()
            }
            
            response = client.get("/portal/status/CLAIM001")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
    
    def test_bulk_status_check(self, client):
        """Test bulk status checking."""
        request_data = {
            "claim_ids": ["CLAIM001", "CLAIM002"],
            "update_local": True
        }
        
        with patch('api.routers.portal.get_db') as mock_get_db, \
             patch('api.routers.portal.portal_service') as mock_service:
            
            mock_repo = Mock()
            mock_claims = [
                Mock(id="CLAIM001", portal_reference="PMJAY123"),
                Mock(id="CLAIM002", portal_reference="PMJAY124")
            ]
            mock_repo.get_claim_by_id.side_effect = mock_claims
            mock_get_db.return_value = mock_repo
            
            mock_service.check_status.return_value = {"status": "approved"}
            
            response = client.post("/portal/bulk-status", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 2


class TestDocumentsAPI:
    """Test suite for Documents API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_upload_document(self, client):
        """Test document upload endpoint."""
        # Create a mock file
        file_content = b"fake pdf content"
        files = {"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")}
        data = {"claim_id": "CLAIM001", "document_type": "prescription"}
        
        with patch('api.routers.documents.get_document_service') as mock_service:
            mock_service.return_value.process_document.return_value = {
                "document_id": "DOC001",
                "status": "processed"
            }
            
            response = client.post("/documents/upload", files=files, data=data)
            
            assert response.status_code == 200
            data = response.json()
            assert "document_id" in data
    
    def test_get_document_status(self, client):
        """Test document status retrieval."""
        with patch('api.routers.documents.get_document_service') as mock_service:
            mock_service.return_value.get_document_status.return_value = {
                "document_id": "DOC001",
                "status": "processed",
                "processing_time": 45.2
            }
            
            response = client.get("/documents/DOC001/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["document_id"] == "DOC001"
    
    def test_get_document_content(self, client):
        """Test document content retrieval."""
        with patch('api.routers.documents.get_document_service') as mock_service:
            mock_service.return_value.get_document_content.return_value = {
                "extracted_text": "Sample medical document text",
                "confidence": 0.95
            }
            
            response = client.get("/documents/DOC001/content")
            
            assert response.status_code == 200
            data = response.json()
            assert "extracted_text" in data
    
    def test_list_documents_by_claim(self, client):
        """Test listing documents by claim ID."""
        with patch('api.routers.documents.get_document_service') as mock_service:
            mock_service.return_value.list_documents_by_claim.return_value = [
                {"document_id": "DOC001", "type": "prescription"},
                {"document_id": "DOC002", "type": "lab_report"}
            ]
            
            response = client.get("/documents/claim/CLAIM001")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2


class TestAPIErrorHandling:
    """Test suite for API error handling."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_404_not_found(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404
    
    def test_422_validation_error(self, client):
        """Test validation error handling."""
        invalid_data = {"invalid": "data"}
        
        response = client.post("/claims/", json=invalid_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_500_internal_server_error(self, client):
        """Test internal server error handling."""
        with patch('api.routers.claims.get_db') as mock_get_db:
            mock_get_db.side_effect = Exception("Database connection failed")
            
            response = client.get("/claims/CLAIM001")
            
            assert response.status_code == 500


class TestAPIPerformance:
    """Test suite for API performance."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_response_time_claims_list(self, client):
        """Test response time for claims listing."""
        import time
        
        with patch('api.routers.claims.get_db') as mock_get_db:
            mock_repo = Mock()
            mock_repo.get_claims_by_hospital.return_value = []
            mock_get_db.return_value = mock_repo
            
            start_time = time.time()
            response = client.get("/claims/?limit=100")
            end_time = time.time()
            
            assert response.status_code == 200
            assert (end_time - start_time) < 1.0  # Should respond within 1 second
    
    def test_response_time_analytics_dashboard(self, client):
        """Test response time for analytics dashboard."""
        import time
        
        with patch('api.routers.analytics.get_db') as mock_get_db:
            mock_repo = Mock()
            mock_get_db.return_value = mock_repo
            
            start_time = time.time()
            response = client.get("/analytics/dashboard")
            end_time = time.time()
            
            assert response.status_code == 200
            assert (end_time - start_time) < 2.0  # Should respond within 2 seconds
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            with patch('api.routers.analytics.get_db') as mock_get_db:
                mock_repo = Mock()
                mock_get_db.return_value = mock_repo
                
                response = client.get("/analytics/status-metrics")
                results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__])
