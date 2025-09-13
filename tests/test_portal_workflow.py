import asyncio
import os
import pytest
import uvicorn
import threading
from datetime import datetime
from dataclasses import asdict
from decimal import Decimal
from models.claims import ComplianceStatus

from models.medical import MedicalRecord, PatientInfo, Diagnosis, MedicalProcedure
from models.claims import PackageRecommendation, ComplianceStatus
from services.portal_automation_service import (
    PortalAuthenticationService, 
    PortalFormService, 
    PortalStatusTracker,
    PortalStatus,
    SubmissionResult
)

# Import the mock server
from tests.mock_pmjay_server import app

# Test configuration
TEST_CONFIG = {
    "PMJAY_PORTAL_URL": "http://localhost:8000",  # Mock server URL (without /api)
    "PMJAY_USERNAME": "test_user",
    "PMJAY_PASSWORD": "test_password",
    "PMJAY_CLIENT_ID": "test_client_id",
    "PMJAY_CLIENT_SECRET": "test_client_secret"
}

# Set environment variables for the test
for key, value in TEST_CONFIG.items():
    os.environ[key] = value

# Start the mock server in a separate thread
@pytest.fixture(scope="module")
def mock_server():
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run)
    thread.daemon = True
    thread.start()
    
    # Wait for server to start
    import time
    time.sleep(1)
    
    yield
    
    # Cleanup
    if hasattr(server, 'should_exit'):
        server.should_exit = True

class TestPortalWorkflow:
    @pytest.fixture(autouse=True)
    def setup(self, mock_server):
        self.auth_service = PortalAuthenticationService()
        self.form_service = PortalFormService(auth_service=self.auth_service)

    @pytest.mark.asyncio
    async def test_authentication(self):
        """Test successful authentication with the mock server"""
        result = await self.auth_service.authenticate()
        assert result.success is True
        assert result.session_token is not None
        assert result.expires_at is not None

    @pytest.mark.asyncio
    async def test_claim_submission(self):
        """Test submitting a claim to the mock server"""
        # Create medical record using proper model
        medical_record = MedicalRecord(
            patient_info=PatientInfo(
                patient_id="PAT123",
                name="Test Patient",
                age=35,
                gender="M"
            ),
            hospital_id="HOSP456",
            admission_date=datetime.now(),
            discharge_date=datetime.now(),
            procedures=[
                MedicalProcedure(
                    procedure_code="P001",
                    procedure_name="Chest X-ray",
                    procedure_date=datetime.now()
                )
            ],
            diagnoses=[
                Diagnosis(
                    diagnosis_code="J18.9",
                    diagnosis_name="Pneumonia, unspecified",
                    diagnosis_date=datetime.now()
                )
            ],
            total_amount=Decimal('10000.0'),
            document_confidence=0.95
        )
        
        # Create package recommendation as dict to match expected type
        package_recommendation = {
            "package_code": "PKG001",
            "package_name": "Pneumonia Treatment Package",
            "confidence": 0.95
        }

        # Populate form data
        form_data = await self.form_service.populate_claim_form(medical_record, package_recommendation)
        
        # Submit claim form
        claim_id = "TEST_CLAIM_123"
        result = await self.form_service.submit_claim_form(form_data, claim_id)
        
        # Verify results
        assert result.success is True
        assert result.submission_id is not None
        assert result.status in [PortalStatus.FORM_SUBMITTED, PortalStatus.STATUS_UPDATED]

    @pytest.mark.asyncio
    async def test_claim_status_check(self):
        """Test checking claim status"""
        # First submit a claim to get a valid portal_reference
        medical_record = MedicalRecord(
            patient_info=PatientInfo(
                patient_id="PAT999",
                name="Status Test",
                age=30,
                gender="M"
            ),
            hospital_id="HOSP456",
            admission_date=datetime.now(),
            discharge_date=datetime.now(),
            procedures=[],
            diagnoses=[
                Diagnosis(
                    diagnosis_code="Z00.0",
                    diagnosis_name="General medical examination",
                    diagnosis_date=datetime.now()
                )
            ],
            total_amount=Decimal('1000.00'),
            document_confidence=0.9
        )

        package_recommendation = {
            "package_code": "PKGSTATUS",
            "package_name": "Status Check Package",
            "confidence": 0.9
        }

        form_data = await self.form_service.populate_claim_form(medical_record, package_recommendation)
        submit_result = await self.form_service.submit_claim_form(form_data, "STATUS_TEST_1")
        assert submit_result.success is True
        assert submit_result.portal_reference is not None

        # Now check status using the real portal reference
        status_tracker = PortalStatusTracker(self.auth_service)
        status_result = await status_tracker.track_claim_status(submit_result.portal_reference)

        # Verify the status result has expected fields
        assert hasattr(status_result, 'claim_status')
        assert hasattr(status_result, 'last_updated')
        assert hasattr(status_result, 'portal_messages')
        assert status_result.last_updated <= datetime.now()

if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main(["-v", "-s", __file__]))
