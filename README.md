# Sanjeevni Plus

> An intelligent healthcare claims processing system for PM-JAY (Ayushman Bharat) that automates claim submission, tracks status, and provides real-time analytics.

Sanjeevni Plus is a comprehensive healthcare claims management platform designed to streamline the process of submitting, tracking, and managing health insurance claims under the Pradhan Mantri Jan Arogya Yojana (PM-JAY) scheme. The system leverages AI and automation to reduce manual effort, minimize errors, and accelerate claim processing times.

## 🌟 Key Features

### 📋 Document Processing
- **Smart OCR**: Extract text from scanned medical documents with high accuracy
- **Data Extraction**: Automatically extract relevant claim information from documents
- **Validation**: Cross-verify extracted data against PM-JAY guidelines
- **Document Classification**: Categorize documents into types (prescriptions, bills, lab reports, etc.)

### 🤖 Portal Automation
- **Automated Form Filling**: Auto-populate PM-JAY portal forms with extracted data
- **Bulk Processing**: Handle multiple claims simultaneously
- **Status Tracking**: Monitor claim status in real-time
- **Error Handling**: Automatic retries and notifications for failed submissions

### 📊 Analytics & Reporting
- **Claim Status Dashboard**: Visualize claim processing metrics
- **Performance Analytics**: Track approval rates and processing times
- **Hospital Performance**: Monitor provider performance metrics
- **Custom Reports**: Generate detailed reports for audits and analysis

### 🔍 Knowledge Base Management
- **PM-JAY Guidelines**: Searchable repository of policies and procedures
- **Medical Packages**: Up-to-date information on treatment packages
- **Code Lookup**: Quick reference for medical codes and procedures
- **Multilingual Support**: Content available in multiple Indian languages

### 🔄 Status Tracking & Notifications
- **Real-time Updates**: Get instant notifications on claim status changes
- **Multi-channel Alerts**: Receive updates via email, SMS, or in-app notifications
- **Status History**: Track complete claim lifecycle
- **Automated Escalations**: Automatic escalation of delayed claims

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Sanjeevni Plus System                      │
│                                                                 │
│  ┌─────────────────┐     ┌─────────────────┐     ┌───────────┐  │
│  │                 │     │                 │     │           │  │
│  │   API Layer     │◄───►│  Services Layer │◄───►│   Data    │  │
│  │  (FastAPI)      │     │  (Business      │     │  Access   │  │
│  │                 │     │   Logic)        │     │   Layer   │  │
│  └────────┬────────┘     └────────┬────────┘     └─────┬─────┘  │
│           │                       │                    │        │
│  ┌───────▼───────┐     ┌─────────▼────────┐    ┌──────▼──────┐  │
│  │               │     │                  │    │             │  │
│  │  Web Frontend │     │  Background      │    │  Database   │  │
│  │  (React)      │     │  Workers         │    │  (SQLite)   │  │
│  │               │     │                  │    │             │  │
│  └───────────────┘     └──────────────────┘    └─────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- SQLite (for development)
- Node.js 16+ (for frontend)
- Chrome/Chromium browser (for portal automation)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-organization/sanjeevni-plus.git
   cd sanjeevni-plus
   ```

2. **Set up Python virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Copy `.env.example` to `.env` and update the values:
   ```bash
   cp .env.example .env
   ```

5. **Initialize the database**
   ```bash
   python -m database.init_db
   ```

6. **Start the API server**
   ```bash
   uvicorn api.main:app --reload
   ```

7. **Start the frontend (optional)**
   ```bash
   cd frontend
   npm install
   npm start
   ```

## 🛠️ API Documentation

Once the API server is running, you can access:

- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## 📂 Project Structure

```
sanjeevni-plus/
├── api/                    # FastAPI application
│   ├── routers/            # API route handlers
│   ├── models/             # Pydantic models
│   └── main.py             # FastAPI app initialization
├── services/               # Business logic
│   ├── document_service.py # Document processing
│   ├── portal_service.py   # PM-JAY portal automation
│   ├── status_service.py   # Status tracking
│   └── notification.py     # Notification handling
├── database/               # Database models and migrations
├── models/                 # Data models
├── tests/                  # Test suite
└── frontend/               # React frontend (optional)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For support, please open an issue in the GitHub repository or contact our support team at support@sanjeevni.plus

## 📚 Documentation

For detailed documentation, please visit our [documentation site](https://docs.sanjeevni.plus).

- **Auth endpoint:** `POST /auth/token`
- **Claim submission:** `POST /claims/submit`
- **Claim status:** `GET /claims/{portal_reference}/status`

Implemented in `tests/mock_pmjay_server.py`. The workflow tests are in `tests/test_portal_workflow.py`.

### Prerequisites

- Python 3.11+
- Project virtual environment activated
- Dependencies from `requirements.txt`

> Note: The mock server manually parses `application/x-www-form-urlencoded`, so you do NOT need `python-multipart` installed.

### Environment variables

The test config sets these variables automatically inside `tests/test_portal_workflow.py`, but if you want to run services manually they are:

```
PMJAY_PORTAL_URL=http://localhost:8000
PMJAY_USERNAME=test_user
PMJAY_PASSWORD=test_password
PMJAY_CLIENT_ID=test_client_id
PMJAY_CLIENT_SECRET=test_client_secret
```

### Run the workflow tests

Run only the mock‑driven workflow tests:

```bash
python -m pytest tests/test_portal_workflow.py -v
```

You should see tests for:

- Authentication
- Claim submission (populate form + submit)
- Status tracking (poll by `portal_reference`)

### How it works

- The tests boot the mock FastAPI server in a background thread.
- `PortalAuthenticationService` posts form‑encoded credentials to `/auth/token` and stores the returned Bearer token.
- `PortalFormService` submits to `/claims/submit`, which returns a `submission_id` and `portal_reference`.
- `PortalStatusTracker` queries `/claims/{portal_reference}/status` for the latest status.

### Troubleshooting

- __Windows pytest cache warnings__: You may see `.pytest_cache` permission warnings on Windows; these are benign and do not affect the results.
- __422 on /auth/token__: Ensure the mock server version in `tests/mock_pmjay_server.py` is current; it parses form‑encoded data from the raw body.
- __Connection refused__: The mock server starts in the test fixture; if running services manually, ensure something is listening on `http://localhost:8000`.

### Related files

- `services/portal_automation_service.py` — Portal Authentication, Form Submission, Status Tracking services
- `tests/mock_pmjay_server.py` — FastAPI mock server
- `tests/test_portal_workflow.py` — End‑to‑end tests against the mock server

---

## 📚 Knowledge Base Management Service (KBMS)

The KBMS provides hot-reload knowledge base management with Indian-named collections for PM-JAY guidelines, packages, and medical codes. It supports multilingual content ingestion, vector search, and CRUD operations.

### Collections

- **`pmjay_margdarshika`** - PM-JAY guidelines and circulars
- **`ayushman_package_suchi`** - Package master data  
- **`rog_nidan_code_sangrah`** - Medical code systems (ICD-10, CPT, etc.)

### API Endpoints

#### Ingest Documents
```bash
POST /kb/ingest
Content-Type: application/json

{
  "collection": "pmjay_margdarshika",
  "id_prefix": "guide",
  "documents": [
    {
      "id": "guide_pre_auth_001",
      "text": "Pre-auth must be filed within 24 hours of admission for high-value packages.",
      "metadata": {
        "lang": "hi",
        "guideline_type": "circular",
        "applies_to": "pre_auth",
        "section": "Pre-Auth",
        "clause_ref": "Chapter-2.4",
        "state": "All-India",
        "scheme_version": "PMJAY_2025_01",
        "tags": "pre_auth, timelines, compliance"
      }
    }
  ]
}
```

#### Search Knowledge Base
```bash
POST /kb/search
Content-Type: application/json

{
  "collection": "pmjay_margdarshika",
  "q": "pre authorization timeline",
  "limit": 5,
  "where": {
    "lang": "hi",
    "applies_to": "pre_auth"
  }
}
```

#### List Sources (Pagination)
```bash
POST /kb/sources
Content-Type: application/json

{
  "collection": "pmjay_margdarshika",
  "limit": 20,
  "offset": 0,
  "where": {
    "guideline_type": "circular"
  }
}
```

#### Delete Items
```bash
DELETE /kb/item
Content-Type: application/json

{
  "collection": "pmjay_margdarshika",
  "ids": ["guide_pre_auth_001", "guide_discharge_002"]
}
```

#### Hot Reload Collections
```bash
POST /kb/reload
Content-Type: application/json

{
  "collections": ["pmjay_margdarshika", "ayushman_package_suchi", "rog_nidan_code_sangrah"]
}
```

### Metadata Schema Examples

#### Guidelines (`pmjay_margdarshika`)
```json
{
  "lang": "hi|en|mr",
  "guideline_type": "circular|sop|notification",
  "applies_to": "pre_auth|discharge|eligibility|billing",
  "section": "Pre-Auth|Discharge|Eligibility",
  "clause_ref": "Chapter-X.Y",
  "state": "All-India|Maharashtra|Karnataka",
  "scheme_version": "PMJAY_2025_01",
  "tags": "comma, separated, keywords"
}
```

#### Packages (`ayushman_package_suchi`)
```json
{
  "package_code": "HBP-001",
  "specialty": "Cardiothoracic Surgery",
  "procedure_type": "Surgical|Medical|Diagnostic",
  "package_rate": 150000,
  "pre_auth_required": true,
  "state_specific": false,
  "effective_date": "2025-01-01"
}
```

#### Medical Codes (`rog_nidan_code_sangrah`)
```json
{
  "code_system": "ICD-10|CPT|SNOMED",
  "code": "I25.10",
  "category": "Cardiovascular",
  "subcategory": "Coronary artery disease",
  "pmjay_covered": true,
  "reimbursement_rate": 5000,
  "synonyms": "CAD, Coronary atherosclerosis"
}
```

### Environment Variables

```bash
# Use Hugging Face embeddings (optional, defaults to local)
USE_HF_EMBEDDINGS=1
HF_TOKEN=your_hugging_face_token
GRANITE_EMBEDDING_MODEL=ibm-granite/granite-embedding-50m-english
```

### Testing

```bash
# Run KBMS tests
python -m pytest tests/test_kbms.py -v

# Test with sample data
python -c "
from services.knowledge_base_service import get_knowledge_base_service
kb = get_knowledge_base_service()
print(kb.hot_reload())
print(kb.get_knowledge_base_stats())
"
```

### Related Files

- `services/knowledge_base_service.py` — KBMS core service with Indian collections
- `services/chroma_service.py` — ChromaDB integration with generic helpers
- `tests/test_kbms.py` — KBMS functionality tests