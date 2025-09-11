
Sanjeevani Plus is an AI-orchestrated healthcare claims processing system designed specifically for Tier-2/3 hospitals in India. The system employs a four-stage pipeline architecture that transforms unstructured multilingual medical documents into processed government portal submissions while maintaining compliance with PM-JAY guidelines.

The design leverages IBM's enterprise AI stack including Docling for document processing, Granite models via Hugging Face for intelligent reasoning, **Chroma DB for vector storage and retrieval**, and WatsonX Orchestrate ADK for portal automation. The system addresses critical inefficiencies in Indian healthcare claims processing by reducing processing time from 2-4 hours to minutes while improving accuracy and reducing denial rates.

## Architecture

### High-Level Architecture

```

graph TB
subgraph "Input Layer"
A[Document Upload] --> B[Language Detection]
B --> C[OCR Processing]
end

    subgraph "Processing Core"
        C --> D[Data Extraction]
        D --> E[Claims Reasoning]
        E --> F[Package Recommendation]
        F --> G[Validation Engine]
    end
    
    subgraph "Knowledge Base"
        H[PM-JAY Guidelines]
        I[Medical Codes]
        J[Empanelment DB]
        K[Chroma Vector Store]
    end
    
    subgraph "Output Layer"
        G --> L[Portal Automation]
        L --> M[Status Tracking]
        M --> N[API Layer]
    end
    
    subgraph "External Systems"
        O[PM-JAY Portal]
        P[Government DBs]
    end
    
    Processing Core -.->|RAG Queries| Knowledge Base
    L --> O
    L --> P
    ```

### Component Architecture

The system follows a microservices architecture with the following core components:

1. **Document Processing Service**: Handles multilingual OCR and data extraction
2. **Claims Intelligence Service**: Performs RAG-based reasoning and package recommendation  
3. **Portal Automation Service**: Manages government portal interactions
4. **Knowledge Base Service**: Maintains PM-JAY guidelines and medical codes in Chroma DB
5. **API Gateway**: Provides REST endpoints for status tracking and integration

## Components and Interfaces

### Document Processing Service

**Technology**: IBM Docling + IBM MAX-OCR
**Purpose**: Convert unstructured multilingual documents to structured JSON

```

class DocumentProcessor:
def detect_language(self, document: bytes) -> List[str]
def extract_text_ocr(self, document: bytes, confidence_threshold: float = 0.8) -> OCRResult
def structure_medical_data(self, raw_text: str) -> MedicalRecord
def flag_for_verification(self, result: OCRResult) -> bool

```

**Interfaces**:
- Input: Medical documents (PDF, images) via REST API
- Output: Structured JSON with patient details, procedures, medical codes
- Error Handling: Confidence scoring with human verification flags

### Claims Intelligence Service  

**Technology**: IBM Granite 3.3-8B + Granite Embedding via Hugging Face + Chroma DB
**Purpose**: Analyze medical data and recommend PM-JAY packages

```

class ClaimsIntelligence:
def perform_rag_analysis(self, medical_data: MedicalRecord) -> RAGAnalysis
def check_eligibility(self, patient_data: PatientInfo) -> EligibilityResult
def recommend_packages(self, analysis: RAGAnalysis) -> List[PackageRecommendation]
def calculate_risk_score(self, claim_data: ClaimData) -> RiskAssessment
def validate_compliance(self, recommendation: PackageRecommendation) -> ComplianceResult

```

**Interfaces**:
- Input: Structured medical records from Document Processing Service
- Output: Package recommendations with confidence scores and risk assessments
- Knowledge Base Integration: Chroma vector similarity search with <1s response time

### Portal Automation Service

**Technology**: WatsonX Orchestrate ADK
**Purpose**: Automate government portal submissions with robust error handling

```

class PortalAutomation:
def authenticate_session(self, portal_type: PortalType) -> SessionToken
def populate_forms(self, claim_data: ClaimData) -> FormData
def submit_claim(self, form_data: FormData) -> SubmissionResult
def track_status(self, submission_id: str) -> ClaimStatus
def handle_retries(self, failed_submission: SubmissionResult) -> RetryResult

```

**Interfaces**:
- Input: Validated claim data from Claims Intelligence Service
- Output: Submission confirmations and status updates
- Retry Logic: Exponential backoff with maximum 5 attempts

### Knowledge Base Service

**Technology**: IBM Granite Embedding + Chroma DB
**Purpose**: Maintain up-to-date PM-JAY guidelines and medical codes

```

class KnowledgeBase:
def __init__(self):
self.chroma_client = chromadb.Client()
self.pmjay_collection = self.chroma_client.create_collection(
name="pmjay_guidelines",
embedding_function=GraniteEmbeddingFunction()
)

    def ingest_pmjay_data(self, guidelines: PMJAYGuidelines) -> IngestionResult
    def hot_reload_updates(self, updated_data: Any) -> ReloadResult  
    def vector_search(self, query: str, n_results: int = 10) -> List[SearchResult]
    def validate_data_format(self, data: Any) -> ValidationResult
    def maintain_version_control(self, changes: DataChanges) -> VersionInfo
    ```

**Chroma DB Configuration**:
```


# Chroma collection setup for PM-JAY knowledge base

chroma_client = chromadb.Client()

# Create collections for different data types

pmjay_collection = chroma_client.create_collection(
name="pmjay_guidelines",
embedding_function=sentence_transformers_ef.SentenceTransformerEmbeddingFunction(
model_name="ibm-granite/granite-embedding-english-r2"
)
)

medical_codes_collection = chroma_client.create_collection(
name="medical_codes",
embedding_function=sentence_transformers_ef.SentenceTransformerEmbeddingFunction(
model_name="ibm-granite/granite-embedding-english-r2"
)
)

```

**Interfaces**:
- Input: PM-JAY updates, medical code releases
- Output: Vector embeddings for RAG queries via Chroma DB
- Versioning: Audit trails and rollback capabilities
- Performance: Sub-second similarity search using Chroma's optimized indexing

## Data Models

### Core Data Structures

```

@dataclass
class MedicalRecord:
patient_id: str
patient_name: str
age: int
gender: str
procedures: List[MedicalProcedure]
diagnoses: List[Diagnosis]
hospital_id: str
admission_date: datetime
discharge_date: datetime
total_amount: Decimal
document_confidence: float

@dataclass
class PackageRecommendation:
package_code: str
package_name: str
confidence_score: float
estimated_amount: Decimal
approval_probability: float
risk_factors: List[str]
compliance_status: ComplianceStatus
chroma_similarity_scores: List[float]  \# Added for transparency

@dataclass
class ClaimSubmission:
claim_id: str
hospital_id: str
patient_data: MedicalRecord
recommended_package: PackageRecommendation
submission_status: SubmissionStatus
portal_reference: str
created_at: datetime
updated_at: datetime
retry_count: int

```

### Vector Database Schema (Chroma DB)

```


# PM-JAY Guidelines Collection Schema

{
"collection_name": "pmjay_guidelines",
"embedding_function": "granite-embedding-english-r2",
"documents": [
{
"id": "guideline_001",
"document": "PM-JAY package eligibility criteria...",
"metadata": {
"category": "eligibility",
"version": "2025.1",
"effective_date": "2025-01-01",
"package_codes": ["HBP-001", "HBP-002"]
}
}
]
}

# Medical Codes Collection Schema

{
"collection_name": "medical_codes",
"embedding_function": "granite-embedding-english-r2",
"documents": [
{
"id": "icd_001",
"document": "Hypertension - Essential hypertension",
"metadata": {
"code_type": "ICD-10",
"code": "I10",
"category": "cardiovascular",
"reimbursement_rate": 5000.00
}
}
]
}

```

### Database Schema (SQLite)

```

-- Claims tracking table
CREATE TABLE claims (
claim_id TEXT PRIMARY KEY,
hospital_id TEXT NOT NULL,
patient_id TEXT NOT NULL,
package_code TEXT,
status TEXT NOT NULL,
confidence_score REAL,
risk_score REAL,
portal_reference TEXT,
chroma_query_time REAL,  -- Added for performance monitoring
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chroma query logs for optimization
CREATE TABLE vector_queries (
id INTEGER PRIMARY KEY AUTOINCREMENT,
claim_id TEXT NOT NULL,
query_text TEXT NOT NULL,
collection_name TEXT NOT NULL,
similarity_scores TEXT,  -- JSON array of scores
query_time_ms REAL,
timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
FOREIGN KEY (claim_id) REFERENCES claims (claim_id)
);

```

## Error Handling

### Failure Scenarios and Recovery

**OCR Confidence Below Threshold**:
- Trigger: OCR confidence < 80%
- Response: Flag for human verification, log confidence metrics
- Recovery: Queue for manual review with highlighted problem areas

**Portal Submission Failures**:
- Trigger: Network timeouts, authentication failures, form validation errors
- Response: Exponential backoff retry (1s, 2s, 4s, 8s, 16s)
- Recovery: Alert administrators after 5 failed attempts

**Chroma DB Unavailability**:
- Trigger: Vector database connection failures, embedding service downtime
- Response: Fallback to cached embeddings and pre-computed recommendations
- Recovery: Automatic reconnection with health checks, rebuild collections if corrupted

**Package Recommendation Low Confidence**:
- Trigger: Confidence score < 90% or low Chroma similarity scores
- Response: Flag for specialist review, provide alternative options with reasoning
- Recovery: Manual override capability with audit logging

### Monitoring and Alerting

```

class ErrorHandler:
def log_ocr_failure(self, document_id: str, confidence: float)
def alert_portal_downtime(self, portal: str, duration: timedelta)
def track_denial_patterns(self, denials: List[ClaimDenial])
def monitor_processing_times(self, metrics: ProcessingMetrics)
def alert_chroma_performance(self, query_time: float, threshold: float = 1.0)
def log_vector_search_quality(self, query: str, top_similarities: List[float])

```

## Testing Strategy

### Unit Testing
- **Document Processing**: Test OCR accuracy with sample multilingual documents
- **Claims Intelligence**: Validate RAG responses against known PM-JAY scenarios  
- **Portal Automation**: Mock portal interactions with various failure scenarios
- **Chroma Integration**: Test vector search accuracy, embedding quality, and collection management

### Integration Testing
- **End-to-End Pipeline**: Process complete claim workflow from document to portal submission
- **API Endpoints**: Validate REST API responses and error handling
- **Database Operations**: Test concurrent access and data consistency
- **Chroma Performance**: Test vector search under load, collection updates, and failover scenarios

### Performance Testing
- **OCR Processing**: Target <30 seconds for standard medical documents
- **Claims Reasoning**: Target <10 seconds for package recommendations
- **Chroma Vector Search**: Ensure <1 second response time for similarity queries
- **Concurrent Load**: Test 50+ simultaneous claim processing requests

### Compliance Testing
- **Data Privacy**: Validate HIPAA-equivalent data handling
- **PM-JAY Compliance**: Test against latest government guidelines using Chroma knowledge base
- **Audit Trails**: Verify complete traceability of all claim decisions including vector search results
- **Multilingual Support**: Test Hindi, Marathi, and English processing accuracy

### Test Data Strategy
- **Synthetic Medical Records**: Generate compliant test data for various scenarios
- **Portal Simulation**: Create mock PM-JAY portal for integration testing  
- **Chroma Test Collections**: Build test vector databases with known similarity relationships
- **Edge Cases**: Test with corrupted documents, network failures, and vector database inconsistencies
```

<span style="display:none">[^1][^10][^11][^12][^13][^14][^15][^16][^17][^18][^19][^2][^20][^21][^3][^4][^5][^6][^7][^8][^9]</span>

<div style="text-align: center">⁂</div>

[^1]: Requirements-Document-3.md

[^2]: https://www.ibm.com/think/topics/vector-database

[^3]: https://www.ibm.com/think/tutorials/build-multimodal-rag-langchain-with-docling-granite

[^4]: https://skillsbuild.org/college-students/course-catalog/build-an-ai-powered-document-retrieval-system-with-ibm-granite-and-docling

[^5]: https://www.ibm.com/granite/docs/use-cases/multimodal-rag/

[^6]: https://allthingsopen.org/articles/how-to-run-and-fine-tune-ibm-granite-ai-models

[^7]: https://www.youtube.com/watch?v=hxFZLDGi9iI

[^8]: https://www.projectpro.io/article/chromadb/1044

[^9]: https://huggingface.co/ibm-granite/granite-embedding-english-r2

[^10]: https://www.ibm.com/docs/en/watsonx/saas?topic=autoai-coding-rag-experiment-chroma

[^11]: https://unstructured.io/blog/streamlining-healthcare-compliance-with-ai?modal=try-for-free

[^12]: https://www.ibm.com/think/topics/vector-search

[^13]: https://www.ibm.com/think/tutorials/agentic-rag

[^14]: https://www.lexjansen.com/phuse-us/2025/ml/PAP_ML28.pdf

[^15]: https://github.com/ibm-granite/granite-embedding-models

[^16]: https://dataplatform.cloud.ibm.com/exchange/public/entry/view/d3a5f957-a93b-46cd-82c1-c8d37d4f62c6

[^17]: https://research.aimultiple.com/open-source-vector-databases/

[^18]: https://www.ibm.com/think/topics/milvus

[^19]: https://www.ibm.com/think/tutorials/chunking-strategies-for-rag-with-langchain-watsonx-ai

[^20]: https://zeet.co/blog/exploring-chroma-vector-database-capabilities

[^21]: https://www.youtube.com/watch?v=WKmeWN9UhQY

