<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Requirements Document

## Introduction

Sanjeevani Plus is an AI co-pilot system designed to automate healthcare claims processing workflows for Tier-2/3 hospitals in India. The system addresses critical inefficiencies including high claim denial rates (40-60%), excessive processing times (2-4 hours per claim), and complexity from unstructured multilingual medical records. The solution employs a four-stage AI orchestration approach: data ingestion using IBM Docling, intelligent reasoning with IBM Granite models via Hugging Face, portal automation via IBM WatsonX Orchestrate ADK, and comprehensive output generation with real-time tracking capabilities.

## Requirements

### Requirement 1: Multilingual Document Processing

**User Story:** As a hospital claims administrator, I want the system to automatically process multilingual medical documents (Hindi, Marathi, English), so that I can eliminate manual data entry and reduce documentation errors.

#### Acceptance Criteria

1. WHEN a medical document is uploaded to the system THEN the system SHALL automatically detect the document language with >95% accuracy using IBM Granite models via Hugging Face
2. WHEN processing typed text documents THEN the system SHALL achieve >95% OCR accuracy using IBM Docling
3. WHEN processing handwritten documents THEN the system SHALL achieve >85% OCR accuracy using IBM MAX-OCR
4. WHEN extracting medical information THEN the system SHALL structure data into standardized JSON format with patient details, procedures, and medical codes
5. IF a document contains multiple languages THEN the system SHALL process each language section independently and merge results
6. WHEN OCR confidence falls below 80% THEN the system SHALL flag the document for human verification

### Requirement 2: AI-Powered Claims Reasoning

**User Story:** As a claims processing specialist, I want the system to intelligently analyze medical data and recommend appropriate PM-JAY packages, so that I can ensure accurate claim submissions and reduce denial rates.

#### Acceptance Criteria

1. WHEN medical data is structured THEN the system SHALL perform RAG analysis against PM-JAY knowledge base using IBM Granite models from Hugging Face
2. WHEN analyzing patient eligibility THEN the system SHALL cross-reference with empanelment databases using IBM Granite Embedding via Hugging Face and return eligibility status
3. WHEN selecting treatment packages THEN the system SHALL recommend packages with confidence scores >90% using IBM Granite 3.3-8B from Hugging Face
4. WHEN risk factors are identified THEN the system SHALL generate risk scores using IBM Granite Guardian via Hugging Face and highlight potential denial reasons
5. IF multiple packages are applicable THEN the system SHALL rank them by approval probability and reimbursement value
6. WHEN package selection is complete THEN the system SHALL validate against PM-JAY guidelines and flag non-compliant elements

### Requirement 3: Government Portal Automation

**User Story:** As a hospital administrator, I want the system to automatically submit claims to government portals and track their status, so that I can reduce manual workload and ensure timely submissions.

#### Acceptance Criteria

1. WHEN a claim is ready for submission THEN the system SHALL automatically populate government portal forms using Watson Orchestrate ADK
2. WHEN submitting to PM-JAY portal THEN the system SHALL handle authentication and session management automatically
3. WHEN portal submission fails THEN the system SHALL implement exponential backoff retry mechanism with maximum 5 attempts
4. WHEN claim status updates THEN the system SHALL automatically fetch and update internal tracking system
5. IF portal requires additional documentation THEN the system SHALL alert administrators and provide specific requirements
6. WHEN submission is successful THEN the system SHALL generate confirmation receipt and store submission reference numbers

### Requirement 4: Status Tracking and Reporting APIs

**User Story:** As a hospital administrator, I want programmatic access to claim processing status and metrics, so that I can integrate with existing systems or build custom interfaces later.

#### Acceptance Criteria

1. WHEN a claim is processed THEN the system SHALL log status updates to SQLite database
2. WHEN queried via REST API THEN the system SHALL return claim metrics in JSON format
3. WHEN processing completes THEN the system SHALL generate CSV export of claim data
4. WHEN agent runs THEN the system SHALL display progress updates in chat interface

### Requirement 5: Knowledge Base Management

**User Story:** As a claims compliance manager, I want the system to stay updated with latest PM-JAY guidelines and medical codes, so that claim processing remains compliant with current regulations.

#### Acceptance Criteria

1. WHEN PM-JAY guidelines are updated THEN the system SHALL support hot-reload of knowledge base without downtime using IBM Granite Embedding from Hugging Face
2. WHEN new medical codes are released THEN the system SHALL integrate updates within 24 hours of availability
3. WHEN conflicting guidelines exist THEN the system SHALL prioritize latest official circulars and flag conflicts using IBM Granite 3.3-8B via Hugging Face
4. WHEN knowledge base changes THEN the system SHALL validate existing claims against new criteria
5. IF regulatory updates affect pending claims THEN the system SHALL automatically re-evaluate and alert administrators
6. WHEN searching knowledge base THEN vector similarity search SHALL return relevant results with <1 second response time
7. WHEN PM-JAY data is ingested THEN the system SHALL validate data format and completeness
8. WHEN knowledge base is updated THEN the system SHALL maintain version control and audit trails
9. WHEN processing claims THEN the system SHALL use latest PM-JAY package rates and guidelines via IBM Granite models from Hugging Face
10. IF data sources are unavailable THEN the system SHALL fallback to cached knowledge base
11. WHEN compliance rules change THEN the system SHALL flag affected pending claims

***

<span style="display:none"></span>[^1]

<div style="text-align: center">⁂</div>
<span style="display:none">[^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div style="text-align: center">⁂</div>

[^1]: https://www.flowforma.com/demo-library/how-to-automate-healthcare-claims-processing

[^2]: https://www.keragon.com/blog/healthcare-claims-processing-workflow

[^3]: https://www.cflowapps.com/healthcare-claims-processing-workflow-automation/

[^4]: https://www.blueprism.com/guides/claims-process-automation/

[^5]: https://www.artsyltech.com/Claims-Processing-Automation

[^6]: https://www.flowforma.com/en-gb/blog/healthcare-claims-automation

[^7]: https://www.gnani.ai/resources/blogs/ai-health-insurance-claims-complete-automation-guide-2025/

[^8]: https://automationedge.com/home-health-care-automation/blogs/rpa-healthcare-insurance-claims-processing/

[^9]: https://www.klippa.com/en/blog/information/claims-processing-automation/

