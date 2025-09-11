"""Knowledge base service for PM-JAY data ingestion and management."""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .chroma_service import ChromaService

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """Service for managing PM-JAY knowledge base data ingestion."""
    
    def __init__(self, chroma_service: ChromaService):
        self.chroma_service = chroma_service
    
    def ingest_pmjay_packages_data(self, packages_data: List[Dict[str, Any]]) -> bool:
        """Ingest PM-JAY package data into ChromaDB."""
        try:
            documents = []
            metadatas = []
            ids = []
            
            for package in packages_data:
                # Create document text from package information
                doc_text = self._create_package_document(package)
                documents.append(doc_text)
                
                # Create metadata
                metadata = {
                    "category": "package",
                    "package_code": package.get("package_code", ""),
                    "package_name": package.get("package_name", ""),
                    "specialty": package.get("specialty", ""),
                    "procedure_type": package.get("procedure_type", ""),
                    "package_rate": package.get("package_rate", 0),
                    "version": "2025.1",
                    "effective_date": package.get("effective_date", "")
                }
                metadatas.append(metadata)
                
                # Create unique ID
                package_id = f"pkg_{package.get('package_code', f'unknown_{len(ids)}')}"
                ids.append(package_id)
            
            # Add to ChromaDB
            self.chroma_service.add_pmjay_documents(documents, metadatas, ids)
            
            logger.info(f"Successfully ingested {len(packages_data)} PM-JAY packages")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest PM-JAY packages: {e}")
            return False
    
    def ingest_pmjay_guidelines_data(self, guidelines_data: List[Dict[str, Any]]) -> bool:
        """Ingest PM-JAY guidelines and compliance rules."""
        try:
            documents = []
            metadatas = []
            ids = []
            
            for guideline in guidelines_data:
                # Create document text from guideline
                doc_text = self._create_guideline_document(guideline)
                documents.append(doc_text)
                
                # Create metadata
                metadata = {
                    "category": "guideline",
                    "guideline_type": guideline.get("type", ""),
                    "section": guideline.get("section", ""),
                    "applicable_packages": guideline.get("applicable_packages", []),
                    "version": "2025.1",
                    "effective_date": guideline.get("effective_date", ""),
                    "priority": guideline.get("priority", "medium")
                }
                metadatas.append(metadata)
                
                # Create unique ID
                guideline_id = f"guide_{guideline.get('id', f'unknown_{len(ids)}')}"
                ids.append(guideline_id)
            
            # Add to ChromaDB
            self.chroma_service.add_pmjay_documents(documents, metadatas, ids)
            
            logger.info(f"Successfully ingested {len(guidelines_data)} PM-JAY guidelines")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest PM-JAY guidelines: {e}")
            return False
    
    def ingest_medical_codes_data(self, codes_data: List[Dict[str, Any]]) -> bool:
        """Ingest medical codes (ICD-10, CPT, etc.) into ChromaDB."""
        try:
            documents = []
            metadatas = []
            ids = []
            
            for code_entry in codes_data:
                # Create document text from medical code
                doc_text = self._create_medical_code_document(code_entry)
                documents.append(doc_text)
                
                # Create metadata
                metadata = {
                    "code_type": code_entry.get("code_type", ""),
                    "code": code_entry.get("code", ""),
                    "category": code_entry.get("category", ""),
                    "subcategory": code_entry.get("subcategory", ""),
                    "reimbursement_rate": code_entry.get("reimbursement_rate", 0),
                    "pmjay_covered": code_entry.get("pmjay_covered", False),
                    "version": "2025.1"
                }
                metadatas.append(metadata)
                
                # Create unique ID
                code_id = f"{code_entry.get('code_type', 'code')}_{code_entry.get('code', f'unknown_{len(ids)}')}"
                ids.append(code_id)
            
            # Add to ChromaDB
            self.chroma_service.add_medical_codes(documents, metadatas, ids)
            
            logger.info(f"Successfully ingested {len(codes_data)} medical codes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest medical codes: {e}")
            return False
    
    def _create_package_document(self, package: Dict[str, Any]) -> str:
        """Create a searchable document from package data."""
        doc_parts = []
        
        # Package basic info
        if package.get("package_code"):
            doc_parts.append(f"Package Code: {package['package_code']}")
        
        if package.get("package_name"):
            doc_parts.append(f"Package Name: {package['package_name']}")
        
        if package.get("description"):
            doc_parts.append(f"Description: {package['description']}")
        
        # Clinical information
        if package.get("specialty"):
            doc_parts.append(f"Medical Specialty: {package['specialty']}")
        
        if package.get("procedure_type"):
            doc_parts.append(f"Procedure Type: {package['procedure_type']}")
        
        if package.get("indications"):
            doc_parts.append(f"Medical Indications: {', '.join(package['indications'])}")
        
        # Financial information
        if package.get("package_rate"):
            doc_parts.append(f"Package Rate: ₹{package['package_rate']}")
        
        # Requirements and criteria
        if package.get("eligibility_criteria"):
            doc_parts.append(f"Eligibility Criteria: {package['eligibility_criteria']}")
        
        if package.get("pre_authorization_required"):
            doc_parts.append(f"Pre-authorization Required: {package['pre_authorization_required']}")
        
        if package.get("documentation_required"):
            doc_parts.append(f"Required Documentation: {', '.join(package['documentation_required'])}")
        
        return " | ".join(doc_parts)
    
    def _create_guideline_document(self, guideline: Dict[str, Any]) -> str:
        """Create a searchable document from guideline data."""
        doc_parts = []
        
        if guideline.get("title"):
            doc_parts.append(f"Title: {guideline['title']}")
        
        if guideline.get("content"):
            doc_parts.append(f"Content: {guideline['content']}")
        
        if guideline.get("type"):
            doc_parts.append(f"Guideline Type: {guideline['type']}")
        
        if guideline.get("section"):
            doc_parts.append(f"Section: {guideline['section']}")
        
        if guideline.get("applicable_conditions"):
            doc_parts.append(f"Applicable Conditions: {', '.join(guideline['applicable_conditions'])}")
        
        if guideline.get("compliance_requirements"):
            doc_parts.append(f"Compliance Requirements: {guideline['compliance_requirements']}")
        
        if guideline.get("exceptions"):
            doc_parts.append(f"Exceptions: {guideline['exceptions']}")
        
        return " | ".join(doc_parts)
    
    def _create_medical_code_document(self, code_entry: Dict[str, Any]) -> str:
        """Create a searchable document from medical code data."""
        doc_parts = []
        
        if code_entry.get("code_type") and code_entry.get("code"):
            doc_parts.append(f"{code_entry['code_type']} Code: {code_entry['code']}")
        
        if code_entry.get("description"):
            doc_parts.append(f"Description: {code_entry['description']}")
        
        if code_entry.get("category"):
            doc_parts.append(f"Category: {code_entry['category']}")
        
        if code_entry.get("subcategory"):
            doc_parts.append(f"Subcategory: {code_entry['subcategory']}")
        
        if code_entry.get("synonyms"):
            doc_parts.append(f"Synonyms: {', '.join(code_entry['synonyms'])}")
        
        if code_entry.get("related_procedures"):
            doc_parts.append(f"Related Procedures: {', '.join(code_entry['related_procedures'])}")
        
        if code_entry.get("pmjay_covered"):
            doc_parts.append("PM-JAY Covered: Yes")
        
        if code_entry.get("reimbursement_rate"):
            doc_parts.append(f"Reimbursement Rate: ₹{code_entry['reimbursement_rate']}")
        
        return " | ".join(doc_parts)
    
    def load_sample_data(self) -> bool:
        """Load sample PM-JAY data for testing and development."""
        try:
            # Sample PM-JAY packages
            sample_packages = [
                {
                    "package_code": "HBP-001",
                    "package_name": "Coronary Artery Bypass Graft Surgery",
                    "description": "Surgical procedure to improve blood flow to the heart",
                    "specialty": "Cardiothoracic Surgery",
                    "procedure_type": "Surgical",
                    "package_rate": 150000,
                    "indications": ["Coronary artery disease", "Multiple vessel disease"],
                    "eligibility_criteria": "Age 18-70, documented CAD, surgical fitness",
                    "pre_authorization_required": True,
                    "documentation_required": ["ECG", "Echo", "Angiography", "Blood tests"],
                    "effective_date": "2025-01-01"
                },
                {
                    "package_code": "HBP-002",
                    "package_name": "Percutaneous Coronary Intervention",
                    "description": "Minimally invasive procedure to open blocked coronary arteries",
                    "specialty": "Interventional Cardiology",
                    "procedure_type": "Interventional",
                    "package_rate": 75000,
                    "indications": ["Acute coronary syndrome", "Stable angina"],
                    "eligibility_criteria": "Documented coronary stenosis >70%",
                    "pre_authorization_required": False,
                    "documentation_required": ["ECG", "Angiography", "Troponin levels"],
                    "effective_date": "2025-01-01"
                }
            ]
            
            # Sample guidelines
            sample_guidelines = [
                {
                    "id": "GUIDE_001",
                    "title": "Pre-authorization Requirements",
                    "content": "All high-value procedures require pre-authorization within 48 hours of admission",
                    "type": "procedural",
                    "section": "authorization",
                    "applicable_conditions": ["Cardiac surgery", "Neurosurgery", "Oncology"],
                    "compliance_requirements": "Submit complete documentation including medical history",
                    "priority": "high",
                    "effective_date": "2025-01-01"
                },
                {
                    "id": "GUIDE_002",
                    "title": "Eligibility Verification",
                    "content": "Patient eligibility must be verified against empanelment database",
                    "type": "eligibility",
                    "section": "verification",
                    "applicable_conditions": ["All procedures"],
                    "compliance_requirements": "Real-time verification required",
                    "priority": "high",
                    "effective_date": "2025-01-01"
                }
            ]
            
            # Sample medical codes
            sample_codes = [
                {
                    "code_type": "ICD-10",
                    "code": "I25.10",
                    "description": "Atherosclerotic heart disease of native coronary artery without angina pectoris",
                    "category": "Cardiovascular",
                    "subcategory": "Coronary artery disease",
                    "pmjay_covered": True,
                    "reimbursement_rate": 5000,
                    "synonyms": ["CAD", "Coronary atherosclerosis"],
                    "related_procedures": ["Angiography", "PCI", "CABG"]
                },
                {
                    "code_type": "ICD-10",
                    "code": "I21.9",
                    "description": "Acute myocardial infarction, unspecified",
                    "category": "Cardiovascular",
                    "subcategory": "Acute coronary syndrome",
                    "pmjay_covered": True,
                    "reimbursement_rate": 15000,
                    "synonyms": ["Heart attack", "MI", "AMI"],
                    "related_procedures": ["Primary PCI", "Thrombolysis"]
                }
            ]
            
            # Ingest all sample data
            packages_success = self.ingest_pmjay_packages_data(sample_packages)
            guidelines_success = self.ingest_pmjay_guidelines_data(sample_guidelines)
            codes_success = self.ingest_medical_codes_data(sample_codes)
            
            if packages_success and guidelines_success and codes_success:
                logger.info("Successfully loaded all sample data")
                return True
            else:
                logger.error("Failed to load some sample data")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load sample data: {e}")
            return False
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            stats = {}
            
            # Get PM-JAY collection stats
            try:
                pmjay_info = self.chroma_service.get_collection_info("pmjay_guidelines")
                stats["pmjay_documents"] = pmjay_info["count"]
            except:
                stats["pmjay_documents"] = 0
            
            # Get medical codes collection stats
            try:
                codes_info = self.chroma_service.get_collection_info("medical_codes")
                stats["medical_codes"] = codes_info["count"]
            except:
                stats["medical_codes"] = 0
            
            stats["total_documents"] = stats["pmjay_documents"] + stats["medical_codes"]
            stats["status"] = "healthy" if stats["total_documents"] > 0 else "empty"
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get knowledge base stats: {e}")
            return {"status": "error", "error": str(e)}


# Global knowledge base service instance
def get_knowledge_base_service() -> KnowledgeBaseService:
    """Get knowledge base service instance."""
    from .chroma_service import chroma_service
    return KnowledgeBaseService(chroma_service)
