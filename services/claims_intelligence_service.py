"""
Claims Intelligence Service

This service provides intelligent claims processing by combining:
- Document processing and OCR
- Vector search for relevant guidelines
- AI-powered analysis and recommendations
- Risk assessment and compliance checking
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from decimal import Decimal
import json

from models.medical import MedicalRecord, PatientInfo, MedicalProcedure, Diagnosis
from models.claims import (
    PackageRecommendation, 
    RiskAssessment, 
    ClaimSubmission,
    ComplianceStatus,
    SubmissionStatus
)
from services.document_processing_service import DocumentProcessingService
from services.chroma_service import ChromaService
from services.granite_service import GraniteLanguageDetectionService
from services.granite_embedding_service import GraniteEmbeddingService
from database.repository import ClaimsRepository, ProcessingStepsRepository, VectorQueriesRepository

logger = logging.getLogger(__name__)


class ClaimsIntelligenceService:
    """
    Core service for intelligent claims processing using RAG (Retrieval-Augmented Generation).
    
    This service orchestrates the entire claims processing workflow:
    1. Document processing and data extraction
    2. Language detection and text preprocessing
    3. Vector search for relevant guidelines and packages
    4. AI-powered analysis and recommendation generation
    5. Risk assessment and compliance checking
    6. Claims submission preparation
    """
    
    def __init__(
        self,
        document_service: DocumentProcessingService,
        chroma_service: ChromaService,
        language_service: GraniteLanguageDetectionService,
        embedding_service: GraniteEmbeddingService,
        claims_repository: ClaimsRepository,
        steps_repository: ProcessingStepsRepository,
        vector_repository: VectorQueriesRepository
    ):
        self.document_service = document_service
        self.chroma_service = chroma_service
        self.language_service = language_service
        self.embedding_service = embedding_service
        self.claims_repository = claims_repository
        self.steps_repository = steps_repository
        self.vector_repository = vector_repository
        
        # Configuration
        self.confidence_threshold = 0.7
        self.max_recommendations = 5
        self.risk_threshold = 0.8
        
        logger.info("Claims Intelligence Service initialized")
    
    async def process_claim_document(
        self, 
        document_path: str, 
        hospital_id: str,
        claim_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a medical document and generate intelligent claims recommendations.
        
        Args:
            document_path: Path to the medical document
            hospital_id: Hospital identifier
            claim_id: Optional existing claim ID
            
        Returns:
            Complete claims analysis with recommendations
        """
        start_time = datetime.now()
        
        try:
            # Generate claim ID if not provided
            if not claim_id:
                claim_id = f"CLM_{hospital_id}_{int(datetime.now().timestamp())}"
            
            logger.info(f"Processing claim document: {claim_id}")
            
            # Step 1: Document Processing and OCR
            await self._log_processing_step(claim_id, "document_processing", "started")
            
            processing_result = await self.document_service.process_document(document_path)
            if not processing_result.medical_record:
                error_msg = "No medical record extracted" if not processing_result.validation_errors else processing_result.validation_errors[0]
                raise Exception(f"Document processing failed: {error_msg}")
            
            medical_data = processing_result.medical_record
            await self._log_processing_step(claim_id, "document_processing", "completed", 
                                          {"confidence": medical_data.document_confidence})
            
            # Step 2: Language Detection
            await self._log_processing_step(claim_id, "language_detection", "started")
            
            combined_text = self._extract_text_for_analysis(medical_data)
            language_result = self.language_service.detect_language(combined_text)
            
            await self._log_processing_step(claim_id, "language_detection", "completed",
                                          {"detected_language": language_result["detected_language"],
                                           "confidence": language_result["confidence"]})
            
            # Step 3: Vector Search for Relevant Guidelines
            await self._log_processing_step(claim_id, "vector_search", "started")
            
            search_results = await self._perform_rag_search(claim_id, medical_data, combined_text)
            
            await self._log_processing_step(claim_id, "vector_search", "completed",
                                          {"results_count": len(search_results["pmjay_results"]) + 
                                                          len(search_results["medical_codes_results"])})
            
            # Step 4: Package Recommendations
            await self._log_processing_step(claim_id, "package_recommendation", "started")
            
            recommendations = await self._generate_package_recommendations(
                claim_id, medical_data, search_results
            )
            
            await self._log_processing_step(claim_id, "package_recommendation", "completed",
                                          {"recommendations_count": len(recommendations)})
            
            # Step 5: Risk Assessment
            await self._log_processing_step(claim_id, "risk_assessment", "started")
            
            risk_assessment = await self._perform_risk_assessment(
                claim_id, medical_data, recommendations, search_results
            )
            
            await self._log_processing_step(claim_id, "risk_assessment", "completed",
                                          {"risk_score": risk_assessment.overall_risk_score})
            
            # Step 6: Compliance Checking
            await self._log_processing_step(claim_id, "compliance_check", "started")
            
            compliance_status = await self._check_compliance(
                medical_data, recommendations, search_results
            )
            
            await self._log_processing_step(claim_id, "compliance_check", "completed",
                                          {"compliance_status": compliance_status.value})
            
            # Step 7: Create Claim Submission
            claim_submission = ClaimSubmission(
                claim_id=claim_id,
                hospital_id=hospital_id,
                patient_data=medical_data,
                recommended_package=recommendations[0] if recommendations else None,
                submission_status=SubmissionStatus.DRAFT,
                created_at=start_time,
                updated_at=datetime.now()
            )
            
            # Save to database
            await self.claims_repository.create_claim(claim_submission)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Prepare final result
            result = {
                "claim_id": claim_id,
                "processing_status": "completed",
                "processing_time_ms": processing_time,
                "document_analysis": {
                    "ocr_confidence": medical_data.document_confidence,
                    "language_detected": language_result["detected_language"],
                    "language_confidence": language_result["confidence"]
                },
                "medical_data": medical_data.dict(),
                "package_recommendations": [rec.dict() for rec in recommendations],
                "risk_assessment": risk_assessment.dict(),
                "compliance_status": compliance_status.value,
                "search_results": search_results,
                "claim_submission": claim_submission.dict()
            }
            
            logger.info(f"Claims processing completed for {claim_id} in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Claims processing failed for {claim_id}: {e}")
            await self._log_processing_step(claim_id, "processing", "failed", {"error": str(e)})
            raise
    
    async def _perform_rag_search(
        self, 
        claim_id: str, 
        medical_data: MedicalRecord, 
        text: str
    ) -> Dict[str, List[Dict]]:
        """
        Perform RAG (Retrieval-Augmented Generation) search for relevant guidelines and codes.
        """
        search_queries = self._generate_search_queries(medical_data, text)
        
        results = {
            "pmjay_results": [],
            "medical_codes_results": [],
            "search_queries": search_queries
        }
        
        for query in search_queries:
            try:
                # Search PM-JAY guidelines
                pmjay_results = self.chroma_service.search_pmjay_guidelines(
                    query, n_results=5
                )
                results["pmjay_results"].extend(pmjay_results)
                
                # Search medical codes
                codes_results = self.chroma_service.search_medical_codes(
                    query, n_results=3
                )
                results["medical_codes_results"].extend(codes_results)
                
                # Log vector query
                await self.vector_repository.log_vector_query(
                    claim_id=claim_id,
                    query_text=query,
                    collection_name="pmjay_guidelines",
                    results_count=len(pmjay_results),
                    avg_similarity=sum(r.get("distance", 0) for r in pmjay_results) / max(len(pmjay_results), 1)
                )
                
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
        
        return results
    
    async def _generate_package_recommendations(
        self,
        claim_id: str,
        medical_data: MedicalRecord,
        search_results: Dict[str, List[Dict]]
    ) -> List[PackageRecommendation]:
        """
        Generate package recommendations based on medical data and search results.
        """
        recommendations = []
        
        # Analyze procedures and diagnoses
        procedures_text = " ".join([proc.procedure_name for proc in medical_data.procedures])
        diagnoses_text = " ".join([diag.diagnosis_name for diag in medical_data.diagnoses])
        
        # Extract relevant packages from search results
        relevant_packages = self._extract_packages_from_search(search_results)
        
        for package_info in relevant_packages[:self.max_recommendations]:
            # Calculate confidence based on text similarity and medical relevance
            confidence = self._calculate_package_confidence(
                package_info, procedures_text, diagnoses_text, medical_data
            )
            
            if confidence >= self.confidence_threshold:
                # Estimate approval probability based on historical data and compliance
                approval_prob = self._estimate_approval_probability(
                    package_info, medical_data, search_results
                )
                
                recommendation = PackageRecommendation(
                    package_code=package_info.get("package_code", f"PKG_{len(recommendations)+1:03d}"),
                    package_name=package_info.get("package_name", "Standard Medical Package"),
                    confidence_score=confidence,
                    estimated_amount=self._estimate_package_amount(package_info, medical_data),
                    approval_probability=approval_prob,
                    risk_factors=package_info.get("risk_factors", []),
                    compliance_status=ComplianceStatus.COMPLIANT if approval_prob > 0.7 else ComplianceStatus.REQUIRES_REVIEW,
                    chroma_similarity_scores=[package_info.get("similarity_score", 0.5)]
                )
                
                recommendations.append(recommendation)
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return recommendations
    
    async def _perform_risk_assessment(
        self,
        claim_id: str,
        medical_data: MedicalRecord,
        recommendations: List[PackageRecommendation],
        search_results: Dict[str, List[Dict]]
    ) -> RiskAssessment:
        """
        Perform comprehensive risk assessment for the claim.
        """
        risk_factors = []
        risk_scores = []
        
        # Document quality risk
        doc_risk = 1.0 - medical_data.document_confidence
        if doc_risk > 0.3:
            risk_factors.append(f"Low document confidence: {medical_data.document_confidence:.2f}")
        risk_scores.append(doc_risk)
        
        # Amount risk (high amounts = higher risk)
        amount_risk = min(float(medical_data.total_amount) / 1000000, 1.0)  # Normalize to 1M
        if amount_risk > 0.5:
            risk_factors.append(f"High claim amount: ₹{medical_data.total_amount}")
        risk_scores.append(amount_risk)
        
        # Recommendation confidence risk
        if recommendations:
            rec_risk = 1.0 - recommendations[0].confidence_score
            if rec_risk > 0.4:
                risk_factors.append(f"Low recommendation confidence: {recommendations[0].confidence_score:.2f}")
            risk_scores.append(rec_risk)
        else:
            risk_factors.append("No suitable package recommendations found")
            risk_scores.append(0.8)
        
        # Compliance risk
        compliance_risk = 0.0
        if recommendations and recommendations[0].compliance_status != ComplianceStatus.COMPLIANT:
            compliance_risk = 0.6
            risk_factors.append("Compliance issues detected")
        risk_scores.append(compliance_risk)
        
        # Calculate overall risk score
        overall_risk = sum(risk_scores) / len(risk_scores)
        
        # Determine risk level
        if overall_risk < 0.3:
            risk_level = "LOW"
        elif overall_risk < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return RiskAssessment(
            overall_risk_score=overall_risk,
            risk_level=risk_level,
            risk_factors=risk_factors,
            fraud_indicators=[],  # Would be populated by fraud detection algorithms
            recommendation="AUTO_APPROVE" if overall_risk < 0.3 else "MANUAL_REVIEW"
        )
    
    async def _check_compliance(
        self,
        medical_data: MedicalRecord,
        recommendations: List[PackageRecommendation],
        search_results: Dict[str, List[Dict]]
    ) -> ComplianceStatus:
        """
        Check compliance with PM-JAY guidelines and regulations.
        """
        compliance_issues = []
        
        # Check basic data completeness
        if not medical_data.patient_info.patient_id:
            compliance_issues.append("Missing patient ID")
        
        if not medical_data.procedures:
            compliance_issues.append("No medical procedures documented")
        
        if not medical_data.diagnoses:
            compliance_issues.append("No diagnoses documented")
        
        # Check amount reasonableness
        if medical_data.total_amount <= 0:
            compliance_issues.append("Invalid claim amount")
        
        # Check date validity
        if medical_data.discharge_date and medical_data.admission_date:
            if medical_data.discharge_date < medical_data.admission_date:
                compliance_issues.append("Invalid admission/discharge dates")
        
        # Check against guidelines from search results
        guideline_violations = self._check_guideline_compliance(medical_data, search_results)
        compliance_issues.extend(guideline_violations)
        
        # Determine compliance status
        if not compliance_issues:
            return ComplianceStatus.COMPLIANT
        elif len(compliance_issues) <= 2:
            return ComplianceStatus.REQUIRES_REVIEW
        else:
            return ComplianceStatus.NON_COMPLIANT
    
    def _extract_text_for_analysis(self, medical_data: MedicalRecord) -> str:
        """Extract relevant text from medical data for analysis."""
        text_parts = []
        
        # Patient information
        text_parts.append(f"Patient: {medical_data.patient_info.name}")
        text_parts.append(f"Age: {medical_data.patient_info.age}")
        text_parts.append(f"Gender: {medical_data.patient_info.gender}")
        
        # Procedures
        for proc in medical_data.procedures:
            text_parts.append(f"Procedure: {proc.procedure_name}")
            if proc.procedure_code:
                text_parts.append(f"Code: {proc.procedure_code}")
        
        # Diagnoses
        for diag in medical_data.diagnoses:
            text_parts.append(f"Diagnosis: {diag.diagnosis_name}")
            if diag.diagnosis_code:
                text_parts.append(f"Code: {diag.diagnosis_code}")
        
        return " ".join(text_parts)
    
    def _generate_search_queries(self, medical_data: MedicalRecord, text: str) -> List[str]:
        """Generate search queries for RAG retrieval."""
        queries = []
        
        # Primary diagnosis query
        if medical_data.diagnoses:
            primary_diagnosis = medical_data.diagnoses[0].diagnosis_name
            queries.append(f"PM-JAY package for {primary_diagnosis}")
        
        # Procedure-based queries
        for proc in medical_data.procedures[:3]:  # Limit to top 3 procedures
            queries.append(f"Medical package {proc.procedure_name}")
        
        # Combined query
        if medical_data.procedures and medical_data.diagnoses:
            queries.append(f"{medical_data.diagnoses[0].diagnosis_name} {medical_data.procedures[0].procedure_name} treatment")
        
        return queries
    
    def _extract_packages_from_search(self, search_results: Dict[str, List[Dict]]) -> List[Dict]:
        """Extract package information from search results."""
        packages = []
        
        for result in search_results["pmjay_results"]:
            # Extract package info from metadata or document content
            package_info = {
                "package_code": result.get("metadata", {}).get("package_code", "PKG_001"),
                "package_name": result.get("metadata", {}).get("package_name", "Standard Package"),
                "evidence": [result.get("document", "")],
                "similarity_score": 1.0 - result.get("distance", 0.5)
            }
            packages.append(package_info)
        
        return packages
    
    def _calculate_package_confidence(
        self, 
        package_info: Dict, 
        procedures_text: str, 
        diagnoses_text: str, 
        medical_data: MedicalRecord
    ) -> float:
        """Calculate confidence score for package recommendation."""
        base_confidence = package_info.get("similarity_score", 0.5)
        
        # Adjust based on document confidence
        doc_factor = medical_data.document_confidence
        
        # Adjust based on data completeness
        completeness_factor = 1.0
        if not medical_data.procedures:
            completeness_factor *= 0.7
        if not medical_data.diagnoses:
            completeness_factor *= 0.7
        
        return min(base_confidence * doc_factor * completeness_factor, 1.0)
    
    def _estimate_approval_probability(
        self, 
        package_info: Dict, 
        medical_data: MedicalRecord, 
        search_results: Dict
    ) -> float:
        """Estimate approval probability based on various factors."""
        base_prob = 0.7  # Base approval probability
        
        # Adjust based on document confidence
        if medical_data.document_confidence < 0.8:
            base_prob *= 0.9
        
        # Adjust based on amount reasonableness
        if medical_data.total_amount > Decimal("500000"):  # High amount
            base_prob *= 0.85
        
        # Adjust based on data completeness
        if len(medical_data.procedures) == 0:
            base_prob *= 0.8
        if len(medical_data.diagnoses) == 0:
            base_prob *= 0.8
        
        return min(base_prob, 1.0)
    
    def _estimate_package_amount(self, package_info: Dict, medical_data: MedicalRecord) -> Decimal:
        """Estimate package amount based on procedures and historical data."""
        # Use actual amount if available, otherwise estimate
        if medical_data.total_amount > 0:
            return medical_data.total_amount
        
        # Basic estimation based on procedure count and complexity
        base_amount = Decimal("50000")  # Base package amount
        procedure_factor = len(medical_data.procedures) * Decimal("10000")
        
        return base_amount + procedure_factor
    
    def _check_guideline_compliance(
        self, 
        medical_data: MedicalRecord, 
        search_results: Dict[str, List[Dict]]
    ) -> List[str]:
        """Check compliance against PM-JAY guidelines from search results."""
        violations = []
        
        # This would contain more sophisticated compliance checking logic
        # based on the actual guidelines retrieved from the vector search
        
        # Example checks:
        if medical_data.total_amount > Decimal("1000000"):
            violations.append("Amount exceeds maximum limit for standard packages")
        
        return violations
    
    async def _log_processing_step(
        self, 
        claim_id: str, 
        step_name: str, 
        status: str, 
        metadata: Optional[Dict] = None
    ):
        """Log processing step to database."""
        try:
            await self.steps_repository.log_step(
                claim_id=claim_id,
                step_name=step_name,
                status=status,
                metadata=json.dumps(metadata) if metadata else None
            )
        except Exception as e:
            logger.warning(f"Failed to log processing step: {e}")
    
    async def get_claim_analysis(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve complete analysis for an existing claim.
        """
        try:
            # Get claim from database
            claim = await self.claims_repository.get_claim(claim_id)
            if not claim:
                return None
            
            # Get processing steps
            steps = await self.steps_repository.get_steps_for_claim(claim_id)
            
            # Get vector queries
            queries = await self.vector_repository.get_queries_for_claim(claim_id)
            
            return {
                "claim": claim,
                "processing_steps": steps,
                "vector_queries": queries
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve claim analysis for {claim_id}: {e}")
            return None
    
    async def reprocess_claim(self, claim_id: str) -> Dict[str, Any]:
        """
        Reprocess an existing claim with updated logic.
        """
        try:
            # Get existing claim
            claim = await self.claims_repository.get_claim(claim_id)
            if not claim:
                raise ValueError(f"Claim {claim_id} not found")
            
            # Increment retry count
            await self.claims_repository.increment_retry_count(claim_id)
            
            # Update status
            await self.claims_repository.update_claim_status(
                claim_id, SubmissionStatus.PROCESSING
            )
            
            # Reprocess with existing patient data
            # This would involve re-running the analysis pipeline
            logger.info(f"Reprocessing claim {claim_id}")
            
            # For now, return existing claim data
            # In a full implementation, this would re-run the entire pipeline
            return {"claim_id": claim_id, "status": "reprocessed"}
            
        except Exception as e:
            logger.error(f"Failed to reprocess claim {claim_id}: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the claims intelligence service.
        """
        try:
            # Check all dependent services
            health_status = {
                "service": "claims_intelligence",
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "dependencies": {}
            }
            
            # Check document service
            try:
                doc_health = await self.document_service.health_check()
                health_status["dependencies"]["document_service"] = "healthy"
            except Exception as e:
                health_status["dependencies"]["document_service"] = f"unhealthy: {e}"
                health_status["status"] = "degraded"
            
            # Check chroma service
            try:
                chroma_health = self.chroma_service.health_check()
                health_status["dependencies"]["chroma_service"] = "healthy"
            except Exception as e:
                health_status["dependencies"]["chroma_service"] = f"unhealthy: {e}"
                health_status["status"] = "degraded"
            
            # Check language service
            try:
                lang_health = self.language_service.health_check()
                health_status["dependencies"]["language_service"] = "healthy"
            except Exception as e:
                health_status["dependencies"]["language_service"] = f"unhealthy: {e}"
                health_status["status"] = "degraded"
            
            # Check embedding service
            try:
                embed_health = await self.embedding_service.health_check()
                health_status["dependencies"]["embedding_service"] = "healthy"
            except Exception as e:
                health_status["dependencies"]["embedding_service"] = f"unhealthy: {e}"
                health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "service": "claims_intelligence",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
