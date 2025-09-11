"""Document processing service using IBM Docling for OCR and text extraction."""

import logging
import os
import time
from typing import Dict, Any, Optional, List, BinaryIO
from pathlib import Path
import tempfile
from dataclasses import dataclass

# Import Docling components
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logging.warning("IBM Docling not available. Document processing will use fallback methods.")

from services.granite_service import granite_language_service
from models.medical import MedicalRecord, PatientInfo, MedicalProcedure, Diagnosis
from decimal import Decimal
from datetime import datetime
import re
import json

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result of OCR processing."""
    text: str
    confidence: float
    language: str
    processing_time_ms: float
    page_count: int
    method: str
    metadata: Dict[str, Any]


@dataclass
class DocumentProcessingResult:
    """Result of complete document processing."""
    ocr_result: OCRResult
    extracted_data: Optional[Dict[str, Any]]
    medical_record: Optional[MedicalRecord]
    validation_errors: List[str]
    requires_verification: bool
    processing_summary: Dict[str, Any]


class DocumentProcessingService:
    """Service for processing medical documents using IBM Docling."""
    
    def __init__(self):
        """Initialize document processing service."""
        self.docling_available = DOCLING_AVAILABLE
        self.confidence_threshold = 0.8
        self.supported_formats = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        
        # Initialize Docling converter if available
        self.converter = None
        if self.docling_available:
            self._initialize_docling()
    
    def _initialize_docling(self):
        """Initialize IBM Docling document converter."""
        try:
            # Configure PDF pipeline options for medical documents
            pdf_options = PdfPipelineOptions()
            pdf_options.do_ocr = True
            pdf_options.do_table_structure = True
            pdf_options.table_structure_options.do_cell_matching = True
            
            # Initialize converter with options
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
                }
            )
            
            logger.info("IBM Docling converter initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Docling converter: {e}")
            self.converter = None
    
    def process_document(
        self, 
        file_path: str, 
        confidence_threshold: Optional[float] = None
    ) -> DocumentProcessingResult:
        """
        Process a medical document through complete pipeline.
        
        Args:
            file_path: Path to the document file
            confidence_threshold: Minimum OCR confidence threshold
            
        Returns:
            Complete processing result with extracted medical data
        """
        try:
            start_time = time.time()
            threshold = confidence_threshold or self.confidence_threshold
            
            # Validate file
            if not self._validate_file(file_path):
                return self._create_error_result("Invalid file format or file not found")
            
            # Perform OCR
            ocr_result = self.extract_text_ocr(file_path)
            
            # Check OCR confidence
            requires_verification = ocr_result.confidence < threshold
            
            # Extract structured data
            extracted_data = None
            medical_record = None
            validation_errors = []
            
            if ocr_result.text and not requires_verification:
                try:
                    extracted_data = self.extract_medical_data(ocr_result.text)
                    medical_record = self.create_medical_record(extracted_data)
                except Exception as e:
                    validation_errors.append(f"Data extraction failed: {str(e)}")
                    requires_verification = True
            
            processing_time = (time.time() - start_time) * 1000
            
            # Create processing summary
            processing_summary = {
                "total_processing_time_ms": processing_time,
                "ocr_confidence": ocr_result.confidence,
                "language_detected": ocr_result.language,
                "requires_verification": requires_verification,
                "extraction_successful": medical_record is not None,
                "validation_errors_count": len(validation_errors)
            }
            
            return DocumentProcessingResult(
                ocr_result=ocr_result,
                extracted_data=extracted_data,
                medical_record=medical_record,
                validation_errors=validation_errors,
                requires_verification=requires_verification,
                processing_summary=processing_summary
            )
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return self._create_error_result(f"Processing failed: {str(e)}")
    
    def extract_text_ocr(self, file_path: str) -> OCRResult:
        """
        Extract text from document using OCR.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            OCR result with extracted text and metadata
        """
        try:
            start_time = time.time()
            
            if self.converter and self.docling_available:
                result = self._docling_ocr(file_path)
            else:
                result = self._fallback_ocr(file_path)
            
            # Detect language
            if result.text:
                lang_result = granite_language_service.detect_language(result.text)
                result.language = lang_result.get("detected_language", "unknown")
                
                # Adjust confidence based on language detection confidence
                lang_confidence = lang_result.get("confidence", 0.5)
                result.confidence = (result.confidence + lang_confidence) / 2
            
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            logger.info(f"OCR completed: {len(result.text)} characters, confidence: {result.confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                language="unknown",
                processing_time_ms=0,
                page_count=0,
                method="error",
                metadata={"error": str(e)}
            )
    
    def _docling_ocr(self, file_path: str) -> OCRResult:
        """Perform OCR using IBM Docling."""
        try:
            # Convert document
            result = self.converter.convert(file_path)
            
            # Extract text content
            text_content = result.document.export_to_markdown()
            
            # Calculate confidence (Docling doesn't provide direct confidence scores)
            # We'll estimate based on text quality indicators
            confidence = self._estimate_ocr_confidence(text_content)
            
            # Get document metadata
            metadata = {
                "pages": getattr(result.document, 'page_count', 1),
                "format": Path(file_path).suffix.lower(),
                "docling_version": "2.4.1",
                "tables_detected": len(getattr(result.document, 'tables', [])),
                "figures_detected": len(getattr(result.document, 'figures', []))
            }
            
            return OCRResult(
                text=text_content,
                confidence=confidence,
                language="unknown",  # Will be detected later
                processing_time_ms=0,  # Will be set by caller
                page_count=metadata.get("pages", 1),
                method="docling",
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Docling OCR failed: {e}")
            raise
    
    def _fallback_ocr(self, file_path: str) -> OCRResult:
        """Fallback OCR method when Docling is not available."""
        try:
            # Simple text extraction for PDFs or basic image processing
            # This is a basic implementation - in production, you might use pytesseract or similar
            
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                # Try to extract text from PDF
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                        
                        confidence = 0.7 if text.strip() else 0.1
                        
                        return OCRResult(
                            text=text,
                            confidence=confidence,
                            language="unknown",
                            processing_time_ms=0,
                            page_count=len(pdf_reader.pages),
                            method="pypdf2",
                            metadata={"pages": len(pdf_reader.pages)}
                        )
                except ImportError:
                    logger.warning("PyPDF2 not available for PDF text extraction")
            
            # For images or when PDF extraction fails, return placeholder
            return OCRResult(
                text="[OCR processing requires IBM Docling or additional libraries]",
                confidence=0.1,
                language="unknown",
                processing_time_ms=0,
                page_count=1,
                method="fallback",
                metadata={"note": "Limited OCR capability without Docling"}
            )
            
        except Exception as e:
            logger.error(f"Fallback OCR failed: {e}")
            raise
    
    def _estimate_ocr_confidence(self, text: str) -> float:
        """Estimate OCR confidence based on text quality indicators."""
        if not text:
            return 0.0
        
        # Basic heuristics for confidence estimation
        confidence_factors = []
        
        # Length factor (longer text usually indicates successful OCR)
        length_factor = min(len(text) / 1000, 1.0)
        confidence_factors.append(length_factor * 0.3)
        
        # Character diversity (good OCR should have varied characters)
        unique_chars = len(set(text.lower()))
        diversity_factor = min(unique_chars / 50, 1.0)
        confidence_factors.append(diversity_factor * 0.2)
        
        # Word structure (presence of complete words)
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        word_factor = min(avg_word_length / 5, 1.0)
        confidence_factors.append(word_factor * 0.2)
        
        # Medical terminology presence
        medical_terms = ['patient', 'diagnosis', 'treatment', 'hospital', 'doctor', 'medical', 'procedure']
        medical_count = sum(1 for term in medical_terms if term.lower() in text.lower())
        medical_factor = min(medical_count / len(medical_terms), 1.0)
        confidence_factors.append(medical_factor * 0.3)
        
        return min(sum(confidence_factors), 1.0)
    
    def extract_medical_data(self, text: str) -> Dict[str, Any]:
        """Extract structured medical data from OCR text."""
        try:
            extracted_data = {
                "patient_info": self._extract_patient_info(text),
                "procedures": self._extract_procedures(text),
                "diagnoses": self._extract_diagnoses(text),
                "hospital_info": self._extract_hospital_info(text),
                "dates": self._extract_dates(text),
                "amounts": self._extract_amounts(text)
            }
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Medical data extraction failed: {e}")
            raise
    
    def _extract_patient_info(self, text: str) -> Dict[str, Any]:
        """Extract patient information from text."""
        patient_info = {}
        
        # Patient ID patterns
        id_patterns = [
            r'patient\s*id[:\s]*([A-Z0-9]+)',
            r'id[:\s]*([A-Z0-9]+)',
            r'patient\s*number[:\s]*([A-Z0-9]+)'
        ]
        
        for pattern in id_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                patient_info['patient_id'] = match.group(1)
                break
        
        # Name patterns
        name_patterns = [
            r'patient\s*name[:\s]*([A-Za-z\s]+)',
            r'name[:\s]*([A-Za-z\s]+)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 2 and len(name) < 50:  # Reasonable name length
                    patient_info['name'] = name
                    break
        
        # Age patterns
        age_patterns = [
            r'age[:\s]*(\d{1,3})',
            r'(\d{1,3})\s*years?\s*old',
            r'aged\s*(\d{1,3})'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                if 0 < age < 150:  # Reasonable age range
                    patient_info['age'] = age
                    break
        
        # Gender patterns
        gender_patterns = [
            r'gender[:\s]*(male|female|m|f)',
            r'sex[:\s]*(male|female|m|f)',
            r'\b(male|female)\b'
        ]
        
        for pattern in gender_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                gender = match.group(1).lower()
                if gender in ['m', 'male']:
                    patient_info['gender'] = 'Male'
                elif gender in ['f', 'female']:
                    patient_info['gender'] = 'Female'
                break
        
        return patient_info
    
    def _extract_procedures(self, text: str) -> List[Dict[str, str]]:
        """Extract medical procedures from text."""
        procedures = []
        
        # Common procedure patterns
        procedure_patterns = [
            r'procedure[:\s]*([^.\n]+)',
            r'surgery[:\s]*([^.\n]+)',
            r'operation[:\s]*([^.\n]+)',
            r'treatment[:\s]*([^.\n]+)'
        ]
        
        for pattern in procedure_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                procedure_text = match.group(1).strip()
                if len(procedure_text) > 5:  # Reasonable procedure description length
                    procedures.append({
                        "code": "",  # Would need medical coding system
                        "description": procedure_text
                    })
        
        return procedures
    
    def _extract_diagnoses(self, text: str) -> List[Dict[str, str]]:
        """Extract diagnoses from text."""
        diagnoses = []
        
        # Diagnosis patterns
        diagnosis_patterns = [
            r'diagnosis[:\s]*([^.\n]+)',
            r'diagnosed\s*with[:\s]*([^.\n]+)',
            r'condition[:\s]*([^.\n]+)'
        ]
        
        for pattern in diagnosis_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                diagnosis_text = match.group(1).strip()
                if len(diagnosis_text) > 3:
                    diagnoses.append({
                        "code": "",  # Would need ICD coding
                        "description": diagnosis_text
                    })
        
        return diagnoses
    
    def _extract_hospital_info(self, text: str) -> Dict[str, str]:
        """Extract hospital information from text."""
        hospital_info = {}
        
        # Hospital ID patterns
        hospital_patterns = [
            r'hospital\s*id[:\s]*([A-Z0-9]+)',
            r'facility[:\s]*([A-Z0-9]+)',
            r'institution[:\s]*([A-Z0-9]+)'
        ]
        
        for pattern in hospital_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                hospital_info['hospital_id'] = match.group(1)
                break
        
        return hospital_info
    
    def _extract_dates(self, text: str) -> Dict[str, str]:
        """Extract relevant dates from text."""
        dates = {}
        
        # Date patterns (various formats)
        date_patterns = [
            r'admission\s*date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'discharge\s*date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'date\s*of\s*admission[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'date\s*of\s*discharge[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                if 'admission' in pattern:
                    dates['admission_date'] = date_str
                elif 'discharge' in pattern:
                    dates['discharge_date'] = date_str
        
        return dates
    
    def _extract_amounts(self, text: str) -> Dict[str, str]:
        """Extract monetary amounts from text."""
        amounts = {}
        
        # Amount patterns
        amount_patterns = [
            r'total[:\s]*₹?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'amount[:\s]*₹?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'bill[:\s]*₹?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'₹\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                amounts['total_amount'] = amount_str
                break
        
        return amounts
    
    def create_medical_record(self, extracted_data: Dict[str, Any]) -> MedicalRecord:
        """Create a MedicalRecord object from extracted data."""
        try:
            # Create patient info
            patient_data = extracted_data.get("patient_info", {})
            patient_info = PatientInfo(
                patient_id=patient_data.get("patient_id", "UNKNOWN"),
                name=patient_data.get("name", "Unknown Patient"),
                age=patient_data.get("age", 0),
                gender=patient_data.get("gender", "Unknown")
            )
            
            # Create procedures
            procedures = []
            for proc_data in extracted_data.get("procedures", []):
                procedure = MedicalProcedure(
                    code=proc_data.get("code", ""),
                    description=proc_data.get("description", "")
                )
                procedures.append(procedure)
            
            # Create diagnoses
            diagnoses = []
            for diag_data in extracted_data.get("diagnoses", []):
                diagnosis = Diagnosis(
                    code=diag_data.get("code", ""),
                    description=diag_data.get("description", "")
                )
                diagnoses.append(diagnosis)
            
            # Parse dates
            dates_data = extracted_data.get("dates", {})
            admission_date = self._parse_date(dates_data.get("admission_date", ""))
            discharge_date = self._parse_date(dates_data.get("discharge_date", ""))
            
            # Parse amounts
            amounts_data = extracted_data.get("amounts", {})
            total_amount = self._parse_amount(amounts_data.get("total_amount", "0"))
            
            # Get hospital info
            hospital_data = extracted_data.get("hospital_info", {})
            hospital_id = hospital_data.get("hospital_id", "UNKNOWN")
            
            medical_record = MedicalRecord(
                patient_info=patient_info,
                hospital_id=hospital_id,
                admission_date=admission_date,
                discharge_date=discharge_date,
                procedures=procedures,
                diagnoses=diagnoses,
                total_amount=total_amount,
                document_confidence=0.8  # Default confidence
            )
            
            return medical_record
            
        except Exception as e:
            logger.error(f"Failed to create medical record: {e}")
            raise
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object."""
        if not date_str:
            return datetime.now()
        
        # Try different date formats
        date_formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%m-%d-%Y",
            "%d/%m/%y", "%d-%m-%y", "%m/%d/%y", "%m-%d-%y"
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # If parsing fails, return current date
        logger.warning(f"Could not parse date: {date_str}")
        return datetime.now()
    
    def _parse_amount(self, amount_str: str) -> Decimal:
        """Parse amount string to Decimal."""
        if not amount_str:
            return Decimal("0.00")
        
        try:
            # Remove currency symbols and commas
            cleaned = re.sub(r'[₹,\s]', '', amount_str)
            return Decimal(cleaned)
        except:
            logger.warning(f"Could not parse amount: {amount_str}")
            return Decimal("0.00")
    
    def _validate_file(self, file_path: str) -> bool:
        """Validate if file exists and has supported format."""
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        if path.suffix.lower() not in self.supported_formats:
            logger.error(f"Unsupported file format: {path.suffix}")
            return False
        
        return True
    
    def _create_error_result(self, error_message: str) -> DocumentProcessingResult:
        """Create an error result."""
        ocr_result = OCRResult(
            text="",
            confidence=0.0,
            language="unknown",
            processing_time_ms=0,
            page_count=0,
            method="error",
            metadata={"error": error_message}
        )
        
        return DocumentProcessingResult(
            ocr_result=ocr_result,
            extracted_data=None,
            medical_record=None,
            validation_errors=[error_message],
            requires_verification=True,
            processing_summary={"error": error_message}
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Check if document processing service is healthy."""
        return {
            "status": "healthy" if self.docling_available else "limited",
            "docling_available": self.docling_available,
            "supported_formats": self.supported_formats,
            "confidence_threshold": self.confidence_threshold,
            "converter_initialized": self.converter is not None
        }


# Global document processing service instance
document_processing_service = DocumentProcessingService()
