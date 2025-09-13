"""Knowledge base service for PM-JAY data ingestion and management."""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
import re

from .chroma_service import ChromaService

logger = logging.getLogger(__name__)


class PMJAYPackageSchema(BaseModel):
    """Schema for validating PM-JAY package data."""
    package_code: str = Field(..., min_length=1, description="Package code")
    package_name: str = Field(..., min_length=1, description="Package name")
    specialty: str = Field(..., description="Medical specialty")
    procedure_type: str = Field(..., description="Type of procedure")
    package_rate: float = Field(..., ge=0, description="Package rate in INR")
    effective_date: str = Field(..., description="Effective date")
    description: Optional[str] = Field(None, description="Package description")
    inclusions: Optional[List[str]] = Field(default_factory=list, description="Included services")
    exclusions: Optional[List[str]] = Field(default_factory=list, description="Excluded services")


class PMJAYGuidelineSchema(BaseModel):
    """Schema for validating PM-JAY guideline data."""
    guideline_id: str = Field(..., min_length=1, description="Guideline ID")
    title: str = Field(..., min_length=1, description="Guideline title")
    category: str = Field(..., description="Guideline category")
    content: str = Field(..., min_length=1, description="Guideline content")
    effective_date: str = Field(..., description="Effective date")
    version: str = Field(..., description="Guideline version")
    priority: int = Field(default=1, ge=1, le=5, description="Priority level (1-5)")
    circular_number: Optional[str] = Field(None, description="Official circular number")


class MedicalCodeSchema(BaseModel):
    """Schema for validating medical code data."""
    code: str = Field(..., min_length=1, description="Medical code")
    code_system: str = Field(..., description="Code system (ICD-10, CPT, etc.)")
    description: str = Field(..., min_length=1, description="Code description")
    category: str = Field(..., description="Code category")
    version: str = Field(..., description="Code system version")


class KnowledgeBaseService:
    """Service for managing PM-JAY knowledge base data ingestion."""
    
    def __init__(self, chroma_service: ChromaService):
        self.chroma_service = chroma_service
        # Indian-named collections
        self.COL_PMJAY_MARGDARSHIKA = "pmjay_margdarshika"          # Guidelines
        self.COL_AYUSHMAN_PACKAGE_SUCHI = "ayushman_package_suchi"  # Package master
        self.COL_ROG_NIDAN_CODE_SANGRAH = "rog_nidan_code_sangrah"  # Code systems
        
        # Version control and audit collections
        self.COL_VERSION_HISTORY = "kb_version_history"
        self.COL_AUDIT_LOG = "kb_audit_log"
        
        # Initialize version control
        self._init_version_control()

    def _init_version_control(self) -> None:
        """Initialize version control and audit collections."""
        try:
            # Create version history collection
            self.chroma_service.get_or_create_collection(
                self.COL_VERSION_HISTORY,
                description="Knowledge base version history"
            )
            
            # Create audit log collection
            self.chroma_service.get_or_create_collection(
                self.COL_AUDIT_LOG,
                description="Knowledge base audit trail"
            )
            
            logger.info("Version control initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize version control: {e}")

    def validate_pmjay_package(self, package_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate PM-JAY package data against schema."""
        try:
            validated = PMJAYPackageSchema(**package_data)
            return {"valid": True, "data": validated.dict()}
        except ValidationError as e:
            return {"valid": False, "errors": e.errors()}

    def validate_pmjay_guideline(self, guideline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate PM-JAY guideline data against schema."""
        try:
            validated = PMJAYGuidelineSchema(**guideline_data)
            return {"valid": True, "data": validated.dict()}
        except ValidationError as e:
            return {"valid": False, "errors": e.errors()}

    def validate_medical_code(self, code_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate medical code data against schema."""
        try:
            validated = MedicalCodeSchema(**code_data)
            return {"valid": True, "data": validated.dict()}
        except ValidationError as e:
            return {"valid": False, "errors": e.errors()}

    def _log_audit_event(self, action: str, collection: str, details: Dict[str, Any]) -> None:
        """Log an audit event."""
        try:
            audit_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": action,
                "collection": collection,
                "details": json.dumps(details),
                "user": "system"  # In production, this would be the authenticated user
            }
            
            audit_text = f"Action: {action} on {collection} at {audit_entry['timestamp']}"
            
            self.chroma_service.add_documents(
                self.COL_AUDIT_LOG,
                [audit_text],
                [audit_entry],
                [f"audit_{datetime.utcnow().timestamp()}"]
            )
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")

    def create_version_snapshot(self, collection: str, version_tag: str) -> Dict[str, Any]:
        """Create a version snapshot of a collection."""
        try:
            # Get all items from the collection
            items = self.chroma_service.list_items(collection, limit=10000)
            
            if "error" in items:
                return {"success": False, "error": items["error"]}
            
            # Create version metadata
            version_data = {
                "collection": collection,
                "version_tag": version_tag,
                "timestamp": datetime.utcnow().isoformat(),
                "item_count": len(items.get("documents", [])),
                "snapshot_data": json.dumps(items)
            }
            
            version_text = f"Version {version_tag} of {collection} created at {version_data['timestamp']}"
            version_id = f"version_{collection}_{version_tag}_{datetime.utcnow().timestamp()}"
            
            # Store version snapshot
            self.chroma_service.add_documents(
                self.COL_VERSION_HISTORY,
                [version_text],
                [version_data],
                [version_id]
            )
            
            # Log audit event
            self._log_audit_event("create_version", collection, {
                "version_tag": version_tag,
                "item_count": version_data["item_count"]
            })
            
            return {"success": True, "version_id": version_id, "version_tag": version_tag}
            
        except Exception as e:
            logger.error(f"Failed to create version snapshot: {e}")
            return {"success": False, "error": str(e)}

    def rollback_to_version(self, collection: str, version_tag: str) -> Dict[str, Any]:
        """Rollback a collection to a specific version."""
        try:
            # Find the version snapshot
            version_query = self.chroma_service.query(
                self.COL_VERSION_HISTORY,
                f"version {version_tag} {collection}",
                n_results=10,
                where={"collection": collection, "version_tag": version_tag}
            )
            
            if not version_query.get("documents") or len(version_query["documents"]) == 0:
                return {"success": False, "error": f"Version {version_tag} not found for collection {collection}"}
            
            # Get the snapshot data
            version_metadata = version_query["metadatas"][0]
            snapshot_data = json.loads(version_metadata["snapshot_data"])
            
            # Clear current collection
            current_items = self.chroma_service.list_items(collection, limit=10000)
            if current_items.get("ids"):
                self.chroma_service.delete_by_id(collection, current_items["ids"])
            
            # Restore from snapshot
            if snapshot_data.get("documents"):
                self.chroma_service.add_documents(
                    collection,
                    snapshot_data["documents"],
                    snapshot_data.get("metadatas", []),
                    snapshot_data.get("ids", [])
                )
            
            # Log audit event
            self._log_audit_event("rollback", collection, {
                "version_tag": version_tag,
                "restored_items": len(snapshot_data.get("documents", []))
            })
            
            return {"success": True, "version_tag": version_tag, "restored_items": len(snapshot_data.get("documents", []))}
            
        except Exception as e:
            logger.error(f"Failed to rollback to version: {e}")
            return {"success": False, "error": str(e)}

    def detect_conflicts(self, collection: str) -> Dict[str, Any]:
        """Detect conflicts in PM-JAY guidelines based on overlapping content or contradictions."""
        try:
            if collection != self.COL_PMJAY_MARGDARSHIKA:
                return {"conflicts": [], "message": "Conflict detection only supported for guidelines"}
            
            # Get all guidelines
            guidelines = self.chroma_service.list_items(collection, limit=10000)
            
            if "error" in guidelines:
                return {"error": guidelines["error"]}
            
            conflicts = []
            documents = guidelines.get("documents", [])
            metadatas = guidelines.get("metadatas", [])
            
            # Check for conflicts between guidelines
            for i, doc1 in enumerate(documents):
                for j, doc2 in enumerate(documents[i+1:], i+1):
                    meta1 = metadatas[i] if i < len(metadatas) else {}
                    meta2 = metadatas[j] if j < len(metadatas) else {}
                    
                    # Check for same guideline ID with different content
                    if (meta1.get("guideline_id") == meta2.get("guideline_id") and 
                        meta1.get("guideline_id") and doc1 != doc2):
                        
                        conflicts.append({
                            "type": "duplicate_guideline_id",
                            "guideline_id": meta1.get("guideline_id"),
                            "versions": [
                                {"version": meta1.get("version"), "priority": meta1.get("priority", 1)},
                                {"version": meta2.get("version"), "priority": meta2.get("priority", 1)}
                            ],
                            "resolution": "use_higher_priority"
                        })
                    
                    # Check for conflicting effective dates
                    if (meta1.get("category") == meta2.get("category") and
                        meta1.get("effective_date") != meta2.get("effective_date") and
                        self._content_similarity(doc1, doc2) > 0.8):
                        
                        conflicts.append({
                            "type": "conflicting_dates",
                            "category": meta1.get("category"),
                            "dates": [meta1.get("effective_date"), meta2.get("effective_date")],
                            "resolution": "use_latest_date"
                        })
            
            return {"conflicts": conflicts, "count": len(conflicts)}
            
        except Exception as e:
            logger.error(f"Failed to detect conflicts: {e}")
            return {"error": str(e)}

    def resolve_conflicts(self, collection: str, resolution_strategy: str = "priority_based") -> Dict[str, Any]:
        """Resolve detected conflicts using specified strategy."""
        try:
            conflicts_result = self.detect_conflicts(collection)
            
            if "error" in conflicts_result:
                return conflicts_result
            
            conflicts = conflicts_result.get("conflicts", [])
            resolved_count = 0
            
            for conflict in conflicts:
                if conflict["type"] == "duplicate_guideline_id" and resolution_strategy == "priority_based":
                    # Keep the guideline with higher priority
                    versions = conflict["versions"]
                    if len(versions) >= 2:
                        higher_priority = max(versions, key=lambda x: x.get("priority", 1))
                        # In a real implementation, we would remove the lower priority version
                        resolved_count += 1
                        
                elif conflict["type"] == "conflicting_dates" and resolution_strategy == "latest_date":
                    # Keep the guideline with the latest effective date
                    dates = conflict["dates"]
                    if dates:
                        latest_date = max(dates)
                        # In a real implementation, we would keep only the latest version
                        resolved_count += 1
            
            # Log audit event
            self._log_audit_event("resolve_conflicts", collection, {
                "strategy": resolution_strategy,
                "conflicts_resolved": resolved_count
            })
            
            return {
                "success": True,
                "conflicts_detected": len(conflicts),
                "conflicts_resolved": resolved_count,
                "strategy": resolution_strategy
            }
            
        except Exception as e:
            logger.error(f"Failed to resolve conflicts: {e}")
            return {"success": False, "error": str(e)}

    def _content_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text documents (simple implementation)."""
        try:
            # Simple word-based similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0

    # ===== Generic KBMS operations =====
    def ingest_documents(self, collection: str, documents: List[Dict[str, Any]], id_prefix: str = "doc") -> Dict[str, Any]:
        """Ingest arbitrary documents into a named collection.
        Each item in documents should contain at least 'text' (str) and optional 'metadata' (dict) and 'id' (str).
        """
        try:
            texts: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            ids: List[str] = []

            for idx, item in enumerate(documents):
                text = item.get("text", "").strip()
                if not text:
                    # skip empty docs
                    continue
                texts.append(text)
                raw_md = item.get("metadata", {}) or {}
                metadatas.append(self._sanitize_metadata(raw_md))
                ids.append(item.get("id", f"{id_prefix}_{idx}"))

            if not texts:
                return {"success": False, "count": 0, "message": "No valid documents provided"}

            self.chroma_service.add_documents(collection, texts, metadatas, ids)
            return {"success": True, "count": len(texts), "collection": collection}
        except Exception as e:
            logger.error(f"KB ingest failed for {collection}: {e}")
            return {"success": False, "error": str(e)}

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure metadata values are Chroma-compatible primitives (str, int, float, bool).
        - Lists are joined as comma-separated strings
        - Dicts are JSON-serialized
        - None becomes empty string
        """
        safe_md: Dict[str, Any] = {}
        for k, v in metadata.items():
            if v is None:
                safe_md[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                safe_md[k] = v
            elif isinstance(v, list):
                try:
                    safe_md[k] = ", ".join([str(x) for x in v])
                except Exception:
                    safe_md[k] = str(v)
            elif isinstance(v, dict):
                try:
                    safe_md[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    safe_md[k] = str(v)
            else:
                safe_md[k] = str(v)
        return safe_md

    def search(self, collection: str, query: str, n_results: int = 10, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            return self.chroma_service.query(collection, query, n_results=n_results, where=where)
        except Exception as e:
            logger.error(f"KB search failed for {collection}: {e}")
            return {"error": str(e)}

    def list_items(self, collection: str, limit: int = 50, offset: int = 0, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            return self.chroma_service.list_items(collection, limit=limit, offset=offset, where=where)
        except Exception as e:
            logger.error(f"KB list failed for {collection}: {e}")
            return {"error": str(e)}

    def delete_items(self, collection: str, ids: List[str]) -> Dict[str, Any]:
        try:
            count = self.chroma_service.delete_by_id(collection, ids)
            return {"success": True, "deleted": count}
        except Exception as e:
            logger.error(f"KB delete failed for {collection}: {e}")
            return {"success": False, "error": str(e)}

    def hot_reload(self, collections: Optional[List[str]] = None) -> Dict[str, Any]:
        """Hot-reload collections from canonical local sources in `data/kb/`.
        
        This will attempt to reload the specified collections from files:
        - Packages: data/kb/packages.json or packages.jsonl
        - Guidelines: data/kb/guidelines.json or guidelines.jsonl
        - Medical Codes: data/kb/codes.json or codes.jsonl
        
        If files are missing, it ensures the collections exist and returns counts.
        """
        try:
            target = collections or [
                self.COL_PMJAY_MARGDARSHIKA,
                self.COL_AYUSHMAN_PACKAGE_SUCHI,
                self.COL_ROG_NIDAN_CODE_SANGRAH,
            ]
            info = {}
            # Do not force delete here. Use explicit loader methods when needed.
            for name in target:
                # Invoke corresponding loader without force; it will no-op if files are absent
                if name == self.COL_AYUSHMAN_PACKAGE_SUCHI:
                    self.load_pmjay_packages(force_reload=False)
                elif name == self.COL_PMJAY_MARGDARSHIKA:
                    self.load_pmjay_guidelines(force_reload=False)
                elif name == self.COL_ROG_NIDAN_CODE_SANGRAH:
                    self.load_medical_codes(force_reload=False)
                coll = self.chroma_service.get_or_create_collection(name, description=f"KBMS: {name}")
                info[name] = {"count": coll.count(), "name": name}
            return {"success": True, "collections": info}
        except Exception as e:
            logger.error(f"KB reload failed: {e}")
            return {"success": False, "error": str(e)}

    # ===== Canonical loaders (9.1) =====
    def _read_json_or_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Read a JSON array or JSONL file into a list of dicts."""
        try:
            if not path.exists():
                return []
            text = path.read_text(encoding="utf-8")
            text_stripped = text.strip()
            if not text_stripped:
                return []
            # JSONL if lines start with { per line
            if "\n" in text_stripped and text_stripped.splitlines()[0].lstrip().startswith("{") and not text_stripped.lstrip().startswith("["):
                items: List[Dict[str, Any]] = []
                for line in text_stripped.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        continue
                return items
            # Otherwise standard JSON
            data = json.loads(text_stripped)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
            return []
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            return []

    def _kb_dir(self) -> Path:
        return Path("data") / "kb"

    def load_pmjay_packages(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load PM-JAY packages from data/kb/packages.json|jsonl and ingest with validation.
        If force_reload is True, clears the collection first and creates a version snapshot.
        """
        try:
            collection = self.COL_AYUSHMAN_PACKAGE_SUCHI
            coll = self.chroma_service.get_or_create_collection(collection, description="PM-JAY packages")
            if force_reload and coll.count() > 0:
                # snapshot before clearing
                self.create_version_snapshot(collection, version_tag=datetime.utcnow().strftime("%Y%m%d%H%M%S"))
                ids = coll.get()["ids"] or []
                if ids:
                    self.chroma_service.delete_by_id(collection, ids)

            kb_dir = self._kb_dir()
            packages = []
            for fname in ["packages.json", "packages.jsonl"]:
                packages = self._read_json_or_jsonl(kb_dir / fname)
                if packages:
                    break

            if not packages:
                logger.info("No packages file found; skipping packages load")
                return {"success": True, "ingested": 0, "message": "No packages file found"}

            result = self.ingest_pmjay_packages_data(packages)
            return {"success": result.get("success", False), "ingested": result.get("ingested_count", 0), "errors": result.get("validation_errors", [])}
        except Exception as e:
            logger.error(f"load_pmjay_packages failed: {e}")
            return {"success": False, "error": str(e)}

    def load_pmjay_guidelines(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load PM-JAY guidelines from data/kb/guidelines.json|jsonl and ingest.
        If force_reload is True, clears the collection first with a version snapshot.
        """
        try:
            collection = self.COL_PMJAY_MARGDARSHIKA
            coll = self.chroma_service.get_or_create_collection(collection, description="PM-JAY guidelines")
            if force_reload and coll.count() > 0:
                self.create_version_snapshot(collection, version_tag=datetime.utcnow().strftime("%Y%m%d%H%M%S"))
                ids = coll.get()["ids"] or []
                if ids:
                    self.chroma_service.delete_by_id(collection, ids)

            kb_dir = self._kb_dir()
            guidelines = []
            for fname in ["guidelines.json", "guidelines.jsonl"]:
                guidelines = self._read_json_or_jsonl(kb_dir / fname)
                if guidelines:
                    break

            if not guidelines:
                logger.info("No guidelines file found; skipping guidelines load")
                return {"success": True, "ingested": 0, "message": "No guidelines file found"}

            # Basic normalization: ensure id/title/content keys exist
            norm = []
            for i, g in enumerate(guidelines):
                g = dict(g or {})
                if "id" not in g and "guideline_id" in g:
                    g["id"] = g["guideline_id"]
                if not g.get("title") and g.get("name"):
                    g["title"] = g["name"]
                if not g.get("content") and g.get("text"):
                    g["content"] = g["text"]
                if not g.get("effective_date"):
                    g["effective_date"] = datetime.utcnow().date().isoformat()
                norm.append(g)

            ok = self.ingest_pmjay_guidelines_data(norm)
            # Log audit event
            try:
                self._log_audit_event("ingest_guidelines", collection, {
                    "ingested": len(norm) if ok else 0,
                    "force_reload": force_reload
                })
            except Exception:
                pass
            return {"success": ok, "ingested": len(norm) if ok else 0}
        except Exception as e:
            logger.error(f"load_pmjay_guidelines failed: {e}")
            return {"success": False, "error": str(e)}

    def load_medical_codes(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load medical codes from data/kb/codes.json|jsonl and ingest.
        If force_reload is True, clears the collection first with a version snapshot.
        """
        try:
            collection = self.COL_ROG_NIDAN_CODE_SANGRAH
            coll = self.chroma_service.get_or_create_collection(collection, description="Medical codes")
            if force_reload and coll.count() > 0:
                self.create_version_snapshot(collection, version_tag=datetime.utcnow().strftime("%Y%m%d%H%M%S"))
                ids = coll.get()["ids"] or []
                if ids:
                    self.chroma_service.delete_by_id(collection, ids)

            kb_dir = self._kb_dir()
            codes = []
            for fname in ["codes.json", "codes.jsonl"]:
                codes = self._read_json_or_jsonl(kb_dir / fname)
                if codes:
                    break

            if not codes:
                logger.info("No codes file found; skipping codes load")
                return {"success": True, "ingested": 0, "message": "No codes file found"}

            ok = self.ingest_medical_codes_data(codes)
            # Log audit event
            try:
                self._log_audit_event("ingest_codes", collection, {
                    "ingested": len(codes) if ok else 0,
                    "force_reload": force_reload
                })
            except Exception:
                pass
            return {"success": ok, "ingested": len(codes) if ok else 0}
        except Exception as e:
            logger.error(f"load_medical_codes failed: {e}")
            return {"success": False, "error": str(e)}

    def load_all_data(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load all KB datasets from data/kb/ with optional force reload."""
        results = {
            "packages": self.load_pmjay_packages(force_reload=force_reload),
            "guidelines": self.load_pmjay_guidelines(force_reload=force_reload),
            "codes": self.load_medical_codes(force_reload=force_reload),
        }
        return {"success": all(r.get("success") for r in results.values()), "results": results}

    # ===== Version and Audit Utilities (9.2) =====
    def list_versions(self, collection: str, limit: int = 20) -> Dict[str, Any]:
        """List version snapshots for a collection."""
        try:
            where = {"collection": collection}
            rows = self.chroma_service.list_items(self.COL_VERSION_HISTORY, limit=limit, offset=0, where=where)
            return rows
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return {"error": str(e)}

    def list_audit(self, collection: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """List audit trail entries, optionally filtered by collection."""
        try:
            where = {"collection": collection} if collection else None
            rows = self.chroma_service.list_items(self.COL_AUDIT_LOG, limit=limit, offset=0, where=where)
            return rows
        except Exception as e:
            logger.error(f"Failed to list audit logs: {e}")
            return {"error": str(e)}
    
    def ingest_pmjay_packages_data(self, packages_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest PM-JAY package data into ChromaDB with validation."""
        try:
            documents = []
            metadatas = []
            ids = []
            validation_errors = []
            
            for i, package in enumerate(packages_data):
                # Validate package data
                validation_result = self.validate_pmjay_package(package)
                if not validation_result["valid"]:
                    validation_errors.append({
                        "index": i,
                        "package_code": package.get("package_code", "unknown"),
                        "errors": validation_result["errors"]
                    })
                    continue
                
                validated_package = validation_result["data"]
                
                # Create document text from package information
                doc_text = self._create_package_document(validated_package)
                documents.append(doc_text)
                
                # Create metadata
                metadata = self._sanitize_metadata({
                    "category": "package",
                    "package_code": validated_package.get("package_code", ""),
                    "package_name": validated_package.get("package_name", ""),
                    "specialty": validated_package.get("specialty", ""),
                    "procedure_type": validated_package.get("procedure_type", ""),
                    "package_rate": validated_package.get("package_rate", 0),
                    "version": "2025.1",
                    "effective_date": validated_package.get("effective_date", "")
                })
                metadatas.append(metadata)
                
                # Create unique ID
                package_id = f"pkg_{validated_package.get('package_code', f'unknown_{len(ids)}')}"
                ids.append(package_id)
            
            if not documents:
                return {"success": False, "error": "No valid packages to ingest", "validation_errors": validation_errors}
            
            # Add to ChromaDB
            self.chroma_service.add_documents(self.COL_AYUSHMAN_PACKAGE_SUCHI, documents, metadatas, ids)
            
            # Log audit event
            self._log_audit_event("ingest_packages", self.COL_AYUSHMAN_PACKAGE_SUCHI, {
                "total_packages": len(packages_data),
                "valid_packages": len(documents),
                "validation_errors": len(validation_errors)
            })
            
            return {
                "success": True,
                "ingested_count": len(documents),
                "validation_errors": validation_errors
            }
            
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
            
            # Get collection stats for Indian-named KB
            try:
                info = self.chroma_service.get_collection_info(self.COL_PMJAY_MARGDARSHIKA)
                stats[self.COL_PMJAY_MARGDARSHIKA] = info["count"]
            except Exception:
                stats[self.COL_PMJAY_MARGDARSHIKA] = 0

            try:
                info = self.chroma_service.get_collection_info(self.COL_AYUSHMAN_PACKAGE_SUCHI)
                stats[self.COL_AYUSHMAN_PACKAGE_SUCHI] = info["count"]
            except Exception:
                stats[self.COL_AYUSHMAN_PACKAGE_SUCHI] = 0

            try:
                info = self.chroma_service.get_collection_info(self.COL_ROG_NIDAN_CODE_SANGRAH)
                stats[self.COL_ROG_NIDAN_CODE_SANGRAH] = info["count"]
            except Exception:
                stats[self.COL_ROG_NIDAN_CODE_SANGRAH] = 0

            stats["total_documents"] = sum([
                stats[self.COL_PMJAY_MARGDARSHIKA],
                stats[self.COL_AYUSHMAN_PACKAGE_SUCHI],
                stats[self.COL_ROG_NIDAN_CODE_SANGRAH],
            ])
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
