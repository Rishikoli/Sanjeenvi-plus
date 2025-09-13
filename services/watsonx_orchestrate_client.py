"""
WatsonX Orchestrate client wrapper.

This module provides a lightweight async client for invoking WatsonX Orchestrate
skills (aka automations) and checking their run statuses. It uses HTTP APIs
with API key authentication, configurable via environment variables.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class OrchestrateRun:
    success: bool
    run_id: Optional[str]
    status: str
    output: Optional[Dict[str, Any]]
    error: Optional[str]


class WatsonXOrchestrateClient:
    """Thin client for calling WatsonX Orchestrate skills."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.base_url = (base_url or os.getenv("WXO_BASE_URL") or "https://api.ibm.com/wxo").rstrip("/")
        self.api_key = api_key or os.getenv("WXO_API_KEY")
        self.tenant_id = tenant_id or os.getenv("WXO_TENANT_ID")
        self.timeout = httpx.Timeout(timeout_seconds)

        if not self.api_key:
            logger.warning("WXO_API_KEY not set. Orchestrate calls will fail until configured.")

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.tenant_id:
            headers["X-Tenant-Id"] = self.tenant_id
        return headers

    async def run_skill(self, skill_id: str, inputs: Dict[str, Any]) -> OrchestrateRun:
        """Invoke a skill and return the run metadata.

        Args:
            skill_id: The Orchestrate skill (automation) identifier
            inputs: JSON-serializable inputs expected by the skill
        """
        try:
            url = f"{self.base_url}/skills/{skill_id}/runs"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, headers=self._headers(), json={"inputs": inputs})
                if resp.status_code not in (200, 201, 202):
                    return OrchestrateRun(False, None, "failed", None, f"HTTP {resp.status_code}: {resp.text}")
                data = resp.json()
                return OrchestrateRun(True, data.get("run_id"), data.get("status", "queued"), data.get("output"), None)
        except Exception as e:
            logger.error(f"WXO run_skill error: {e}")
            return OrchestrateRun(False, None, "error", None, str(e))

    async def get_run_status(self, run_id: str) -> OrchestrateRun:
        """Poll a run for completion and get its output."""
        try:
            url = f"{self.base_url}/runs/{run_id}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, headers=self._headers())
                if resp.status_code != 200:
                    return OrchestrateRun(False, run_id, "failed", None, f"HTTP {resp.status_code}: {resp.text}")
                data = resp.json()
                return OrchestrateRun(True, run_id, data.get("status", "running"), data.get("output"), None)
        except Exception as e:
            logger.error(f"WXO get_run_status error: {e}")
            return OrchestrateRun(False, run_id, "error", None, str(e))

    async def run_and_wait(self, skill_id: str, inputs: Dict[str, Any], poll_interval: float = 2.0, timeout_seconds: int = 120) -> OrchestrateRun:
        """Helper to invoke a skill and wait until it completes or times out."""
        import asyncio
        start = datetime.utcnow()
        run = await self.run_skill(skill_id, inputs)
        if not run.success or not run.run_id:
            return run
        while (datetime.utcnow() - start).total_seconds() < timeout_seconds:
            status = await self.get_run_status(run.run_id)
            if status.status in ("completed", "failed", "canceled"):
                return status
            await asyncio.sleep(poll_interval)
        return OrchestrateRun(False, run.run_id, "timeout", None, "Timed out waiting for run to complete")


# Global client instance (optional)
wxo_client = WatsonXOrchestrateClient()
