"""
Async HTTP client for the Slurm REST API (slurmrestd).

Every call forwards the user's JWT as both:
  X-SLURM-USER-TOKEN: <jwt>
  X-SLURM-USER-NAME:  <username>

This lets Slurm apply per-user permissions natively — the MCP server
never needs to know what each user is allowed to do.
"""
import httpx

from config import settings


class SlurmClient:
    def __init__(self, username: str, token: str):
        self.username = username
        self.token    = token
        self.base     = f"{settings.slurm_api_url}/slurm/{settings.slurm_api_version}"
        self._headers = {
            "X-SLURM-USER-NAME":  username,
            "X-SLURM-USER-TOKEN": token,
            "Content-Type":       "application/json",
        }

    async def get(self, path: str, params: dict | None = None) -> dict:
        async with httpx.AsyncClient(
            verify=settings.slurm_verify_ssl,
            timeout=settings.slurm_timeout,
        ) as client:
            resp = await client.get(
                f"{self.base}{path}", headers=self._headers, params=params or {}
            )
            resp.raise_for_status()
            return resp.json()

    async def post(self, path: str, body: dict) -> dict:
        async with httpx.AsyncClient(
            verify=settings.slurm_verify_ssl,
            timeout=settings.slurm_timeout,
        ) as client:
            resp = await client.post(
                f"{self.base}{path}", headers=self._headers, json=body
            )
            resp.raise_for_status()
            return resp.json()

    async def delete(self, path: str) -> dict:
        async with httpx.AsyncClient(
            verify=settings.slurm_verify_ssl,
            timeout=settings.slurm_timeout,
        ) as client:
            resp = await client.delete(f"{self.base}{path}", headers=self._headers)
            resp.raise_for_status()
            return resp.json()
