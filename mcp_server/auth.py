"""
JWT authentication for the Slurm MCP server.

Validates the same Aria JWT issued by the backend at login.
The username is extracted from the standard 'sub' claim.
"""
import time
from typing import Optional

from jose import JWTError, jwt

from config import settings


def validate_token(token: str) -> Optional[str]:
    """
    Validate an Aria JWT and return the username (sub claim), or None if invalid.
    """
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        exp = payload.get("exp")
        if exp and time.time() > exp:
            return None
        return payload.get("sub") or None
    except JWTError:
        return None
