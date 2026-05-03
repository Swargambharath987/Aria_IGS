"""
JWT authentication using Slurm's own HMAC tokens.

Each user generates a token on the cluster:
    scontrol token [lifespan=3600]

The server validates using the cluster's jwt_hs256.key (base64-encoded
in SLURM_JWT_KEY). The username is extracted from the 'sun' JWT claim
(Slurm User Name), which Slurm uses for access control automatically.
"""
import base64
import time
from typing import Optional

from authlib.jose import JsonWebToken


def _decode_key(b64_key: str) -> bytes:
    padded = b64_key + "=" * (-len(b64_key) % 4)
    return base64.b64decode(padded)


def validate_token(token: str, jwt_key_b64: str) -> Optional[str]:
    """
    Validate a Slurm JWT and return the username, or None if invalid.
    The token is also forwarded as-is to the Slurm REST API.
    """
    if not jwt_key_b64:
        return None
    try:
        key = _decode_key(jwt_key_b64)
        jwt = JsonWebToken(["HS256"])
        claims = jwt.decode(token, key)

        # Reject expired tokens
        exp = claims.get("exp")
        if exp and time.time() > exp:
            return None

        # Slurm stores the username in the 'sun' claim
        username = claims.get("sun") or claims.get("sub") or claims.get("client_id")
        return username or None

    except Exception:
        return None
