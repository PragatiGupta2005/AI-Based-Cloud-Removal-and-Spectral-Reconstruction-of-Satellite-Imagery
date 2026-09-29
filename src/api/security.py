"""Optional JWT auth + hardening (audit Sec.18).

Auth is OFF by default (AUTH_ENABLED=false) so existing tests/UI keep working.
Set AUTH_ENABLED=true and API_JWT_SECRET to require Bearer tokens on
/predict, /feedback, /admin/*.
"""
import os
import time
from typing import Optional

import jwt
from fastapi import Header, HTTPException


def auth_enabled() -> bool:
    return os.getenv("AUTH_ENABLED", "false").lower() in ("1", "true", "yes")


def _secret() -> str:
    return os.getenv("API_JWT_SECRET", "cloudclear-dev-secret")


def create_token(subject: str = "user", ttl_s: int = 86400) -> str:
    now = int(time.time())
    return jwt.encode({"sub": subject, "iat": now, "exp": now + ttl_s}, _secret(), algorithm="HS256")


def require_auth(authorization: Optional[str] = Header(default=None)) -> str:
    if not auth_enabled():
        return "anonymous (auth disabled)"
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, _secret(), algorithms=["HS256"])
        return str(payload.get("sub", "user"))
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
