from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from app.config import API_KEY

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str = Security(_api_key_header)):
    """No-op when API_KEY env var is not set. Enforces key when it is."""
    if not API_KEY:
        return
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")