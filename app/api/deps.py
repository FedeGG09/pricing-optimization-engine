from fastapi import Header, HTTPException
from app.core.config import settings
from app.core.security import decode_token


def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    if not settings.enable_auth:
        return {"sub": "local-user", "role": settings.default_role}

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        return decode_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
