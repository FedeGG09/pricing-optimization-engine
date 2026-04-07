from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import settings
from app.core.logging import configure_logging
from app.core.observability import MetricsMiddleware


configure_logging()

app = FastAPI(title=settings.app_name, version="2.0.0")
app.add_middleware(MetricsMiddleware)
app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "name": settings.app_name,
        "env": settings.app_env,
        "status": "ready",
        "mode": "hybrid-pricing",
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "error": str(exc)})
