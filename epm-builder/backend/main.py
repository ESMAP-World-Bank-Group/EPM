"""
EPM User Interface - FastAPI Backend

Main application entry point for the EPM web interface.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from routes import scenarios, jobs, results, templates, uploads


class ForceHTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """
    Middleware to ensure redirects use HTTPS when behind a proxy.
    Koyeb/Cloudflare terminate SSL, so FastAPI thinks it's HTTP.
    """
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # If there's a redirect with http://, change to https://
        if response.status_code in (301, 302, 307, 308):
            location = response.headers.get("location", "")
            if location.startswith("http://") and "koyeb.app" in location:
                response.headers["location"] = location.replace("http://", "https://", 1)
        return response


app = FastAPI(
    title="EPM User Interface",
    description="Web interface for the Electricity Planning Model",
    version="0.1.0"
    # Note: redirect_slashes is True by default. The ForceHTTPSRedirectMiddleware
    # handles converting http:// redirects to https:// when behind Koyeb/Cloudflare
)

# Add HTTPS redirect fix middleware (must be added before CORS)
app.add_middleware(ForceHTTPSRedirectMiddleware)

# CORS configuration - allow all origins for MVP
# TODO: Restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(scenarios.router, prefix="/api/scenarios", tags=["scenarios"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(results.router, prefix="/api/results", tags=["results"])
app.include_router(templates.router, prefix="/api/templates", tags=["templates"])
app.include_router(uploads.router, prefix="/api/uploads", tags=["uploads"])


@app.get("/")
async def root():
    return {
        "name": "EPM User Interface API",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}
