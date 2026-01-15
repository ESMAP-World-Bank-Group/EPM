"""
EPM User Interface - FastAPI Backend

Main application entry point for the EPM web interface.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import scenarios, jobs, results, templates, uploads

app = FastAPI(
    title="EPM User Interface",
    description="Web interface for the Electricity Planning Model",
    version="0.1.0"
)

# CORS configuration - allow all origins for MVP
# TODO: Restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when using "*"
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
