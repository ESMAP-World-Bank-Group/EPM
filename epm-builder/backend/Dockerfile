# =============================================================================
# EPM User Interface - Backend (MVP)
# =============================================================================
# This Dockerfile is for MVP deployment only.
# It bundles template data (data/mvp_template/) for self-contained deployment.
#
# For production, consider:
#   - Using volume mounts for EPM data
#   - Database-backed configuration
#   - Setting EPM_DATA_PATH environment variable
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend code including MVP data bundle
COPY . .

# Create runs folder for scenario outputs (ephemeral in container)
RUN mkdir -p /app/runs

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
