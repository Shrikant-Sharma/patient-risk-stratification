# Multi-stage build for Patient Risk Stratification API

# Stage 1: builder — install dependencies in a temporary stage
FROM python:3.13-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: runtime — only copy what's needed to run
FROM python:3.13-slim AS runtime

# Create non-root user FIRST so we can install packages with proper ownership
RUN useradd --create-home --shell /bin/bash apiuser

WORKDIR /app

# Copy installed packages from builder, but assign ownership to apiuser
COPY --from=builder --chown=apiuser:apiuser /root/.local /home/apiuser/.local

# Set environment AFTER copying so PATH points to apiuser's location
ENV PATH=/home/apiuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy app source and models with proper ownership
COPY --chown=apiuser:apiuser src/ ./src/
COPY --chown=apiuser:apiuser models/ ./models/

# Switch to non-root user
USER apiuser

EXPOSE 8000

# Healthcheck — Docker can verify the API is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]