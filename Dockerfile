# Lightweight Python base image
FROM python:3.11-slim

# Don't write .pyc files; print logs immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Install dependencies first so Docker caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code (secrets and data are excluded by .dockerignore)
COPY . .
# Strip Windows line endings (CRLF) in case the repo was cloned on Windows
RUN sed -i "s/\r$//" run_pipeline.sh && chmod +x run_pipeline.sh

# Streamlit front end (used by the `app` step)
EXPOSE 8501

# Default: run the whole pipeline. Override with a step name, e.g. `docker run ... rf`
ENTRYPOINT ["./run_pipeline.sh"]
CMD ["all"]
