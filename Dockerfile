# Container for the demo page: the saved model, the JSON reports and src/serve.py.
#
#   docker build -t url-detector .
#   docker run -p 7860:7860 url-detector          # then open http://localhost:7860
#
# Used as-is by Hugging Face Spaces (deploy/push_to_hf_space.py) and works on any
# host that runs a Dockerfile and sets PORT (Render, Railway, Cloud Run, Fly.io).
# The server needs about 1 GB of memory while the model loads, 600 MB after.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Hugging Face Spaces runs containers as user 1000, so files belong to that user.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY --chown=app src/ src/
COPY --chown=app web/ web/
COPY --chown=app reports/metrics.json reports/external_eval.json reports/red_team.json reports/
COPY --chown=app data/processed/prepare_report.json data/processed/
COPY --chown=app models/malicious_url_rf.joblib models/

USER app
ENV HOST=0.0.0.0 PORT=7860
EXPOSE 7860
CMD ["python", "src/serve.py", "--no-open"]
