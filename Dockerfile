# Container for the demo page: the saved model, the JSON reports and src/serve.py.
#
#   docker build -t url-detector .
#   docker run -p 7860:7860 url-detector          # then open http://localhost:7860
#
# Deployed on Render's free plan (render.yaml) and usable on any host that runs a
# Dockerfile and sets PORT (Hugging Face Spaces with PRO, Railway, Cloud Run, Fly.io).
# It ships the compact model (src/compact_forest.py): same predictions as the
# scikit-learn forest, and the server peaks near 320 MB instead of 940 MB, which
# fits a 512 MB free plan.
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
COPY --chown=app models/malicious_url_rf.npz models/

USER app
ENV HOST=0.0.0.0 PORT=7860
EXPOSE 7860
CMD ["python", "src/serve.py", "--no-open"]
