FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=9696

WORKDIR /app

# Dependencies first, so that this layer is cached when only the code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY common.py download_data.py train.py predict.py params.json ./

# Download the data and train the model while the image is built:
# a fresh clone then needs only Docker and internet access.
RUN python download_data.py && python train.py

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 9696

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '9696'), timeout=2)"]

# gunicorn needs Unix APIs, so it is used here only (locally on Windows use: python predict.py)
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 2 --preload --timeout 60 --access-logfile - predict:app"]
