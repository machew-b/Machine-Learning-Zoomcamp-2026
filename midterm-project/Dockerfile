FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=9696

# XGBoost needs the OpenMP runtime, which the slim image does not ship
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common.py download_data.py train.py predict.py params.json ./

# Train at build time so that the image is self-contained and reproducible:
# downloads the dataset, trains the model, writes model.bin and reports/metrics.json
RUN python download_data.py && python train.py

# Do not run the service as root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 9696

# $PORT defaults to 9696 and is overridden by platforms such as Cloud Run / Render
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 60 predict:app"]
