FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_VERSION=v0.2 \
    PORT=80

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models/ ./models/

EXPOSE 80

CMD ["sh", "-c", "exec gunicorn --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} app:app"]
