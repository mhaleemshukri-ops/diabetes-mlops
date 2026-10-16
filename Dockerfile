FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=80

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG MODEL_VERSION=v0.2
ENV MODEL_VERSION=${MODEL_VERSION}

COPY app.py .
COPY models/ ./models/

RUN case "$MODEL_VERSION" in v0.1|v0.2) ;; *) exit 1 ;; esac \
    && test -f "models/model_${MODEL_VERSION}.joblib"

EXPOSE 80

CMD ["sh", "-c", "exec gunicorn --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} app:app"]
