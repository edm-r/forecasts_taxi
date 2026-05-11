FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install mlflow psycopg2-binary

WORKDIR /mlflow

CMD ["mlflow", "server"]
