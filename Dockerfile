FROM python:3.11-slim

LABEL io.modelcontextprotocol.server.name="io.github.sachitrafa/yourmemory"

ENV PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir yourmemory && \
    python -m spacy download en_core_web_sm && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

RUN mkdir -p /root/.yourmemory

CMD ["yourmemory", "--stdio"]
