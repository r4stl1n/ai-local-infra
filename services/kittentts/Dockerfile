FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=600 --retries=3 -r requirements.txt

COPY app/ app/

RUN if ! getent group 1000 >/dev/null; then groupadd -g 1000 app; fi && \
    if ! getent passwd 1000 >/dev/null; then useradd -u 1000 -g 1000 -m app; fi
USER 1000:1000

EXPOSE 8100

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
