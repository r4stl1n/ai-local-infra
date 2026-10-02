FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# gcc + Python headers: Triton compiles a small C launcher the first time torch
# JIT-compiles a GPU kernel, and fails ("Failed to find C compiler") without them.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3-pip python3.12-venv python3.12-dev gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages --timeout=600 --retries=3 -r requirements.txt

COPY app/ app/

RUN if ! getent group 1000 >/dev/null; then groupadd -g 1000 app; fi && \
    if ! getent passwd 1000 >/dev/null; then useradd -u 1000 -g 1000 -m app; fi
USER 1000:1000

EXPOSE 8300

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8300"]
