FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN python -m pip install --no-cache-dir .

RUN chown -R user:user /app

USER user

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
