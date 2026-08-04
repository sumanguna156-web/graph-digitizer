FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr poppler-utils libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py digitizer.py ./
COPY static ./static
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
