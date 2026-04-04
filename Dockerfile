FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the API needs
COPY api.py .
COPY static/ static/
COPY inference/sleep_best_ckpt.keras inference/sleep_best_ckpt.keras
COPY regression_model.pkl .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300"]
