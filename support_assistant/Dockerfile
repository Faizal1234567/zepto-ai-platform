FROM python:3.12-slim

WORKDIR /app
COPY support_assistant/requirements.txt /app/requirements.txt
# Use CPU-only PyTorch: this RAG service does not need GPU/CUDA libraries.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir -r requirements.txt
COPY support_assistant /app

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
