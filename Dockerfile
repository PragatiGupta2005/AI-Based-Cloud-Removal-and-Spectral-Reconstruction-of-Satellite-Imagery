# CloudClear AI production image (audit Sec.18)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY fetch_satellite_data.py train_models.py ./
COPY models ./models
COPY data ./data
ENV AUTH_ENABLED=false
ENV CORS_ALLOW_ORIGINS=*
ENV MAX_UPLOAD_MB=500
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
