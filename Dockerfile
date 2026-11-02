FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/features.py src/monitor.py src/
COPY service/ service/
COPY models/ models/
ENV LOG_DB=/data/predictions.db FEATURE_DB=/data/ops.db
VOLUME /data
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
