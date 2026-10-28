FROM mcr.microsoft.com/playwright/python:v1.59.0-noble
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install --break-system-packages -r requirements.txt
WORKDIR /app
COPY bench_lite.py .
CMD ["python", "bench_lite.py"]
