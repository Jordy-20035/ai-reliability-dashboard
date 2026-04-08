FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data

RUN mkdir -p /app/artifacts

EXPOSE 8000

CMD ["python", "-m", "src.api", "--host", "0.0.0.0", "--port", "8000"]
