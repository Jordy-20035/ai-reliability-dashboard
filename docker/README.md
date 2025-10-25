# Docker Deployment Guide

This directory contains Docker configuration for deploying the Trustworthy AI Monitor system.

## Quick Start

### Build and Run

```bash
cd docker
docker-compose up --build
```

### Access Services

- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501

### Stop Services

```bash
docker-compose down
```

## Services

### API Service
- **Port**: 8000
- **Description**: FastAPI backend for model serving and monitoring
- **Health Check**: http://localhost:8000/manage/health

### Dashboard Service
- **Port**: 8501
- **Description**: Streamlit dashboard for visualization and monitoring
- **Features**: Data exploration, model training, drift detection, fairness analysis

## Configuration

### Environment Variables

Create a `.env` file in the docker directory:

```env
ENVIRONMENT=production
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info
```

### Volume Mounts

The following directories are mounted:
- `../data`: Dataset storage
- `../logs`: Application logs
- `../models`: Trained models

### Resource Limits

To set resource limits, add to docker-compose.yml:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Production Deployment

### Security Considerations

1. **Change default credentials** for any databases
2. **Use secrets** instead of environment variables for sensitive data
3. **Enable HTTPS** with reverse proxy (nginx/traefik)
4. **Restrict network access** to necessary services only

### Example with nginx

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
      - dashboard
```

### Scaling

To scale the API service:

```bash
docker-compose up --scale api=3
```

## Optional Services

### Enable PostgreSQL

Uncomment the postgres service in docker-compose.yml and add connection to your config:

```python
DATABASE_URL=postgresql://admin:changeme@postgres:5432/trustworthy_ai
```

### Enable Prometheus & Grafana

Uncomment prometheus and grafana services in docker-compose.yml for advanced monitoring.

## Troubleshooting

### Check Logs

```bash
# All services
docker-compose logs

# Specific service
docker-compose logs api
docker-compose logs dashboard

# Follow logs
docker-compose logs -f api
```

### Restart Services

```bash
docker-compose restart api
docker-compose restart dashboard
```

### Rebuild After Code Changes

```bash
docker-compose up --build
```

### Clean Up

```bash
# Stop and remove containers
docker-compose down

# Remove volumes as well
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## Development Mode

For development with hot reload:

```yaml
services:
  api:
    volumes:
      - ..:/app
    environment:
      - ENVIRONMENT=development
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Health Checks

All services include health checks:

```bash
# Check API health
curl http://localhost:8000/manage/health

# Check container health
docker-compose ps
```

## Backup and Restore

### Backup Data

```bash
docker run --rm -v trustworthy-ai-monitor_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz -C /data .
```

### Restore Data

```bash
docker run --rm -v trustworthy-ai-monitor_postgres_data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres_backup.tar.gz -C /data
```

