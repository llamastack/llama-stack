# Production Deployment: Multiple Llama Stack Servers

A production-focused guide for deploying multiple Llama Stack servers using container images and systemd services.

## Table of Contents

1. [Production Use Cases](#production-use-cases)
2. [Container-Based Deployment](#container-based-deployment)
3. [Systemd Service Deployment](#systemd-service-deployment)
4. [Docker Compose Deployment](#docker-compose-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Load Balancing & High Availability](#load-balancing--high-availability)
7. [Monitoring & Logging](#monitoring--logging)
8. [Production Best Practices](#production-best-practices)

---

## Production Use Cases

### When to Deploy Multiple Llama Stack Servers

**Provider Isolation**: Separate servers for different AI providers (local vs. cloud)
```
Server 1: Ollama + local models (internal traffic)
Server 2: OpenAI + Anthropic (external API traffic)
Server 3: Enterprise providers (Bedrock, Azure)
```

**Workload Segmentation**: Different servers for different workloads
```
Server 1: Real-time inference (low latency)
Server 2: Batch processing (high throughput)
Server 3: Embeddings & vector operations
```

**Multi-Tenancy**: Isolated servers per tenant/environment
```
Server 1: Production tenant A
Server 2: Production tenant B
Server 3: Staging environment
```

**High Availability**: Load-balanced instances for fault tolerance
```
Server 1-3: Same config, load balanced
Server 4-6: Backup cluster
```

---

## Container-Based Deployment

### Method 1: Docker Containers

#### Use Official LlamaStack Container Approach

**Option 1: Use Starter Distribution with Container Runtime**


```dockerfile
# Simple Dockerfile leveraging the starter distribution
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install LlamaStack
RUN pip install --no-cache-dir llama-stack

# Initialize starter distribution
RUN llama stack build --template starter --name production-server

# Create non-root user
RUN useradd -r -s /bin/false -m llamastack
USER llamastack

WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8321/v1/health || exit 1

# Use starter distribution configs
CMD ["llama", "stack", "run", "~/.llama/distributions/starter/starter-run.yaml"]
```

**Option 2: Use Standard Python Base Image (Recommended)**

Since LlamaStack doesn't provide a Dockerfile, use the standard Python installation approach:

```bash
# Use the simple approach with standard Python image
# No need to clone and build - just use pip install directly in containers
```

**Option 2b: Check for Official Images (Future)**

```bash
# Check if official images become available
docker search meta-llama/llama-stack
docker search llamastack

# For now, use the pip-based approach in Option 1 or 3
```

**Option 3: Lightweight Container with Starter Distribution**

```dockerfile
FROM python:3.12-alpine

# Install dependencies
RUN apk add --no-cache curl gcc musl-dev linux-headers

# Install LlamaStack
RUN pip install --no-cache-dir llama-stack

# Initialize starter distribution
RUN llama stack build --template starter --name starter

# Create non-root user
RUN adduser -D llamastack
USER llamastack

WORKDIR /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8321/v1/health || exit 1

# Use CLI port override instead of modifying YAML
CMD ["llama", "stack", "run", "/home/llamastack/.llama/distributions/starter/starter-run.yaml", "--port", "8321"]
```

#### Prepare Server Configurations

**Server 1 Config (server1.yaml):**
```yaml
version: 2
image_name: production-server1
providers:
  inference:
    - provider_id: ollama
      provider_type: remote::ollama
      config:
        url: http://ollama-service:11434
  vector_io:
    - provider_id: faiss
      provider_type: inline::faiss
      config:
        kvstore:
          type: sqlite
          db_path: /data/server1/faiss_store.db
metadata_store:
  type: sqlite
  db_path: /data/server1/registry.db
server:
  port: 8321
```

**Server 2 Config (server2.yaml):**
```yaml
version: 2
image_name: production-server2
providers:
  inference:
    - provider_id: openai
      provider_type: remote::openai
      config:
        api_key: ${OPENAI_API_KEY}
    - provider_id: anthropic
      provider_type: remote::anthropic
      config:
        api_key: ${ANTHROPIC_API_KEY}
metadata_store:
  type: sqlite
  db_path: /data/server2/registry.db
server:
  port: 8322
```

#### Build and Run Containers

```bash
# Build images
docker build -t llamastack-server1 -f Dockerfile.server1 .
docker build -t llamastack-server2 -f Dockerfile.server2 .

# Create volumes for persistent data
docker volume create llamastack-server1-data
docker volume create llamastack-server2-data

# Run Server 1
docker run -d \
  --name llamastack-server1 \
  --restart unless-stopped \
  -p 8321:8321 \
  -v llamastack-server1-data:/data/server1 \
  -e GROQ_API_KEY="${GROQ_API_KEY}" \
  --network llamastack-network \
  llamastack-server1

# Run Server 2
docker run -d \
  --name llamastack-server2 \
  --restart unless-stopped \
  -p 8322:8322 \
  -v llamastack-server2-data:/data/server2 \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  --network llamastack-network \
  llamastack-server2
```


## Docker Compose Deployment

### Method 2: Docker Compose (Recommended)

**docker-compose.yml:**
```yaml
# Note: Version specification is optional in modern Docker Compose
# Using latest Docker Compose format
services:
  # Ollama service for local models
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-service
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    networks:
      - llamastack-network

  # Server 1: Local providers
  llamastack-server1:
    build:
      context: .
      dockerfile: Dockerfile.server1
    container_name: llamastack-server1
    restart: unless-stopped
    ports:
      - "8321:8321"
    volumes:
      - server1-data:/data/server1
      - ./configs/server1.yaml:/app/configs/server.yaml:ro
    environment:
      - OLLAMA_URL=http://ollama:11434
      - GROQ_API_KEY=${GROQ_API_KEY}
    depends_on:
      - ollama
    networks:
      - llamastack-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8321/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Server 2: Cloud providers
  llamastack-server2:
    build:
      context: .
      dockerfile: Dockerfile.server2
    container_name: llamastack-server2
    restart: unless-stopped
    ports:
      - "8322:8322"
    volumes:
      - server2-data:/data/server2
      - ./configs/server2.yaml:/app/configs/server.yaml:ro
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GROQ_API_KEY=${GROQ_API_KEY}
    networks:
      - llamastack-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8322/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Load balancer (optional)
  nginx:
    image: nginx:alpine
    container_name: llamastack-lb
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - llamastack-server1
      - llamastack-server2
    networks:
      - llamastack-network

volumes:
  ollama-data:
  server1-data:
  server2-data:

networks:
  llamastack-network:
    driver: bridge
```

**Environment file (.env):**
```bash
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GROQ_API_KEY=your_groq_key_here
```

**Deploy with Docker Compose:**
```bash
# Start all services
docker-compose up -d

# Scale services
docker-compose up -d --scale llamastack-server1=3

# View logs
docker-compose logs -f llamastack-server1
docker-compose logs -f llamastack-server2

# Stop services
docker-compose down

# Update services
docker-compose pull && docker-compose up -d
```

---

## Kubernetes Deployment

### Method 3: Kubernetes

**ConfigMap (llamastack-configs.yaml):**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llamastack-configs
data:
  server1.yaml: |
    version: 2
    image_name: k8s-server1
    providers:
      inference:
        - provider_id: ollama
          provider_type: remote::ollama
          config:
            url: http://ollama-service:11434
    metadata_store:
      type: sqlite
      db_path: /data/registry.db
    server:
      port: 8321

  server2.yaml: |
    version: 2
    image_name: k8s-server2
    providers:
      inference:
        - provider_id: openai
          provider_type: remote::openai
          config:
            api_key: ${OPENAI_API_KEY}
    metadata_store:
      type: sqlite
      db_path: /data/registry.db
    server:
      port: 8322
```

**Deployments (llamastack-deployments.yaml):**
```yaml
# Server 1 Deployment - Local Providers
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llamastack-server1
  labels:
    app.kubernetes.io/name: llamastack-server1
    app.kubernetes.io/component: inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llamastack-server1
  template:
    metadata:
      labels:
        app: llamastack-server1
    spec:
      containers:
      - name: llamastack
        image: llamastack-server1:latest
        ports:
        - containerPort: 8321
          name: http-api
        env:
        - name: GROQ_API_KEY
          valueFrom:
            secretKeyRef:
              name: llamastack-secrets
              key: groq-api-key
        - name: OLLAMA_URL
          value: "http://ollama-service:11434"
        volumeMounts:
        - name: config
          mountPath: /app/configs
          subPath: server1.yaml
        - name: data
          mountPath: /data
        livenessProbe:
          httpGet:
            path: /v1/health
            port: 8321
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /v1/health
            port: 8321
          initialDelaySeconds: 10
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: config
        configMap:
          name: llamastack-configs
      - name: data
        persistentVolumeClaim:
          claimName: llamastack-server1-pvc

---
# Server 2 Deployment - Cloud Providers
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llamastack-server2
  labels:
    app.kubernetes.io/name: llamastack-server2
    app.kubernetes.io/component: inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llamastack-server2
  template:
    metadata:
      labels:
        app: llamastack-server2
    spec:
      containers:
      - name: llamastack
        image: llamastack-server2:latest
        ports:
        - containerPort: 8322
          name: http-api
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llamastack-secrets
              key: openai-api-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: llamastack-secrets
              key: anthropic-api-key
        - name: GROQ_API_KEY
          valueFrom:
            secretKeyRef:
              name: llamastack-secrets
              key: groq-api-key
        volumeMounts:
        - name: config
          mountPath: /app/configs
          subPath: server2.yaml
        - name: data
          mountPath: /data
        livenessProbe:
          httpGet:
            path: /v1/health
            port: 8322
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /v1/health
            port: 8322
          initialDelaySeconds: 10
          periodSeconds: 5
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "250m"
      volumes:
      - name: config
        configMap:
          name: llamastack-configs
      - name: data
        persistentVolumeClaim:
          claimName: llamastack-server2-pvc
```

**Services (llamastack-services.yaml):**
```yaml
# Server 1 Service
apiVersion: v1
kind: Service
metadata:
  name: llamastack-server1-service
  labels:
    app.kubernetes.io/name: llamastack-server1
spec:
  selector:
    app: llamastack-server1
  ports:
  - port: 8321
    targetPort: 8321
    protocol: TCP
    name: http-api
  type: LoadBalancer

---
# Server 2 Service
apiVersion: v1
kind: Service
metadata:
  name: llamastack-server2-service
  labels:
    app.kubernetes.io/name: llamastack-server2
spec:
  selector:
    app: llamastack-server2
  ports:
  - port: 8322
    targetPort: 8322
    protocol: TCP
    name: http-api
  type: LoadBalancer
```

**Persistent Volume Claims (llamastack-pvc.yaml):**
```yaml
# PVC for Server 1
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: llamastack-server1-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd

---
# PVC for Server 2
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: llamastack-server2-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: fast-ssd
```

**Secrets (llamastack-secrets.yaml):**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: llamastack-secrets
type: Opaque
stringData:
  groq-api-key: "your_groq_api_key_here"
  openai-api-key: "your_openai_api_key_here"
  anthropic-api-key: "your_anthropic_api_key_here"
```

**Deploy to Kubernetes:**
```bash
# Apply configurations
kubectl apply -f llamastack-configs.yaml
kubectl apply -f llamastack-secrets.yaml
kubectl apply -f llamastack-pvc.yaml
kubectl apply -f llamastack-deployment.yaml
kubectl apply -f llamastack-service.yaml

# Check status
kubectl get pods -l app=llamastack-server1
kubectl get services

# Scale deployment
kubectl scale deployment llamastack-server1 --replicas=5

# View logs
kubectl logs -f deployment/llamastack-server1
```

---

## Load Balancing & High Availability

### NGINX Load Balancer Configuration

**nginx.conf:**
```nginx
upstream llamastack_local {
    least_conn;
    server llamastack-server1:8321 max_fails=3 fail_timeout=30s;
    server llamastack-server1-2:8321 max_fails=3 fail_timeout=30s;
    server llamastack-server1-3:8321 max_fails=3 fail_timeout=30s;
}

upstream llamastack_cloud {
    least_conn;
    server llamastack-server2:8322 max_fails=3 fail_timeout=30s;
    server llamastack-server2-2:8322 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name localhost;

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }

    # Route to local providers
    location /v1/local/ {
        rewrite ^/v1/local/(.*) /v1/$1 break;
        proxy_pass http://llamastack_local;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Route to cloud providers
    location /v1/cloud/ {
        rewrite ^/v1/cloud/(.*) /v1/$1 break;
        proxy_pass http://llamastack_cloud;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Default routing
    location /v1/ {
        proxy_pass http://llamastack_local;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

---

## Monitoring & Logging

### Prometheus Monitoring

**prometheus.yml:**
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'llamastack-servers'
    static_configs:
      - targets: ['llamastack-server1:8321', 'llamastack-server2:8322']
    metrics_path: /metrics
    scrape_interval: 30s
```

### Grafana Dashboard

**Key metrics to monitor:**
- Request latency (p50, p95, p99)
- Request rate (requests/second)
- Error rate (4xx, 5xx responses)
- Container resource usage (CPU, memory)
- Provider-specific metrics (API quotas, rate limits)

### Centralized Logging

**docker-compose.yml addition:**
```yaml
  # ELK Stack for logging
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    ports:
      - "5601:5601"
```

*This guide focuses on production deployments and operational best practices for multiple Llama Stack servers.*
