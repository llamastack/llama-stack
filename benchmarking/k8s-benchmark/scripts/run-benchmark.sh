#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

# Default values
TARGET="stack"
DURATION=60
CONCURRENT=10

# Parse command line arguments
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -t, --target <stack|vllm>     Target to benchmark (default: stack)"
    echo "  -d, --duration <seconds>      Duration in seconds (default: 60)"
    echo "  -c, --concurrent <users>      Number of concurrent users (default: 10)"
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --target vllm              # Benchmark vLLM direct"
    echo "  $0 --target stack             # Benchmark Llama Stack (default)"
    echo "  $0 -t vllm -d 120 -c 20       # vLLM with 120s duration, 20 users"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -c|--concurrent)
            CONCURRENT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate target
if [[ "$TARGET" != "stack" && "$TARGET" != "vllm" ]]; then
    echo "Error: Target must be 'stack' or 'vllm'"
    usage
    exit 1
fi

# Set configuration based on target
if [[ "$TARGET" == "vllm" ]]; then
    BASE_URL="http://vllm-server:8000/v1"
    JOB_NAME="vllm-benchmark-job"
    echo "Benchmarking vLLM direct..."
else
    BASE_URL="http://llama-stack-benchmark-service:8323/v1/openai/v1"
    JOB_NAME="stack-benchmark-job"
    echo "Benchmarking Llama Stack..."
fi

echo "Configuration:"
echo "  Target: $TARGET"
echo "  Base URL: $BASE_URL"
echo "  Duration: ${DURATION}s"
echo "  Concurrent users: $CONCURRENT"
echo ""

# Create temporary job yaml
TEMP_YAML="/tmp/benchmark-job-temp-$(date +%s).yaml"
cat > "$TEMP_YAML" << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: $JOB_NAME
  namespace: default
spec:
  template:
    spec:
      containers:
      - name: benchmark
        image: python:3.11-slim
        command: ["/bin/bash"]
        args:
        - "-c"
        - |
          pip install aiohttp &&
          python3 /benchmark/benchmark.py \\
            --base-url $BASE_URL \\
            --model \${INFERENCE_MODEL} \\
            --duration $DURATION \\
            --concurrent $CONCURRENT
        env:
        - name: INFERENCE_MODEL
          value: "meta-llama/Llama-3.2-3B-Instruct"
        volumeMounts:
        - name: benchmark-script
          mountPath: /benchmark
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
      volumes:
      - name: benchmark-script
        configMap:
          name: benchmark-script
      restartPolicy: Never
  backoffLimit: 3
EOF

echo "Creating benchmark ConfigMap..."
kubectl create configmap benchmark-script \
  --from-file=benchmark.py=benchmark.py \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Cleaning up any existing benchmark job..."
kubectl delete job $JOB_NAME 2>/dev/null || true

echo "Deploying benchmark Job..."
kubectl apply -f "$TEMP_YAML"

echo "Waiting for job to start..."
kubectl wait --for=condition=Ready pod -l job-name=$JOB_NAME --timeout=60s

echo "Following benchmark logs..."
kubectl logs -f job/$JOB_NAME

echo "Job completed. Checking final status..."
kubectl get job $JOB_NAME

# Clean up temporary file
rm -f "$TEMP_YAML"
