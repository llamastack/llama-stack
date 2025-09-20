#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

# Default values
TARGET="stack"
MAX_SECONDS=30
PROMPT_TOKENS=256
OUTPUT_TOKENS=128
RATE_TYPE="sweep"
RATE="10"
VLLM_REPLICAS="1"
STACK_REPLICAS="1"

# Parse command line arguments
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -t, --target <stack|vllm>     Target to benchmark (default: stack)"
    echo "  -s, --max-seconds <seconds>   Maximum duration in seconds (default: 30)"
    echo "  -p, --prompt-tokens <tokens>  Number of prompt tokens (default: 256)"
    echo "  -o, --output-tokens <tokens>  Number of output tokens (default: 128)"
    echo "  -r, --rate-type <type>        Rate type (default: sweep)"
    echo "  -c, --rate                    Rate"
    echo "  --vllm-replicas <count>       Scale vllm-server to this many replicas"
    echo "  --stack-replicas <count>      Scale llama-stack-benchmark-server to this many replicas"
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --target vllm                              # Benchmark vLLM direct"
    echo "  $0 --target stack                             # Benchmark Llama Stack (default)"
    echo "  $0 -t vllm -s 60 -p 512 -o 256               # vLLM with custom parameters"
    echo "  $0 --target vllm --vllm-replicas 3            # Scale vLLM to 3 replicas before benchmarking"
    echo "  $0 --target stack --stack-replicas 2          # Scale stack server to 2 replicas"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -s|--max-seconds)
            MAX_SECONDS="$2"
            shift 2
            ;;
        -p|--prompt-tokens)
            PROMPT_TOKENS="$2"
            shift 2
            ;;
        -o|--output-tokens)
            OUTPUT_TOKENS="$2"
            shift 2
            ;;
        -r|--rate-type)
            RATE_TYPE="$2"
            shift 2
            ;;
        -c|--rate)
            RATE="$2"
            shift 2
            ;;
        --vllm-replicas)
            VLLM_REPLICAS="$2"
            shift 2
            ;;
        --stack-replicas)
            STACK_REPLICAS="$2"
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
    BASE_URL="http://vllm-server:8000"
    JOB_NAME="guidellm-vllm-benchmark-job"
    echo "Benchmarking vLLM direct with GuideLLM..."
else
    BASE_URL="http://llama-stack-benchmark-service:8323/v1/openai"
    JOB_NAME="guidellm-stack-benchmark-job"
    echo "Benchmarking Llama Stack with GuideLLM..."
fi

# Scale deployments if requested
if [[ "$VLLM_REPLICAS" != "" ]] || [[ "$STACK_REPLICAS" != "" ]]; then
    echo "Scaling deployments..."

    if [[ "$VLLM_REPLICAS" != "" ]]; then
        echo "Scaling vllm-server to $VLLM_REPLICAS replicas..."
        kubectl scale deployment vllm-server --replicas=$VLLM_REPLICAS
        kubectl rollout status deployment vllm-server --timeout=600s
    fi

    if [[ "$STACK_REPLICAS" != "" ]]; then
        echo "Scaling llama-stack-benchmark-server to $STACK_REPLICAS replicas..."
        kubectl scale deployment llama-stack-benchmark-server --replicas=$STACK_REPLICAS
        kubectl rollout status deployment llama-stack-benchmark-server --timeout=600s
    fi

    echo "All scaling operations completed. Waiting additional 30s for services to stabilize..."
    sleep 30
fi

echo "Configuration:"
echo "  Target: $TARGET"
echo "  Base URL: $BASE_URL"
echo "  Max seconds: ${MAX_SECONDS}s"
echo "  Prompt tokens: $PROMPT_TOKENS"
echo "  Output tokens: $OUTPUT_TOKENS"
echo "  Rate type: $RATE_TYPE"
if [[ -n "$VLLM_REPLICAS" ]]; then
    echo "  vLLM replicas: $VLLM_REPLICAS"
fi
if [[ -n "$STACK_REPLICAS" ]]; then
    echo "  Stack replicas: $STACK_REPLICAS"
fi
echo ""

# Create temporary job yaml
TEMP_YAML="/tmp/guidellm-benchmark-job-temp-$(date +%s).yaml"
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
      - name: guidellm-benchmark
        image: python:3.11-slim
        command: ["/bin/bash"]
        args:
        - "-c"
        - |
          # Install uv and guidellm
          pip install uv &&
          uv pip install --system guidellm &&

          # Login to HuggingFace
          uv pip install --system huggingface_hub &&
          python -c "from huggingface_hub import login; login(token='\$HF_TOKEN')" &&

          # Run GuideLLM benchmark and save output
          export COLUMNS=200
          GUIDELLM__PREFERRED_ROUTE="chat_completions" uv run guidellm benchmark run \\
            --target "$BASE_URL" \\
            --rate-type "$RATE_TYPE" \\
            --max-seconds $MAX_SECONDS \\
            --data "prompt_tokens=$PROMPT_TOKENS,output_tokens=$OUTPUT_TOKENS" \\
            --model "$INFERENCE_MODEL" \\
            --rate "$RATE" \\
            --warmup-percent 0.05 \\
            2>&1
        env:
        - name: INFERENCE_MODEL
          value: "meta-llama/Llama-3.2-3B-Instruct"
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token-secret
              key: token
        resources:
          requests:
            memory: "4Gi"
            cpu: "500m"
          limits:
            memory: "8Gi"
            cpu: "2000m"
      restartPolicy: Never
  backoffLimit: 3
EOF

echo "Cleaning up any existing GuideLLM benchmark job..."
kubectl delete job $JOB_NAME 2>/dev/null || true

echo "Deploying GuideLLM benchmark Job..."
kubectl apply -f "$TEMP_YAML"

echo "Waiting for job to start..."
kubectl wait --for=condition=Ready pod -l job-name=$JOB_NAME --timeout=120s

# Prepare file names and create results directory
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p results
OUTPUT_FILE="results/guidellm-benchmark-${TARGET}-${STACK_REPLICAS}-${VLLM_REPLICAS}-${TIMESTAMP}.txt"

echo "Following GuideLLM benchmark logs..."
kubectl logs -f job/$JOB_NAME

echo "Job completed. Checking final status..."
kubectl get job $JOB_NAME

# Save benchmark results using kubectl logs
echo "Saving benchmark results..."
kubectl logs job/$JOB_NAME > "$OUTPUT_FILE"

echo "Benchmark output saved to: $OUTPUT_FILE"

# Clean up temporary file
rm -f "$TEMP_YAML"
