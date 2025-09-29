#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Deploys the benchmark-specific components on top of the base k8s deployment (../k8s/apply.sh).

export STREAM_DELAY_SECONDS=0.005

export POSTGRES_USER=llamastack
export POSTGRES_DB=llamastack
export POSTGRES_PASSWORD=llamastack

export INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
export SAFETY_MODEL=meta-llama/Llama-Guard-3-1B

export BENCHMARK_INFERENCE_MODEL=$INFERENCE_MODEL
export LLAMA_STACK_WORKERS=4

set -euo pipefail
set -x

# Deploy benchmark-specific components
kubectl create configmap llama-stack-config --from-file=stack_run_config.yaml \
  --dry-run=client -o yaml > stack-configmap.yaml

kubectl apply --validate=false -f stack-configmap.yaml

# Deploy our custom llama stack server (overriding the base one)
envsubst < stack-k8s.yaml.template | kubectl apply --validate=false -f -
