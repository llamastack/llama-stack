## Kubernetes Deployment Guide

Instead of starting the Llama Stack and vLLM servers locally. We can deploy them in a Kubernetes cluster.

### Prerequisites
In this guide, we'll use a local [Kind](https://kind.sigs.k8s.io/) cluster and a vLLM inference service in the same cluster for demonstration purposes.

Note: You can also deploy the Llama Stack server in an AWS EKS cluster. See [Deploying Llama Stack Server in AWS EKS](#deploying-llama-stack-server-in-aws-eks) for more details.

First, create a local Kubernetes cluster via Kind:

```
kind create cluster --image kindest/node:v1.32.0 --name llama-stack-test
```

First set your hugging face token as an environment variable.
```
export HF_TOKEN=$(echo -n "your-hf-token" | base64)
```

Now create a Kubernetes PVC and Secret for downloading and storing Hugging Face model:

```
cat <<EOF |kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-models
spec:
  accessModes:
    - ReadWriteOnce
  volumeMode: Filesystem
  resources:
    requests:
      storage: 50Gi
---
apiVersion: v1
kind: Secret
metadata:
  name: hf-token-secret
type: Opaque
data:
  token: $HF_TOKEN
EOF
```


Next, start the vLLM server as a Kubernetes Deployment and Service:

```
cat <<EOF |kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: vllm
  template:
    metadata:
      labels:
        app.kubernetes.io/name: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command: ["/bin/sh", "-c"]
        args: [
          "vllm serve meta-llama/Llama-3.2-1B-Instruct"
        ]
        env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token-secret
              key: token
        ports:
          - containerPort: 8000
        volumeMounts:
          - name: llama-storage
            mountPath: /root/.cache/huggingface
      volumes:
      - name: llama-storage
        persistentVolumeClaim:
          claimName: vllm-models
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-server
spec:
  selector:
    app.kubernetes.io/name: vllm
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP
EOF
```

We can verify that the vLLM server has started successfully via the logs (this might take a couple of minutes to download the model):

```
$ kubectl logs -l app.kubernetes.io/name=vllm
...
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Then we can modify the Llama Stack run configuration YAML with the following inference provider:

```yaml
providers:
  inference:
  - provider_id: vllm
    provider_type: remote::vllm
    config:
      url: http://vllm-server.default.svc.cluster.local:8000/v1
      max_tokens: 4096
      api_token: fake
```

Once we have defined the run configuration for Llama Stack, we can build an image with that configuration and the server source code:

```
tmp_dir=$(mktemp -d) && cat >$tmp_dir/Containerfile.llama-stack-run-k8s <<EOF
FROM distribution-myenv:dev

RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/meta-llama/llama-stack.git /app/llama-stack-source

ADD ./vllm-llama-stack-run-k8s.yaml /app/config.yaml
EOF
podman build -f $tmp_dir/Containerfile.llama-stack-run-k8s -t llama-stack-run-k8s $tmp_dir
```

### Deploying Llama Stack Server in Kubernetes

We can then start the Llama Stack server by deploying a Kubernetes Pod and Service:

```
cat <<EOF |kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: llama-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-stack-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: llama-stack
  template:
    metadata:
      labels:
        app.kubernetes.io/name: llama-stack
    spec:
      containers:
      - name: llama-stack
        image: localhost/llama-stack-run-k8s:latest
        imagePullPolicy: IfNotPresent
        command: ["python", "-m", "llama_stack.core.server.server", "--config", "/app/config.yaml"]
        ports:
          - containerPort: 5000
        volumeMounts:
          - name: llama-storage
            mountPath: /root/.llama
      volumes:
      - name: llama-storage
        persistentVolumeClaim:
          claimName: llama-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: llama-stack-service
spec:
  selector:
    app.kubernetes.io/name: llama-stack
  ports:
  - protocol: TCP
    port: 5000
    targetPort: 5000
  type: ClusterIP
EOF
```

### Verifying the Deployment
We can check that the LlamaStack server has started:

```
$ kubectl logs -l app.kubernetes.io/name=llama-stack
...
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     ASGI 'lifespan' protocol appears unsupported.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://['::', '0.0.0.0']:5000 (Press CTRL+C to quit)
```

Finally, we forward the Kubernetes service to a local port and test some inference requests against it via the Llama Stack Client:

```
kubectl port-forward service/llama-stack-service 5000:5000
llama-stack-client --endpoint http://localhost:5000 inference chat-completion --message "hello, what model are you?"
```

## Deploying Llama Stack Server in AWS EKS

We've also provided a script to deploy the Llama Stack server in an AWS EKS cluster.

Prerequisites:
- Set up an [EKS cluster](https://docs.aws.amazon.com/eks/latest/userguide/getting-started.html).
- Create a [Github OAuth app](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/creating-an-oauth-app) and get the client ID and client secret.
  - Set the `Authorization callback URL` to `http://<your-llama-stack-ui-url>/api/auth/callback/`


Run the following script to deploy the Llama Stack server:
```
export HF_TOKEN=<your-huggingface-token>
export GITHUB_CLIENT_ID=<your-github-client-id>
export GITHUB_CLIENT_SECRET=<your-github-client-secret>
export LLAMA_STACK_UI_URL=<your-llama-stack-ui-url>

cd docs/source/distributions/eks
./apply.sh
```

This script will:

- Set up a default storage class for AWS EKS
- Deploy the Llama Stack server in a Kubernetes Pod and Service