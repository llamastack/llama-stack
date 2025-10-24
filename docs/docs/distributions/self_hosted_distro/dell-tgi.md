---
orphan: true
---
# Dell-TGI Distribution

```{toctree}
:maxdepth: 2
:hidden:

self
```

The `llamastack/distribution-tgi` distribution consists of the following provider configurations.


| **API**         	| **Inference** 	| **Agents**     	| **Memory**                                       	| **Safety**     	|
|-----------------	|---------------	|----------------	|--------------------------------------------------	|----------------	|
| **Provider(s)** 	| remote::tgi   	| meta-reference 	| meta-reference, remote::pgvector, remote::chroma 	| meta-reference 	|


The only difference vs. the `tgi` distribution is that it runs the Dell-TGI server for inference.


### Start the Distribution (Single Node GPU)

> [!NOTE]
> This assumes you have access to GPU to start a TGI server with access to your GPU.

```
$ cd distributions/dell-tgi/
$ ls
compose.yaml  README.md  run.yaml
$ docker compose up
```

The script will first start up TGI server, then start up Llama Stack distribution server hooking up to the remote TGI provider for inference. You should be able to see the following outputs --
```
[text-generation-inference] | 2024-10-15T18:56:33.810397Z  INFO text_generation_router::server: router/src/server.rs:1813: Using config Some(Llama)
[text-generation-inference] | 2024-10-15T18:56:33.810448Z  WARN text_generation_router::server: router/src/server.rs:1960: Invalid hostname, defaulting to 0.0.0.0
[text-generation-inference] | 2024-10-15T18:56:33.864143Z  INFO text_generation_router::server: router/src/server.rs:2353: Connected
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://[::]:8321 (Press CTRL+C to quit)
```

To kill the server
```
docker compose down
```

### (Alternative) Dell-TGI server + llama stack run (Single Node GPU)

#### Start Dell-TGI server locally
```
docker run -it --pull always --shm-size 1g -p 80:80 --gpus 4 \
-e NUM_SHARD=4
-e MAX_BATCH_PREFILL_TOKENS=32768 \
-e MAX_INPUT_TOKENS=8000 \
-e MAX_TOTAL_TOKENS=8192 \
registry.dell.huggingface.co/enterprise-dell-inference-meta-llama-meta-llama-3.1-8b-instruct
```


#### Start Llama Stack server pointing to TGI server

```
docker run --pull always --network host -it -p 8321:8321 -v ./run.yaml:/root/my-run.yaml --gpus=all llamastack/distribution-tgi --yaml_config /root/my-run.yaml
```

Make sure in you `run.yaml` file, you inference provider is pointing to the correct TGI server endpoint. E.g.
```
inference:
  - provider_id: tgi0
    provider_type: remote::tgi
    config:
      url: http://127.0.0.1:5009
```
