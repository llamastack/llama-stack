# iOS SDK

We offer both remote and on-device use of Llama Stack in Swift via a single SDK [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift/) that contains two components:
1. LlamaStackClient for remote
2. Local Inference for on-device

```{image} ../../../_static/remote_or_local.gif
:alt: Seamlessly switching between local, on-device inference and remote hosted inference
:width: 412px
:align: center
```

## Remote Only

If you don't want to run inference on-device, then you can connect to any hosted Llama Stack distribution with #1.

1. Add `https://github.com/meta-llama/llama-stack-client-swift/` as a Package Dependency in Xcode

2. Add `LlamaStackClient` as a framework to your app target

3. Call an API:

```swift
import LlamaStackClient

let agents = RemoteAgents(url: URL(string: "http://localhost:8321")!)
let request = Components.Schemas.CreateAgentTurnRequest(
        agent_id: agentId,
        messages: [
          .UserMessage(Components.Schemas.UserMessage(
            content: .case1("Hello Llama!"),
            role: .user
          ))
        ],
        session_id: self.agenticSystemSessionId,
        stream: true
      )

      for try await chunk in try await agents.createTurn(request: request) {
        let payload = chunk.event.payload
      // ...
```

Check out [iOSCalendarAssistant](https://github.com/meta-llama/llama-stack-client-swift/tree/main/examples/ios_calendar_assistant) for a complete app demo.

## LocalInference

LocalInference provides a local inference implementation powered by [executorch](https://github.com/pytorch/executorch/).

Llama Stack currently supports on-device inference for iOS with Android coming soon. You can run on-device inference on Android today using [executorch](https://github.com/pytorch/executorch/tree/main/examples/demo-apps/android/LlamaDemo), PyTorch’s on-device inference library.

The APIs *work the same as remote* – the only difference is you'll instead use the `LocalAgents` / `LocalInference` classes and pass in a `DispatchQueue`:

```swift
private let runnerQueue = DispatchQueue(label: "org.llamastack.stacksummary")
let inference = LocalInference(queue: runnerQueue)
let agents = LocalAgents(inference: self.inference)
```

Check out [iOSCalendarAssistantWithLocalInf](https://github.com/meta-llama/llama-stack-client-swift/tree/main/examples/ios_calendar_assistant) for a complete app demo.

### Installation

We're working on making LocalInference easier to set up. For now, you'll need to import it via `.xcframework`:

1. Clone the executorch submodule in this repo and its dependencies: `git submodule update --init --recursive`
1. Install [Cmake](https://cmake.org/) for the executorch build`
1. Drag `LocalInference.xcodeproj` into your project
1. Add `LocalInference` as a framework in your app target

### Preparing a model

1. Prepare a `.pte` file [following the executorch docs](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md#step-2-prepare-model)
2. Bundle the `.pte` and `tokenizer.model` file into your app

We now support models quantized using SpinQuant and QAT-LoRA which offer a significant performance boost (demo app on iPhone 13 Pro):


| Llama 3.2 1B | Tokens / Second (total) |  | Time-to-First-Token (sec) |  |
| :---- | :---- | :---- | :---- | :---- |
|  | Haiku | Paragraph | Haiku | Paragraph |
| BF16 | 2.2 | 2.5 | 2.3 | 1.9 |
| QAT+LoRA | 7.1 | 3.3 | 0.37 | 0.24 |
| SpinQuant | 10.1 | 5.2 | 0.2 | 0.2 |


### Using LocalInference

1. Instantiate LocalInference with a DispatchQueue. Optionally, pass it into your agents service:

```swift
  init () {
    runnerQueue = DispatchQueue(label: "org.meta.llamastack")
    inferenceService = LocalInferenceService(queue: runnerQueue)
    agentsService = LocalAgentsService(inference: inferenceService)
  }
```

2. Before making any inference calls, load your model from your bundle:

```swift
let mainBundle = Bundle.main
inferenceService.loadModel(
    modelPath: mainBundle.url(forResource: "llama32_1b_spinquant", withExtension: "pte"),
    tokenizerPath: mainBundle.url(forResource: "tokenizer", withExtension: "model"),
    completion: {_ in } // use to handle load failures
)
```

3. Make inference calls (or agents calls) as you normally would with LlamaStack:

```
for await chunk in try await agentsService.initAndCreateTurn(
    messages: [
    .UserMessage(Components.Schemas.UserMessage(
        content: .case1("Call functions as needed to handle any actions in the following text:\n\n" + text),
        role: .user))
    ]
) {
```

### Troubleshooting

If you receive errors like "missing package product" or "invalid checksum", try cleaning the build folder and resetting the Swift package cache:

(Opt+Click) Product > Clean Build Folder Immediately

```
rm -rf \
  ~/Library/org.swift.swiftpm \
  ~/Library/Caches/org.swift.swiftpm \
  ~/Library/Caches/com.apple.dt.Xcode \
  ~/Library/Developer/Xcode/DerivedData
```
