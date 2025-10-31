## User and assistant conversation

Here is a regular multi-turn user assistant conversation and how its formatted.

##### Input Prompt Format
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

Who are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

##### Model Response Format
```
I'm a helpful assistant, here to provide information, answer questions, and assist with tasks to the best of my abilities. I'm a large language model, which means I can understand and respond to natural language inputs, and I'm constantly learning and improving to provide more accurate and helpful responses.

I can help with a wide range of topics, from general knowledge and trivia to more specific areas like science, history, technology, and more. I can also assist with tasks like language translation, text summarization, and even generating creative content like stories or dialogues.

What can I help you with today?<|eot_id|>
```


##### Notes
This format is unchanged from Llama3.1

## User and assistant conversation with Images

This example shows how to pass and image to the model as part of the messages.

##### Input Prompt Format
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

<|image|>Describe this image in two sentences<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

##### Model Response Format
```
The image depicts a small dog standing on a skateboard, with its front paws firmly planted on the board and its back paws slightly raised. The dog's fur is predominantly brown and white, with a distinctive black stripe running down its back, and it is wearing a black collar around its neck.<|eot_id|>
```


##### Notes

- The `<|image|>` tag is used to indicate presence of the image
- The model isn't an early fusion model so doesn't actually translate an image into several tokens. Instead the cross-attention layers take input "on the side" from a vision encoder
![Image](mm-model.png)
- Its important to postion the <|image|> tag appropriately in the prompt. Image will only attend to the subsequent text tokens
- The <|image|> tag is part of the user message body, implying that it should only come after the header `<|start_header_id|>{role}<|end_header_id|>` in the message body
- We recommend using a single image in one prompt


## Builtin and Zero Shot Tool Calling


Llama3.2 vision models follow the same tool calling format as Llama3.1 models when inputs are text only.
Use `Environment: ipython` to enable tools.
Add `Tools: {{tool_name1}},{{tool_name2}}` for each of the builtin tools.
The same builtin tools as Llama3.1 are available,
- code_interpreter (for executing python code)
- brave_search (to search the web)
- wolfram_alpha (for querying wolfram alpha for mathematical questions)


##### Input Prompt Format
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Tools: brave_search, wolfram_alpha
Cutting Knowledge Date: December 2023
Today Date: 23 September 2024

You are a helpful assistant.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Search the web for the latest price of 1oz gold?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

##### Model Response Format
```
<|python_tag|>brave_search.call(query="latest price of 1oz gold")<|eom_id|>
```


##### Notes

- Note the `<|python_tag|>` before `brave_search` function call.
- The `<|eom_id|>` tag is used to indicate the end of the message.
- Similar to Llama3.1, code_interpreter is not explicitly mentioned but is enabled via `Environment: ipython`.
- Tool Calling does NOT work with images in the prompt as of now.


## Prompt format for base models


For base models (Llama3.2-11B-Vision and Llama3.2-90B-Vision), the prompt format for a simple completion is as follows


##### Input Prompt Format
```
<|begin_of_text|>The color of the sky is blue but sometimes it can also be
```

##### Model Response Format
```
 red, orange, pink, purple, and even black. The color of the sky is determined by the amount of sunlight that is scattered by the atmosphere and the amount of dust and water vapor present in the atmosphere. During sunrise and sunset, the sky can take on a range of colors due to the scattering of light by
```


##### Notes
- Same as Llama3.1

## Prompt format for base models with Image


For base models (Llama3.2-11B-Vision and Llama3.2-90B-Vision), here is an example of how the text completion format looks with an image,


##### Input Prompt Format
```
<|begin_of_text|><|image|>If I had to write a haiku for this one
```

##### Model Response Format
```
, it would be: A skateboarder's delight, a puppy on a board, a furry little thrill-seeker. This puppy is a true skateboarding enthusiast, always eager to hit the streets and show off his skills. He's a master of the board, gliding effortlessly across the pavement with grace and style.
```


##### Notes
- Note the placement of the special tags <|begin_of_text|> and <|image|>

Thank You!
