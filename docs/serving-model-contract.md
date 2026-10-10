# Text serving and tool calls

EIE's `/v1/chat/completions` route serves text chat through the loaded GGUF's
chat template. Function calling uses the model's native tool format and returns
OpenAI-style `tool_calls`. EIE does not execute the functions: the API client
runs a returned call, then sends its result in the next request.

## Tool calling

Send function definitions in `tools`, with each function's name, optional
description and JSON Schema `parameters`. `tool_choice` accepts `"auto"`
(default), `"none"`, `"required"`, or an OpenAI-style named function choice:

```json
{"type":"function","function":{"name":"get_weather"}}
```

`"auto"` lets the model answer or call a tool. `"none"` requests a text answer
without a tool call. `"required"` requests at least one call. A named choice
limits the request to that function. The server validates the requested name
against `tools`. Model output can still be incomplete or malformed, so clients
should check `finish_reason` and the returned call before executing it.

For example, with a model loaded under the alias `model`:

```bash
curl http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "model",
    "messages": [{"role": "user", "content": "What is the weather in Lisbon?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto",
    "max_tokens": 1024
  }'
```

When the model calls the function, the relevant part of the response has this
shape (names and arguments depend on model output):

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "id": "call-example",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\":\"Lisbon\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

`function.arguments` is a JSON **string**. The actual response also contains
the usual ID, model and usage fields. To continue, append the assistant
message containing `tool_calls`, followed by the result with `role: "tool"`
and the matching `tool_call_id`:

```json
{
  "model": "model",
  "messages": [
    {"role": "user", "content": "What is the weather in Lisbon?"},
    {"role": "assistant", "content": "", "tool_calls": [{
      "id": "call-example", "type": "function",
      "function": {"name": "get_weather", "arguments": "{\"city\":\"Lisbon\"}"}
    }]},
    {"role": "tool", "tool_call_id": "call-example", "content": "{\"temperature_c\":22}"}
  ],
  "tools": [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {
      "city": {"type": "string"}
    }, "required": ["city"]}
  }}]
}
```

The same `tools` array should be available on the follow-up if another call
may be needed. Preserve the assistant's returned call ID when sending the
tool result. If the response includes `reasoning_content`, preserve that field
in the assistant message too. EIE formats this history with the model's
template; the client remains responsible for validating arguments and running
the function.

### Streaming

With `stream: true` and `tools`, EIE buffers generation until it can parse the
complete model message. It then emits the parsed `delta.tool_calls` (or
`delta.content` if no call was made), followed by a final chunk with
`finish_reason` and `[DONE]`. This route does not send token-by-token tool
argument fragments. A slow model can therefore produce no SSE data until
generation finishes. The ordinary text-only streaming route keeps its
incremental content deltas.

### Model and compatibility limits

The loaded GGUF must have a usable chat template for its tool syntax. A tool
request alone cannot make an arbitrary model call a function. A call is only
confirmed when the response contains a parsed `tool_calls` entry; an HTTP 200
or text mentioning a function is insufficient. `finish_reason: "length"`
indicates the output limit was reached, including when a call was cut off.

Tool declaration, call and result syntax differs by model. EIE uses the
GGUF's template and the selected llama.cpp runtime parser to map native
output into API fields. Model-specific behavior, sampling settings and output
quality require a separate qualification with that model and build.

## Validation

The [serving tests](../tests/serving/README.md) check HTTP request validation,
tool schema forwarding, structured buffered and SSE responses, and a follow-up
request containing a tool result. The fake-model fixture verifies the API
contract but cannot establish whether a GGUF will choose a function. For a
specific model, run a real EIE request with `tools`, inspect
`choices[0].message.tool_calls` and `finish_reason`, execute a safe test
function, and send a `role: "tool"` follow-up. Compare the same GGUF and prompt
with `llama-server --jinja` when qualifying model behavior.
