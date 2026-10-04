# Using Eke Hive behind LiteLLM

Eke Hive's API server speaks the OpenAI chat API (`/v1/chat/completions`, streaming and not) and the Anthropic messages API
(`/v1/messages`), so it can sit behind a gateway such as [LiteLLM](https://github.com/BerriAI/litellm) like any
OpenAI-compatible backend. This is how the maintainers run it: one gateway in front of Eke Hive and the other models they use.
Everything below applies to both model families; the differences are noted.

## What the server offers

| | |
| --- | --- |
| Base URL | `http://<host>:8430/v1` (`HIVE_PORT`; the server binds 127.0.0.1 unless `HIVE_BIND` says otherwise) |
| Model id | `hive` (`GET /v1/models`) — the server serves one model, the one it was started with |
| Authentication | none by default; with `HIVE_API_KEY` set, every endpoint except `/health` needs `Authorization: Bearer <key>` (or `x-api-key: <key>`) |
| Context | `HIVE_MAX_CTX` (262,144 in both shipped configurations). Longer prompts get HTTP 400 `context_length_exceeded` with the same wording as vLLM ("This model's maximum context length is …"), before any prefill — a 4xx, so a gateway does not retry it as a server error |
| Concurrency | `HIVE_MAX_BATCH` requests decode together (8 for DeepSeek, 4 for GLM); more wait in the daemon's queue |
| Images and video | Images on both families (`image_url` parts, `data:` URLs; http(s) URLs unless `HIVE_IMAGE_FETCH=0`); video on GLM-5.3-Flash (`video_url` parts, same URL rules) |

## Minimal LiteLLM configuration

```yaml
model_list:
  - model_name: deepseek-v4.1-flash          # the name your clients use
    litellm_params:
      model: openai/hive                     # OpenAI-compatible provider + the server's model id
      api_base: http://127.0.0.1:8430/v1
      api_key: "none"                        # or os.environ/HIVE_API_KEY when the server requires one
      timeout: 1800                          # long prompts: a 250K-token prompt took 66 s to its first token on GLM
      stream_timeout: 1800
    model_info:
      max_input_tokens: 262144
      max_output_tokens: 81920               # the gateway's own choice; the server allows the rest of the context
      supports_vision: true
      supports_function_calling: true
      supports_reasoning: true
```

Start LiteLLM with this file and call it as usual:

```bash
curl -s localhost:4000/v1/chat/completions -H 'Content-Type: application/json' -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Hello!"}]}'
```

For GLM-5.3-Flash, run a second Eke Hive instance (another port and state directory) or switch the one instance between the two
families — the gateway entry stays the same apart from the name and the concurrency note above (both take images; GLM also takes
video as a `video_url` part).

## Thinking and other Eke Hive fields through the gateway

Eke Hive reads its thinking switches from fields that are not part of the standard OpenAI request. Put them in `extra_body`
(OpenAI SDKs) or at the top level of the JSON body; LiteLLM passes them on to the backend unchanged:

| Field | Meaning |
| --- | --- |
| `chat_template_kwargs: {"enable_thinking": false}` | thinking off (DeepSeek). GLM-5.3-Flash cannot switch thinking off; it maps "off" to its lowest level `low` (or, with `HIVE_GLM_NOTHINK=empty` on the server, to an empty thinking block) |
| `chat_template_kwargs: {"enable_thinking": true, "reasoning_effort": "low"}` | thinking on at a level. DeepSeek: `minimal`, `low`, `medium`/`high` (default), `xhigh`/`max`, or 1–100. GLM: `low`, `high` (default), `max` — `medium` → `high`, `xhigh` → `max`, numbers below 50 / 90 → `low` / `high`, otherwise `max` |
| `reasoning_effort` (top level) | the same levels; `none` = thinking off. Some LiteLLM versions treat this OpenAI field specially for non-OpenAI backends — the `chat_template_kwargs` form above always arrives as sent |
| `max_thinking_tokens` or `thinking_token_budget` | a hard cap on reasoning tokens ([README](../README.md#reasoning-levels-and-budget)) |
| `hive_session_id` | optional: pins a conversation to a session for prompt-prefix reuse. Without it the server derives the session from the messages (and the `user` field), which is what most clients want |

Python example (OpenAI SDK pointed at LiteLLM):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:4000/v1", api_key=LITELLM_KEY)
r = client.chat.completions.create(
    model="deepseek-v4.1-flash",
    messages=[{"role": "user", "content": "Summarize this in three lines: ..."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"},
                "thinking_token_budget": 1024},
)
print(r.choices[0].message.content)            # the reasoning is in r.choices[0].message.reasoning_content
```

The reasoning text comes back as `reasoning_content` (streamed as it is generated), separate from `content`.

## Things that behave differently from a hosted API

- **Prompt-prefix reuse.** A follow-up turn of the same conversation reuses the saved state of everything before it (a 120K-token
  conversation answered its next turn in 0.6 s on DeepSeek, measured after a wake — [benchmarks.md](benchmarks.md#sleep-and-wake)). Usage reports it the standard way: `prompt_tokens_details.cached_tokens`
  (OpenAI) and `cache_read_input_tokens` (Anthropic). Keep the history append-only for the best reuse.
- **Long prompts take time to the first token.** Prefill runs at roughly 3,500–5,800 tok/s (DeepSeek) and 2,400–4,000 tok/s (GLM);
  set the gateway timeout for your longest prompt, and prefer streaming.
- **Sleep.** While Eke Hive sleeps (`scripts/hive-sleep.sh`), new requests wait inside the daemon until something wakes it
  (`scripts/hive-sleep.sh wake`; the wake itself takes about 3 s) — they are not rejected, and the daemon does not wake by itself. If the GPU may be lent out for long, give LiteLLM a fallback model for that deployment, or use the sleep
  hooks (`HIVE_HOOK_BEFORE_SLEEP` / `HIVE_HOOK_AFTER_WAKE`) to take the deployment out of rotation and back.
- **Retries.** A context overflow is a 400 and should not be retried. A 5xx means the daemon is restarting or failed; LiteLLM's
  default retries and cooldowns are fine for that, but a short cooldown keeps one failure from hiding the backend for long.
- **One engine, one model.** Switching between DeepSeek and GLM restarts the daemon with another configuration; if the gateway
  advertises per-model limits (`model_info` above, or an output cap in `litellm_params`), update them when you switch, since the
  two families differ (vision, the output cap you choose, how thinking is controlled).
