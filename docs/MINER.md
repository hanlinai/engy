# engy-miner — running a verifiable-inference miner

The miner in [`miner/`](../miner/) serves buyer requests routed from a gateway on
a local sglang serve and returns, with each completion, a compact **TOPLOC**
proof that the response really came from the model it claims to run. A validator
recomputes and compares later.

One process does everything — connect, generate, prove, answer. No local HTTP
endpoint and no extra hops.

---

## 1. How it works

```
 gateway ──websocket──► engy_miner.py ──input_ids──► sglang serve /generate :8000
                            ▲                             │  return_hidden_states
   read fin from the ───────┘                             └─► in the HTTP response body
   response, build the proof, reply completion+proof
```

1. The miner **tokenizes** the chat `messages` locally (it loads the checkpoint's
   tokenizer) and calls the serve's **`/generate`** with the `input_ids` and
   `return_hidden_states`.
2. It reads the per-token final hidden state (`fin`) back **from the HTTP
   response** and builds the TOPLOC proof — a locality-sensitive hash of the
   activations.
3. The completion plus the proof go back over the same websocket connection.

### Files

| file | what |
|---|---|
| `miner/engy_miner.py` | the whole miner in one module: dials the gateway, tokenizes, drives the serve, reads the response, builds + returns the proof |


---

## 2. Install

Plan for ~5 minutes plus model load time. 

```bash
pip install "sglang>=0.4.6"                                   # the model server
pip install toploc transformers torch numpy requests websockets   # the miner's deps
```

Then copy **`miner/engy_miner.py`** to the node — that one file is the miner.

Put the **model checkpoint** on local disk (e.g.
`/data/models/Qwen/Qwen3.6-35B-A3B-FP8`) — the miner loads its tokenizer from
there.

Then get a **miner key** from **[provider.engy.ai](https://provider.engy.ai)**. The gateway only admits registered miners; keep the key
secret — it identifies this miner.

---

## 3. Start the serve

Start an sglang serve with `--enable-return-hidden-states` — the miner needs the
activations to build the proof. That flag is the only requirement; nothing is
installed into the serve.

### Reference config — 4 × RTX 4090 (24 GB), Qwen3.6-35B-A3B-FP8

A working production configuration, not an illustration:

```bash
python -m sglang.launch_server \
  --model-path /data/models/Qwen/Qwen3.6-35B-A3B-FP8 \
  --served-model-name Qwen3.6 \
  --tp-size 4 --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.83 \
  --chunked-prefill-size 8192 \
  --max-running-requests 8 --context-length 262144 \
  --enable-return-hidden-states \
  --host 0.0.0.0 --port 8000
```

Wait until it answers:

```bash
curl -s http://127.0.0.1:8000/get_model_info >/dev/null && echo "serve up"
```

---

## 4. Start the miner (connect + declare concurrency)

Point it at your serve and checkpoint and give it your key. On connect it opens N
websocket legs and sends a HELLO — that connect+admit **is** the register.

```bash
GW=wss://<gateway-host>/gw MINER_KEY=<your-key> MODEL=qwen3.6-35b-a3b \
MAX_INFLIGHT=<serve concurrency> \
python miner/engy_miner.py \
    --checkpoint /data/models/Qwen/Qwen3.6-35B-A3B-FP8 \
    --serve-url  http://127.0.0.1:8000
```

`--serve-url` takes one URL, or several comma-separated (the miner
least-in-flight balances across them).

The gateway-host is `api.engy.ai`.

**The only capacity a miner declares is `MAX_INFLIGHT` — how many requests it can
run at once.** The request *shape* limits (max input tokens, max output tokens,
request timeout) are **the model's spec**, held by the gateway in `public.models`
and applied to every miner serving that model.

| env | set to | note |
|---|---|---|
| `MINER_KEY` | your key | required |
| `GW` | `wss://<gateway-host>/gw` | gateway websocket URL — **required**, there is no default |
| `MODEL` | e.g. `qwen3.6-35b-a3b` | the gateway's model id |
| `MAX_INFLIGHT` | `--max-running-requests` × **dp-size** | **the one number you own.** sglang's cap is **per DP replica**, so a dp=2 serve runs 2×. Under-set it and the gateway under-drives the serve; over-set it and requests queue past the model's timeout and are abandoned. |
| `ENGY_WORKER_NAME` | a name for **this machine** | only when several machines share one `MINER_KEY` — each registers as its own named worker. Default derives from hostname+model+serves. |

`MAX_INPUT_TOKENS` / `MAX_OUTPUT_TOKENS` / `MAX_REQUEST_S` are still sent in the
HELLO, but the gateway enforces `min(model spec, what you sent)` — they can only
*lower* your effective limits. Leave them at or above the model spec. Your serve
must still physically support the model's shape:
`MAX_INPUT_TOKENS + MAX_OUTPUT_TOKENS` ≤ `--context-length`.

### The model spec — `qwen3.6-35b-a3b`

The gateway holds these in `public.models` and applies them to every miner
serving the model:

| field | value | meaning |
|---|---|---|
| `max_input_tokens` | `229376` | 224K prompt ceiling |
| `max_output_tokens` | `32768` | 32K generation ceiling |
| `max_request_s` | `1800` | 30 min end-to-end deadline |
| `toploc_k` | `128` | proof top-k |
| `toploc_v` | `40` | mismatch threshold |
| `toploc_p` | `0.99` | pass ratio required |

So this model needs `--context-length 262144` (224K + 32K). The miner already
matches the 1800 s deadline by default — you do not have to set a timeout.
