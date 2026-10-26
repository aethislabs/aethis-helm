# Aethis Helm

Helm ranks an AI agent's available tools for its next step.

You send the task, the calls made so far, and the tools on offer. Helm reads all of it once with a local model (Qwen2.5-1.5B-Instruct plus a LoRA adapter and a span scoring head) and returns a score for every tool, best first. It never generates text, so it cannot name a tool that is not in your list.

## What it does and doesn't do

- **Ranks tool names only.** It does not fill in arguments. Your agent still has to write the call.
- **2 to 10 tools per request.** `Finish` (the "stop and answer" action) is added automatically if you leave it out, because the model was always trained with it present. The response tells you when that happened. So the model scores at most 11 tools, the most it saw in training. If you send `Finish` yourself, it counts toward the 10.
- **Descriptions are cut to 200 characters**, as in training. Long tasks and histories are trimmed from the front. If the prompt is so long that a tool's line would be cut, you get a 422 error, not a silent wrong answer.
- **Trained on ToolBench (RapidAPI-style tools, single-tool tasks).** Expect it to work best on similar tools: short REST-style functions with one-line descriptions. It has not been evaluated on other kinds of toolsets.
- **Scores are a ranking signal.** `prob` is a softmax over the tools in one request. It is not a calibrated probability.

## Quickstart

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'

cp .env.example .env          # then fill in the values below
export HELM_DEVICE=mps        # cuda, mps or cpu (cpu runs in full precision)
export HELM_HF_REPO=YashDT/aethis-helm-weights   # private: needs a token with access
export HF_TOKEN=hf_...                           # or run `hf auth login` once
# or, if you already have the checkpoint on disk:
# export HELM_WEIGHTS_DIR=/path/to/checkpoint    # contains adapter/ and head.pt

.venv/bin/helm-serve          # http://127.0.0.1:8000, one worker
```

The first start downloads the Qwen2.5-1.5B-Instruct backbone (about 3 GB, public) and the Helm weights into the Hugging Face cache. Every setting is listed in `.env.example`.

### Request

```bash
curl -s -X POST http://127.0.0.1:8000/rank -H 'content-type: application/json' -d '{
  "task": "I am flying from San Francisco to Tokyo next Friday. Find me a flight and tell me what the weather will be like when I land.",
  "history": [
    {"tool": "search_flights", "args": {"origin": "SFO", "destination": "HND", "date": "2026-10-30"},
     "result": "{\"flights\": [{\"id\": \"NH7\", \"arrive\": \"15:25+1\", \"price_usd\": 1180}]}"}
  ],
  "tools": [
    {"name": "search_flights", "description": "Search flights between two airports on a given date."},
    {"name": "get_weather_forecast", "description": "Get the daily weather forecast for a city for the next 10 days."},
    {"name": "convert_currency", "description": "Convert an amount of money between two currencies."},
    {"name": "book_hotel", "description": "Book a hotel room in a city for given check-in and check-out dates."},
    {"name": "get_exchange_rates", "description": "Latest exchange rates for a base currency."},
    {"name": "translate_text", "description": "Translate text between languages."}
  ]
}'
```

`history` is optional. Each entry is `{"tool": str, "args": object, "result": str}`.

### Response

From a real run on an Apple Silicon laptop (MPS), scores rounded:

```json
{
  "ranked": [
    {"name": "get_weather_forecast", "score": -2.05, "prob": 0.768},
    {"name": "search_flights",       "score": -4.31, "prob": 0.080},
    {"name": "Finish",               "score": -4.82, "prob": 0.048},
    {"name": "get_exchange_rates",   "score": -4.91, "prob": 0.044},
    {"name": "book_hotel",           "score": -4.91, "prob": 0.044},
    {"name": "convert_currency",     "score": -6.12, "prob": 0.013},
    {"name": "translate_text",       "score": -7.45, "prob": 0.003}
  ],
  "finish_added": true,
  "notes": ["'Finish' was added to the tool list because the model was trained with it present."],
  "latency_ms": 183.5
}
```

`GET /health` returns `{"status": "ok", "model_loaded": true, "device": "mps", "version": "0.1.0"}`.

Errors: 422 for invalid input (fewer than 2 or more than 10 tools, duplicate or multi-line names, empty task, a prompt that would cut a tool's line), 503 while the model is still loading.

## Results from the research

These come from the ActionRank research project: ToolBench G1, Qwen2.5-1.5B, 3 training runs per system, 1,352 held-out decisions that no training run was monitored against. The baseline is the usual approach: the same model fine-tuned with the same LoRA budget to write out the tool name.

| | Helm (score the tools) | Baseline (write the name) |
|---|---:|---:|
| Correct first pick | 64.0% | 66.6% |
| Correct tool in top 5 | 97.5% | 90.7% |
| Names a tool that isn't offered | 0% | 0.9% |
| Time per decision (laptop) | 247 ms | 632 ms |

The baseline is about 2.6 points more accurate on the first pick. Helm never names a tool that isn't offered, ranks the alternatives much better, and is faster. That is the trade. All numbers are for ToolBench-style tools and this backbone; they are not a claim about other toolsets or larger models.

### Measured with this code

- **Same predictions as the research.** On the 1,340 held-out steps that have no duplicate tool names, Helm's top pick matches the research checkpoint's on every step: 64.5% correct first pick and 97.7% in the top 5. (The other 12 of the 1,352 steps list a tool name twice, which Helm rejects.)
- **Latency, from `scripts/bench.py`** (server-side median, one request at a time, Apple M3 Pro):

| | 2 tools | 6 tools | 10 tools |
|---|---:|---:|---:|
| MPS (native, float16), sample request | | about 185 ms | |
| CPU in Docker (float32, 12 vCPUs, short history) | 1.5 s | 1.8 s | 2.0 s |
| CPU in Docker (float32, 12 vCPUs, 8-call history) | 2.8 s | 3.0 s | 3.0 s |

CPU works but is about 10 times slower than a GPU-class device. It also needs about 6.5 GB of RAM for the model in full precision, so give Docker at least 8 GB. Use CPU for trying Helm out, and `cuda` or `mps` for anything interactive.

## Weights

The weights are a LoRA adapter (`adapter/`, about 4 MB) and a span scoring head (`head.pt`, about 50 MB) on top of `Qwen/Qwen2.5-1.5B-Instruct`. Helm loads them from `HELM_WEIGHTS_DIR` if set, otherwise from the Hugging Face Hub repo `HELM_HF_REPO` (at `HELM_HF_REVISION`). `head.pt` is loaded with `torch.load(weights_only=True)`, so it cannot run code.

**The weights repo, `YashDT/aethis-helm-weights`, is private.** The code in this repository is public, but the server can only download the weights with a Hugging Face token that has read access to that repo. Set it as `HF_TOKEN` (in `.env` for Docker), or log in once with `hf auth login`. Without access, startup fails with a Hugging Face authorization error. The backbone itself is public and needs no token.

To publish a checkpoint to the Hub:

```bash
python scripts/upload_weights.py --src /path/to/checkpoint --repo YashDT/aethis-helm-weights --private
```

## Docker

One image runs on CPU by default and on an NVIDIA GPU with `HELM_DEVICE=cuda`:

```bash
docker build -t aethis-helm .
docker run -p 8000:8000 --env-file .env -v helm-models:/models aethis-helm
docker run -p 8000:8000 --env-file .env -e HELM_DEVICE=cuda --gpus all -v helm-models:/models aethis-helm
```

To use local weights in the container, mount them and point `HELM_WEIGHTS_DIR` at the mount.

## Development

```bash
.venv/bin/pytest                                                 # fast tests, no model download
HELM_DEVICE=mps HELM_WEIGHTS_DIR=... .venv/bin/pytest -m slow    # loads the real model
```

Layout: `src/aethis_helm/` holds `types.py` (value types), `config.py` (settings; defaults are the checkpoint's), `verbalize.py` (prompt builder, matches the training format byte for byte), `model.py` (backbone, span pooling, scoring head), `weights.py` (local or Hub weights), `ranker.py` (load and rank) and `serve.py` (FastAPI app).

The server runs one worker and serializes model calls with a lock, so it handles one decision at a time. It has no authentication or rate limiting; put it behind something that does before exposing it.

## License

Apache License 2.0, see [LICENSE](LICENSE). The backbone (Qwen2.5-1.5B-Instruct) and the training data (ToolBench) are also Apache-2.0; see [NOTICE](NOTICE) for attribution.
