# ColabTensor

Serve **gpt-oss-120b** and **pplx-decider-v1.1** (both **Q4 GGUF**) on a
Google Colab **G4** instance (NVIDIA RTX PRO 6000 Blackwell, 96 GB), exposed
to the internet with **localtunnel**. Provisioning is driven by the
[`colab`](https://github.com/googlecolab/colab-cli) CLI; serving is done by
[`llama.cpp`](https://github.com/ggml-org/llama.cpp)'s `llama-server`.

## Models

| Model | Checkpoint | Quant | Size | Port |
|---|---|---|---|---|
| gpt-oss-120b | `unsloth/gpt-oss-120b-GGUF` (`Q4_K_M`, 2 shards) | Q4_K_M | ~63 GB | 8000 |
| pplx-decider-v1.1 | `turintech/pplx-decider-27B-Artemis-GGUF` (`Q4_K_S`) | Q4_K_S | ~15 GB | 8001 |

The decider is **pplx-decider-v1.1-27b** (via TurinTech's Artemis Q4_K_S
quantization), officially tested with `llama-server /v1/systemone`.

Both ports are exposed with `npx localtunnel --port <port>` running on the
VM; the app prints the public `https://<name>.loca.lt` URLs.

## Architecture

```
colab new --gpu G4
  └─ colab install huggingface-hub
  └─ download llama.cpp (prebuilt CUDA 12.8, ~794 MB) → /content/llama
  └─ hf download unsloth/gpt-oss-120b-GGUF Q4_K_M/* → /content/models/gpt-oss-120b
  └─ hf download turintech/pplx-decider-27B-Artemis-GGUF Q4_K_S → /content/models/pplx-decider
  └─ llama-server -m gpt-oss-120b-Q4_K_M-00001-of-00002.gguf --port 8000 -c 2048 -ngl 99
  └─ llama-server -m pplx-decider-27B-Artemis-Q4_K_S.gguf --port 8001 -c 16384 -ub 4096 -b 4096 --no-mmproj -ngl 99
  └─ npx localtunnel --port 8000   → https://<a>.loca.lt
  └─ npx localtunnel --port 8001   → https://<b>.loca.lt
```

Memory: ~63 GB + ~15 GB ≈ **78 GB** of weights — fits a single 96 GB G4
with room for KV cache.

## Prerequisites

- `colab` CLI installed (`uv tool install google-colab-cli`) and
  authenticated. The CLI defaults to OAuth; for headless use, configure
  ADC with the `colaboratory` scope (see `colab skill`).
- A Colab subscription with G4 entitlement (availability is tier-gated).
  Passing `--gpu A100` automatically requests the high-RAM shape
  (`--high-mem`, 80 GB); the default 40 GB A100 is not enough for both
  models. Requires Colab Pro or Pro+.
- Both checkpoints are public on Hugging Face; no token is required.
- `node`/`npx` on the VM (preinstalled on Colab) for localtunnel.

## Install & run

```bash
pip install -e .          # or: PYTHONPATH=Src python -m ColabTensor.Main ...

# review the commands first (no VM is provisioned, no cost):
python -m ColabTensor.Main --session colabtensor --gpu G4 --dry-run

# provision the G4, install, download, launch, tunnel, print URLs:
python -m ColabTensor.Main --session colabtensor --gpu G4

# stop and release the VM when done:
python -m ColabTensor.Main --session colabtensor --stop
```

Downloading ~78 GB of weights takes a while; the app waits for each model
to become healthy (`/health`) before printing the tunnel URLs.

## Endpoints

- **gpt-oss-120b** (llama-server, OpenAI-compatible):
  `POST https://<tunnel>.loca.lt/v1/chat/completions` with
  `"model": "gpt-oss-120b"`.
- **pplx-decider-v1.1** (llama-server, `/v1/systemone`):
  `POST https://<tunnel>.loca.lt/v1/systemone` with a JSON body
  `{"state": ..., "questions": {...}}` (see the model card for the schema).
  `GET /health` returns `{"status": "ok"}`.

## Caveats

- The decider is **pplx-decider-v1.1-27b** (via TurinTech Artemis Q4_K_S).
- llama.cpp is pinned to build `b12441` (prebuilt CUDA 12.8, supports
  A100 and Blackwell sm_120). The CUDA 13.5 build requires a newer driver
  than Colab's 580.82.07 (CUDA 13.0) and fails with "the provided PTX was
  compiled with an unsupported toolchain". The pplx-decider support was
  merged in llama.cpp in October 2026; older builds will not load the
  decider.
- localtunnel URLs are public and unauthenticated; anyone with the URL
  can call the endpoints. Stop the session when you are done.
- Provisioning a G4 and downloading ~78 GB of weights takes time and
  consumes Colab compute units.

## Development

```bash
ruff check . && ruff format --check . && pyrefly check && pytest
```
