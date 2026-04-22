# ID Verification Service — Deployment Guide

## Overview

The ID verification service is a single Docker container microservice using a tiered DoB extraction pipeline. Donut is used only as an escalation path — the majority of requests are handled by PaddleOCR alone, dramatically reducing compute requirements.

- **Image:** built from `Dockerfile` (CPU) or `Dockerfile.gpu` (GPU)
- **Approximate image size:**
  - CPU image: ~2.5–3.5 GB (CPU torch + paddlepaddle + dependencies — model weights not baked in)
  - GPU image: ~5–6 GB (CUDA torch + paddlepaddle + dependencies — model weights not baked in)
- **Process:** `uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1`

---

## CPU vs GPU Deployment

| | `Dockerfile` (CPU) | `Dockerfile.gpu` (GPU) |
|---|---|---|
| Base image | `python:3.11-slim-bullseye` | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` |
| PyTorch build | CPU wheel (`whl/cpu`) | CUDA 12.1 wheel (`whl/cu121`) |
| Donut dtype | `torch.float32` | `torch.float16` |
| PaddleOCR | CPU-only | CPU-only |
| `flash_attn` stub | Not needed | Not needed |
| Host requirements | None | NVIDIA drivers + NVIDIA Container Toolkit |
| Tier 1 inference time | < 1s | < 1s |
| Tier 2 inference time (escalation) | 4–10 seconds | 1–2 seconds |
| Request time (Tier 1 — ~70–90%) | < 2s | < 2s |
| Request time (Tier 2 — ~10–30%) | 5–12s | 2–4s |

`services/donut_service.py` auto-detects `torch.cuda.is_available()` at startup — no code change needed between CPU and GPU deployments.

> **PaddleOCR note:** PaddleOCR runs CPU-only in both images. Only Donut (torch-based) benefits from GPU, and it is invoked significantly less often than before due to the tiered architecture.

---

## Resource Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| vCPU | 2 | 4 |
| RAM | 4 GB | 8 GB |
| Disk (image + volumes) | 8 GB | 12 GB |
| GPU (optional) | — | NVIDIA T4 / L4 / A10G |

**RAM floor breakdown (Donut):**

| Component | Memory |
|---|---|
| Donut weights (FP32, CPU) | ~1.0 GB |
| Donut inference peak | ~0.5–1 GB |
| PaddleOCR model weights | ~0.3 GB |
| PaddleOCR inference peak | ~0.5 GB |
| OS + uvicorn + FastAPI runtime | ~1–2 GB |
| Image buffer per request | ~0.1 GB |
| **Total floor** | **~3.5–5 GB** |

> The Donut migration reduces RAM requirements substantially compared to Florence-2-large-ft (which required ~7–8 GB). A modest 4 GB machine is now viable for low-traffic deployments.

---

## Health Check

Configure your hosting platform to poll:

```
GET /health  →  HTTP 200
```

Expected response:
```json
{
    "status": "ok",
    "model": "Donut-base (escalation only)",
    "paddle_model": "PaddleOCR-en",
    "device": "cpu"
}
```

**Startup grace period:**
- CPU: ~5–15s (Donut is ~500 MB vs Florence-2's 1.54 GB). Set an initial delay of **30 seconds** to be safe.
- GPU: ~3–6s. **15 seconds** initial delay is sufficient.

---

## Scaling

- **Stateless:** No session or user data held in memory. Scale horizontally.
- **Memory Bound:** Each worker requires ~4–5 GB RAM (CPU). Scaling by adding containers (not workers per container) is safer.

---

## Environment Variables

| Variable | Default | Notes |
|---|---|---|
| `ENABLE_SWAGGER` | `true` | Set to `false` to disable Swagger and the `/debug/*` route |
| `HF_HOME` | `/data/models/huggingface` | Set in Dockerfile — do not change without updating compose mounts |

---

## Security

- **Multipart Uploads:** Only JPEG, PNG, and WEBP should be accepted.
- **Statelessness:** All uploaded images are written to `/tmp`, processed, and immediately unlinked in a `finally` block. No images are persisted.
- **Network Isolation:** Ensure port 8001 is only accessible to the NestJS container.

---

## Deployment Checklist

- [ ] `ENABLE_SWAGGER=false` is set in production
- [ ] `/health` returns 200 after startup with `"model": "Donut-base (escalation only)"`
- [ ] NestJS timeout is set to 60s (CPU) or 30s (GPU)
- [ ] Named volumes (`donut_hf_cache`, `donut_paddle_cache`) are created or already exist
- [ ] On first startup, internet access is available to download models (~550 MB total)
- [ ] GPU hosts: NVIDIA Container Toolkit is installed
- [ ] If upgrading from Florence-2: clear the old `florence2_hf_cache` volume — it held Florence-2 weights; Donut uses `donut_hf_cache`
- [ ] NestJS service URL updated: `http://florence2:8001` → `http://donut:8001`
- [ ] If any consumer reads `dob_florence` from `/debug/compare`, update it to `dob_donut`

---

## Docker Compose

| File | Purpose |
|---|---|
| `docker-compose.yml` | Base / Dev — CPU, builds locally, `ENABLE_SWAGGER=true` |
| `docker-compose.gpu.yml` | GPU overlay — switches build to `Dockerfile.gpu`, adds GPU device access |
| `docker-compose.prod.yml` | Production override — registry image, `ENABLE_SWAGGER=false` |
| `docker-compose.prod-gpu.yml` | GPU + Production overlay |

### CPU (default)

```bash
# Dev
make serve

# Prod
export DONUT_IMAGE=your-registry/donut:latest
make serve-prod
```

### GPU

```bash
# Dev (GPU)
make serve-gpu

# Prod (GPU)
export DONUT_IMAGE=your-registry/donut:gpu-latest
make serve-prod-gpu
```

---

## Makefile Reference

| Target | Description |
|---|---|
| `make build` | Build the CPU image from `Dockerfile` |
| `make build-gpu` | Build the GPU image from `Dockerfile.gpu` |
| `make serve` | Start the service (CPU) |
| `make serve-gpu` | Start the service (GPU) |
| `make serve-build` | Rebuild and start (CPU) |
| `make serve-build-gpu` | Rebuild and start (GPU) |
| `make serve-prod` | Start production (CPU) |
| `make serve-prod-gpu` | Start production (GPU) |
| `make build-prod` | Build production CPU image |
| `make build-prod-gpu` | Build production GPU image |

---

## Model Volumes

| Volume | Mount path in container | Content | Size |
|---|---|---|---|
| `donut_hf_cache` | `/data/models/huggingface` | Donut model weights | ~500 MB |
| `donut_paddle_cache` | `/root/.paddleocr` | PaddleOCR English model | ~50 MB |

> **Upgrading from Florence-2?** The old `florence2_hf_cache` volume holds Florence-2-large-ft weights (~1.54 GB). You can remove it to free disk space — Donut uses a separate `donut_hf_cache` volume:
> ```bash
> docker volume rm florence2_hf_cache
> ```

---

## Maintenance & Updates

### Updating Model Weights

To update the Donut model to a newer HuggingFace commit:

1. **Update `MODEL_REVISION`** in `services/donut_service.py`
2. **Clear the HuggingFace volume:** `docker volume rm donut_hf_cache`
3. **Restart the service** — it downloads the new revision on startup

Visit the [Donut DocVQA commits page](https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa/commits/main) to find the latest SHA.
