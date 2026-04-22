# ID Verification Service — Implementation Guide

## Overview

This is a stateless FastAPI microservice that accepts a government-issued ID document image and returns an age verification result. It uses a **tiered pipeline** to minimise VLM invocations — handling the majority of clean IDs with PaddleOCR alone and only escalating to Donut (DocVQA) for ambiguous cases.

- **Tier 1 Model:** `PaddleOCR` (regex + spatial heuristics) — handles ~70–90% of requests
- **Tier 2 Model (escalation):** `naver-clova-ix/donut-base-finetuned-docvqa` (DocVQA) — invoked for ~10–30% of requests
- **Runtime:** CPU or GPU — automatic detection via `torch.cuda.is_available()`
- **Port:** `8001`
- **Framework:** FastAPI + uvicorn
- **Version:** 2.0.0

---

## Architecture Position

```
NestJS Backend
    │
    ├──► POST http://donut:8001/verify/id
    │         multipart/form-data: session_id, {id_image | id_image_url}
    │
    │◄── IDVerificationResponse
    │         {
    │           success,
    │           data: { ..., confidence_tier, tier_used, escalation_reason },
    │           error
    │         }
    │
    │    If success (any tier) → proceed to InsightFace
    │    If dob_extraction_failed → reject
```

---

## File Structure

```
paddle_donut/
├── main.py                              # FastAPI app — routes, request handling, logging
├── coordinator.py                       # DoBCoordinator — tiered extraction logic
├── schemas.py                           # Pydantic request/response models
├── services/
│   ├── __init__.py                      # Package marker
│   ├── constants.py                     # Regex patterns, DoB keywords, escalation thresholds
│   ├── donut_service.py                 # DonutService — VisionEncoderDecoderModel DocVQA inference
│   └── paddle_service.py               # PaddleOCR + regex + PaddleExtractionResult
├── requirements.txt                     # Python dependencies
├── Dockerfile                           # CPU image — python:3.11-slim-bullseye, torch CPU wheel
├── Dockerfile.gpu                       # GPU image — nvidia/cuda base, torch CUDA 12.1 wheel
├── docker-compose.yml                   # Base dev compose — CPU-safe
├── docker-compose.gpu.yml              # GPU overlay — switches to Dockerfile.gpu + GPU devices
├── docker-compose.prod.yml             # Production overrides (CPU)
├── docker-compose.prod-gpu.yml         # GPU + Production overrides
├── Makefile                            # Shortcuts for build / serve (CPU + GPU targets)
├── CLAUDE.md                           # AI assistant context file (read this first!)
├── donut.postman_collection.json       # Postman collection for manual testing
└── docs/
    ├── IMPLEMENTATION.md               # Architecture and internal logic (this file)
    ├── LOCAL_DEVELOPMENT.md            # Getting started guide
    ├── DEPLOYMENT.md                   # Resource requirements and production tips
    └── DEPLOYMENT_COST_ANALYSIS.md    # Cloud cost breakdown by provider
```

---

## Dependencies

```
# Core ML
torch>=2.2.2
torchvision>=0.17.2
# NOTE: torch is installed with an explicit index URL in each Dockerfile:
#   Dockerfile     → --index-url https://download.pytorch.org/whl/cpu       (CPU-only wheel)
#   Dockerfile.gpu → --index-url https://download.pytorch.org/whl/cu121     (CUDA 12.1 wheel)

# Unpinned — Donut (VisionEncoderDecoderModel) requires transformers>=4.45.0;
# the ==4.40.0 pin was Florence-2-specific and is no longer needed.
transformers>=4.45.0
tokenizers>=0.15.2

# HF ecosystem
accelerate>=0.29.3
sentencepiece>=0.1.99

# Image processing
Pillow>=10.3.0
pillow-heif>=0.16.0
opencv-python-headless>=4.9.0.80

# Scientific stack
numpy<2.0.0
scipy>=1.12.0
einops>=0.7.0
timm>=0.9.16

# Utilities
python-dateutil>=2.9.0

# API layer
fastapi>=0.111.0
uvicorn>=0.29.0
python-multipart>=0.0.9
httpx>=0.27.0

# PaddleOCR — blocked on 3.x due to confirmed OneDNN CPU crash across all environments.
# numpy must stay <2.0.0 — paddleocr 2.7.3 compiled against NumPy 1.x ABI.
paddlepaddle>=2.6.2,<3.0.0
paddleocr>=2.7.3,<3.0.0
```

---

## CPU vs GPU Dockerfiles

|                   | `Dockerfile` (CPU)          | `Dockerfile.gpu` (GPU)                          |
| ----------------- | --------------------------- | ----------------------------------------------- |
| Base image        | `python:3.11-slim-bullseye` | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` |
| PyTorch build     | CPU wheel                   | CUDA 12.1 wheel                                 |
| Donut dtype       | `torch.float32`             | `torch.float16`                                 |
| PaddleOCR         | CPU-only                    | CPU-only                                        |
| `flash_attn` stub | Not needed                  | Not needed                                      |
| `make` target     | `make build` / `make serve` | `make build-gpu` / `make serve-gpu`             |

**Why `slim-bullseye`?** PaddlePaddle's manylinux wheels are compiled against **glibc 2.31** (Debian Bullseye). Bookworm (glibc 2.36) causes a heap crash. Do not upgrade.

**Single-pass pip install:** All packages are installed in one `pip install` invocation. PyTorch and PaddlePaddle each bundle their own OpenMP runtime; separate installs cause a segfault on `PaddleOCR()` initialisation.

---

## Model Loading

Models are loaded **once at container startup** using FastAPI's `lifespan` context manager.

### Donut (naver-clova-ix/donut-base-finetuned-docvqa)

- **MODEL_ID:** `naver-clova-ix/donut-base-finetuned-docvqa`
- **MODEL_REVISION:** `b19d2e332684b0e2d35d9144ce34047767335cf8`
- **Architecture:** `VisionEncoderDecoderModel` (encoder-decoder, not causal LM)
- **Size:** ~500 MB
- **Prompt format:** `<s_docvqa>What is the date of birth?</s_docvqa>`
- **Cache path (in container):** `/data/models/huggingface` (controlled by `HF_HOME` env var)
- **Volume:** `donut_hf_cache` mounted to `/data/models/huggingface`

> **To update the model revision:**
>
> 1. Update `MODEL_REVISION` in `services/donut_service.py`
> 2. Clear the volume: `docker volume rm donut_hf_cache`
> 3. Restart — new revision downloads automatically

### Device Detection

```python
if torch.cuda.is_available():
    self.device = "cuda"
    torch_dtype = torch.float16   # FP16 on GPU
else:
    self.device = "cpu"
    torch_dtype = torch.float32   # FP32 on CPU
```

---

## Tiered Extraction Pipeline

### Overview

```
Request
  │
  ▼
[Tier 1] PaddleOCR + regex + spatial heuristics
  │
  ├── keyword-anchored match found? ──► tier1_high → return ✅ (no Donut)
  │
  ├── single unanchored candidate? ──► tier1_low → return ✅ (no Donut)
  │
  └── escalation trigger? ──────────► [Tier 2] Donut DocVQA
                                          │
                                          ├── DoB found ──► tier2_confirmed / tier2_independent / tier2_conflict
                                          │
                                          └── DoB not found ──► tier2_failed → fail ❌
```

### Escalation Triggers (any one → escalate)

| Trigger                        | Constant                               | Default        |
| ------------------------------ | -------------------------------------- | -------------- |
| OCR text too short             | `ESCALATION_MIN_TEXT_LENGTH`           | < 20 chars     |
| No date candidates             | —                                      | 0 candidates   |
| Multiple unanchored candidates | `ESCALATION_MAX_UNANCHORED_CANDIDATES` | > 1 unanchored |

These constants are tunable in `services/constants.py` without a code redesign.

### PaddleExtractionResult

`PaddleService.extract_with_confidence()` returns a rich result used by the coordinator:

```python
@dataclass
class PaddleExtractionResult:
    dob: str | None       # 'YYYY-MM-DD' or None
    candidate_count: int  # total date-pattern matches
    keyword_anchored: bool# was the winning match near a DoB keyword?
    raw_text: str         # full raw OCR output
```

### Confidence Tiers

| Tier                | Source                                  | Donut Invoked? |
| ------------------- | --------------------------------------- | -------------- |
| `tier1_high`        | Paddle keyword-anchored                 | ❌ No          |
| `tier1_low`         | Paddle single unanchored                | ❌ No          |
| `tier2_confirmed`   | Donut confirmed Paddle's DoB            | ✅ Yes         |
| `tier2_independent` | Donut extracted DoB (Paddle found none) | ✅ Yes         |
| `tier2_conflict`    | Donut wins over differing Paddle result | ✅ Yes         |
| `tier2_failed`      | Both failed                             | ✅ Yes         |

### Tier Decision Matrix

| Paddle Result                 | Donut Invoked? | Outcome                       |
| ----------------------------- | -------------- | ----------------------------- |
| Keyword-anchored match        | ❌ No          | `tier1_high`                  |
| Single unanchored match       | ❌ No          | `tier1_low`                   |
| No match, short text          | ✅ Yes         | `tier2_*`                     |
| No match, multiple unanchored | ✅ Yes         | `tier2_*`                     |
| Both agree (after escalation) | ✅ Yes         | `tier2_confirmed`             |
| Donut finds DoB, Paddle None  | ✅ Yes         | `tier2_independent`           |
| Disagree (after escalation)   | ✅ Yes         | `tier2_conflict` (Donut wins) |
| Both fail                     | ✅ Yes         | `tier2_failed`                |

---

## Endpoints

### GET /health

Returns service readiness and active device.

```json
{
	"status": "ok",
	"model": "Donut-base (escalation only)",
	"paddle_model": "PaddleOCR-en",
	"device": "cpu"
}
```

---

### POST /verify/id

**Response — Check Passed, Tier 1 (HTTP 200)**

```json
{
	"success": true,
	"data": {
		"is_valid": true,
		"age_verified": true,
		"extracted_dob": "1989-04-12",
		"extracted_age": 36,
		"failure_reason": null,
		"confidence_tier": "tier1_high",
		"tier_used": "tier1",
		"escalation_reason": null
	},
	"error": null
}
```

**Response — Check Passed, Tier 2 escalation (HTTP 200)**

```json
{
	"success": true,
	"data": {
		"is_valid": true,
		"age_verified": true,
		"extracted_dob": "1989-04-12",
		"extracted_age": 36,
		"failure_reason": null,
		"confidence_tier": "tier2_confirmed",
		"tier_used": "tier2",
		"escalation_reason": "no_candidates"
	},
	"error": null
}
```

---

### POST /debug/compare

Development endpoint (available when `ENABLE_SWAGGER=true`). **Always** runs PaddleOCR first, then Donut — regardless of what Paddle found. Use this to verify both models are working and to see what each independently extracts from a given image.

```json
{
	"success": true,
	"data": {
		"dob_paddle": "1989-04-12",
		"dob_donut": "1989-04-12",
		"raw_paddle_text": "JOHN DOE\nDate of Birth: 12/04/1989\nNIC: 898742345V"
	},
	"error": null
}
```

---

## Failure Codes

| `code`                  | `stage`          | Meaning                                       |
| ----------------------- | ---------------- | --------------------------------------------- |
| `dob_extraction_failed` | `dob_extraction` | No DoB found by either tier                   |
| `age_below_minimum`     | `age_validation` | Extracted age < 18                            |
| `inference_error`       | `ocr`            | Unhandled system error — check container logs |

---

## Environment Variables

| Variable         | Default                    | Description                                                 |
| ---------------- | -------------------------- | ----------------------------------------------------------- |
| `ENABLE_SWAGGER` | `true`                     | Set `false` in production to disable `/docs` and `/debug/*` |
| `HF_HOME`        | `/data/models/huggingface` | HuggingFace cache path (set in Dockerfile)                  |

---

## Pydantic Schemas

```python
class IDVerificationData(BaseModel):
    is_valid: bool
    age_verified: bool
    extracted_dob: str | None
    extracted_age: int | None
    failure_reason: str | None
    confidence_tier: str | None    # tier1_high | tier1_low | tier2_*
    tier_used: str | None          # "tier1" or "tier2"
    escalation_reason: str | None  # None if Tier 1 resolved

class DebugCompareData(BaseModel):
    dob_donut: str | None          # renamed from dob_florence in Donut migration
    dob_paddle: str | None
    raw_paddle_text: str           # raw OCR output — for diagnosing regex failures
```
