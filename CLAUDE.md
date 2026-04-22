# CLAUDE.md — ID Verification Service (Donut + PaddleOCR)

> This file is read by AI coding assistants at the start of every session to avoid redundant exploration. Keep it up to date when the architecture changes.

## What This Service Does

Stateless FastAPI microservice that accepts a government-issued ID image and returns an age verification result. Called by the NestJS backend (`inrcliq-backend`) at `POST http://donut:8001/verify/id`.

**Version:** 2.0.0  
**Port:** 8001

---

## Architecture — Tiered DoB Extraction Pipeline

### Tier 1 (fast path — ~70–90% of requests)

**PaddleOCR + regex + spatial heuristics**

- Runs on every request
- Returns immediately if confident — Donut is **not** invoked
- High confidence: keyword-anchored match (preceded by "birth", "dob", etc.)
- Low confidence: single unanchored candidate — still returns without escalation

### Tier 2 (escalation — ~10–30% of requests)

**Donut (naver-clova-ix/donut-base-finetuned-docvqa)**

- Only invoked when Tier 1 cannot produce a confident result
- Escalation triggers: no candidates found, OCR text too short, multiple unanchored candidates
- Model: `naver-clova-ix/donut-base-finetuned-docvqa` @ revision `b19d2e332684b0e2d35d9144ce34047767335cf8`
- Asks: `<s_docvqa>What is the date of birth?</s_docvqa>`
- ~500 MB — 68% smaller than the previous Florence-2-large-ft model (~1.54 GB)

---

## File Map

```
main.py                  FastAPI app — routes, request handling, logging
coordinator.py           DoBCoordinator — tiered extraction logic
schemas.py               Pydantic request/response models
services/
  constants.py           Regex patterns, DoB keywords, escalation thresholds
  paddle_service.py      PaddleOCR + regex + PaddleExtractionResult dataclass
  donut_service.py       DonutService — VisionEncoderDecoderModel DocVQA inference
requirements.txt         Python dependencies (Docker only — not installed locally)
Dockerfile               CPU image (python:3.11-slim-bullseye, torch CPU wheel)
Dockerfile.gpu           GPU image (nvidia/cuda:12.1.1, torch CUDA 12.1 wheel)
docker-compose.yml       Base dev compose (CPU, ENABLE_SWAGGER=true)
docker-compose.gpu.yml   GPU overlay
docker-compose.prod.yml  Production overrides (ENABLE_SWAGGER=false)
docker-compose.prod-gpu.yml  GPU + production
Makefile                 Build/serve shortcuts
docs/
  IMPLEMENTATION.md      Architecture, pipeline, endpoint reference
  DEPLOYMENT.md          Resource requirements, Docker setup
  DEPLOYMENT_COST_ANALYSIS.md  Cloud cost breakdown
  LOCAL_DEVELOPMENT.md   First-time setup and dev workflow
donut.postman_collection.json  Postman collection for manual testing
```

---

## Key Data Structures

### `PaddleExtractionResult` (paddle_service.py)

```python
@dataclass
class PaddleExtractionResult:
    dob: str | None           # 'YYYY-MM-DD' or None
    candidate_count: int      # number of date-pattern matches found
    keyword_anchored: bool    # was winning match near a DoB keyword?
    raw_text: str             # full raw OCR text
```

### `CoordinatorResult` (coordinator.py)

```python
@dataclass
class CoordinatorResult:
    dob: str | None
    confidence_tier: str      # see tiers below
    dob_donut: str | None     # None if Tier 2 not invoked
    dob_paddle: str | None
    tier_used: str            # "tier1" or "tier2"
    escalation_reason: str | None  # None if Tier 1 resolved
    raw_paddle_text: str      # exposed via /debug/compare
```

### Confidence Tiers

| Tier              | Meaning                                             |
| ----------------- | --------------------------------------------------- |
| `tier1_high`      | Paddle keyword-anchored match — Donut not invoked   |
| `tier1_low`       | Paddle single unanchored match — Donut not invoked  |
| `tier2_confirmed` | Donut confirmed Paddle's DoB                        |
| `tier2_independent`| Donut independently extracted DoB (Paddle had None)|
| `tier2_conflict`  | Both found DoB but disagreed — Donut wins           |
| `tier2_failed`    | Neither model found DoB — request fails             |

### Escalation Reasons

| Reason                           | Trigger                                    |
| -------------------------------- | ------------------------------------------ |
| `ocr_text_too_short`             | Raw OCR text < 20 chars                    |
| `no_candidates`                  | Zero date patterns found                   |
| `multiple_unanchored_candidates` | > 1 date candidates, none keyword-anchored |

---

## API Endpoints

### POST /verify/id

- **Input:** `session_id` (form), `id_image` (file) or `id_image_url` (form)
- **Output:** `IDVerificationResponse` — includes `tier_used`, `escalation_reason`, `confidence_tier`

### GET /health

- Returns `{ "donut_model": "Donut-base (escalation only)", "paddle_model": "PaddleOCR-en", "device": "cpu|cuda" }`

### POST /debug/compare (dev only, ENABLE_SWAGGER=true)

- Returns full tier decision + `raw_paddle_text` and `dob_donut` for diagnosing escalation decisions

---

## Models

| Model                                              | Purpose                  | Size    | When Loaded                |
| -------------------------------------------------- | ------------------------ | ------- | -------------------------- |
| `PaddleOCR` (en)                                   | Tier 1 OCR               | ~50 MB  | Always                     |
| `naver-clova-ix/donut-base-finetuned-docvqa`       | Tier 2 DocVQA escalation | ~500 MB | Always (loaded at startup) |

Both models load at startup via FastAPI `lifespan`. Donut weights are cached in the `donut_hf_cache` Docker named volume.

---

## Environment Variables

| Variable         | Default                    | Description                                                 |
| ---------------- | -------------------------- | ----------------------------------------------------------- |
| `ENABLE_SWAGGER` | `true`                     | Set `false` in production to disable `/docs` and `/debug/*` |
| `HF_HOME`        | `/data/models/huggingface` | Set in Dockerfile — controls HuggingFace cache path         |

---

## Escalation Thresholds (tunable in `services/constants.py`)

| Constant                               | Default | Effect                                                     |
| -------------------------------------- | ------- | ---------------------------------------------------------- |
| `ESCALATION_MIN_TEXT_LENGTH`           | `20`    | Escalate if OCR text < this many chars                     |
| `ESCALATION_MAX_UNANCHORED_CANDIDATES` | `1`     | Escalate if more than this many unanchored date candidates |

---

## Running Locally

Everything runs inside Docker. No local Python setup needed.

```bash
make build && make serve      # CPU
make build-gpu && make serve-gpu  # GPU
```

Service ready when logs show:

```
[DonutService] Model ready for inference.
[PaddleService] Ready.
```

**First startup:** Donut downloads from HuggingFace (~500 MB) and PaddleOCR CDN (~50 MB). Subsequent starts load from Docker named volumes.

---

## NestJS Consumer Notes

- The `confidence_tier` field values remain unchanged from v2.0: `tier1_high`, `tier1_low`, `tier2_confirmed`, `tier2_independent`, `tier2_conflict`, `tier2_failed`
- The `/debug/compare` response field changed: `dob_florence` → `dob_donut`
- New fields in response: `tier_used`, `escalation_reason`
- NestJS timeout should be: **60s (CPU)** / **30s (GPU)** minimum
- The service hostname in docker-compose changed from `florence2` to `donut` — update NestJS service URL accordingly

---

## Important Implementation Notes

- **Single `pip install` pass:** Both Dockerfile variants install all packages in one pip call to avoid OpenMP runtime conflicts between PyTorch and PaddlePaddle (segfault if installed in separate calls)
- **No `flash_attn` stub:** Donut's `VisionEncoderDecoderModel` does not import `flash_attn` at module load time — the stub block has been removed from both Dockerfiles
- **PaddleOCR is CPU-only:** Even in the GPU image, PaddlePaddle runs on CPU. Only Donut (torch-based) benefits from GPU
- **`python:3.11-slim-bullseye` base:** Required for PaddlePaddle glibc 2.31 compatibility. Do not upgrade to Bookworm
- **DonutService architecture:** Uses `VisionEncoderDecoderModel` (encoder-decoder), not `AutoModelForCausalLM`. Prompt format is `<s_docvqa>question</s_docvqa>`. Processor must use `images=` keyword arg
- **Model revision pinned:** `MODEL_REVISION = "b19d2e332684b0e2d35d9144ce34047767335cf8"` in `services/donut_service.py`
- **`transformers>=4.45.0`:** Unpinned from `==4.40.0` — the pin was Florence-2-specific and is no longer needed
