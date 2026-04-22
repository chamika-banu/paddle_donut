# ID Verification Service — Local Development Guide

## Prerequisites

| Tool                     | Version                          | Purpose                                  |
| ------------------------ | -------------------------------- | ---------------------------------------- |
| Docker Desktop           | 4.x+                             | Container runtime                        |
| Docker Compose           | v2 (bundled with Docker Desktop) | Service orchestration                    |
| Postman                  | any                              | Endpoint testing                         |
| make                     | any                              | Shortcut commands                        |
| NVIDIA Container Toolkit | latest                           | **GPU mode only** — not required for CPU |

No local Python environment is required. Everything runs inside Docker.

---

## Service Layout

```
paddle_donut/
├── main.py                              # FastAPI app — routes, request handling
├── coordinator.py                       # DoBCoordinator — tiered extraction logic
├── schemas.py                           # Pydantic request/response models
├── services/
│   ├── __init__.py                      # Package marker
│   ├── constants.py                     # Regex patterns, DoB keywords, escalation thresholds
│   ├── donut_service.py                 # DonutService — VisionEncoderDecoderModel DocVQA
│   └── paddle_service.py               # PaddleOCR + PaddleExtractionResult
├── requirements.txt                     # Python dependencies
├── Dockerfile                           # CPU image — python:3.11-slim-bullseye, torch CPU wheel
├── Dockerfile.gpu                       # GPU image — nvidia/cuda base, torch CUDA 12.1 wheel
├── docker-compose.yml                   # Base dev compose — CPU-safe
├── docker-compose.gpu.yml               # GPU overlay
├── docker-compose.prod.yml              # Production overrides (CPU)
├── docker-compose.prod-gpu.yml          # GPU + Production overrides
├── Makefile                             # Shortcuts for build / serve
├── CLAUDE.md                            # AI assistant context file
├── donut.postman_collection.json        # Postman collection for manual testing
└── docs/                                # Documentation folder
    ├── IMPLEMENTATION.md
    ├── LOCAL_DEVELOPMENT.md
    ├── DEPLOYMENT.md
    └── DEPLOYMENT_COST_ANALYSIS.md
```

---

## First-Time Setup

### CPU (default — works on any machine)

```bash
make build
# or:
docker compose build
```

### GPU (requires NVIDIA Container Toolkit on host)

```bash
make build-gpu
# or:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
```

Model weights are **not** baked into the image. On first startup, the container downloads Donut (~500 MB) and PaddleOCR (~50 MB) and caches them in named Docker volumes. Subsequent starts load from the cache with no network required.

---

## Running the Service

### CPU (default)

```bash
make serve
# or:
docker compose up
```

### GPU

```bash
make serve-gpu
# or:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

### Rebuild and start (after changing requirements or Dockerfile)

```bash
# CPU
make serve-build

# GPU
make serve-build-gpu
```

---

## Startup Output

Look for these lines in the logs:

```
[DonutService] Loading model 'naver-clova-ix/donut-base-finetuned-docvqa' on CPU...
[DonutService] Loading processor...
[DonutService] Loading model (~500 MB)...
[DonutService] Model loaded successfully on cpu
[DonutService] Model ready for inference.
[PaddleService] Downloading/Loading PaddleOCR models... (This is ~50MB and only happens on first run if not cached!)
[PaddleService] Ready.
```

The service will not accept requests until both are ready.

- **First startup (empty volume):** Models download over the network — allow up to **3 minutes** depending on connection speed.
- **CPU (cached volume):** ~5–15 seconds
- **GPU (cached volume):** ~3–6 seconds

---

## Verifying the Active Device

Check the `/health` endpoint after startup:

```bash
curl http://localhost:8001/health
```

**CPU response:**

```json
{
	"status": "ok",
	"model": "Donut-base (escalation only)",
	"paddle_model": "PaddleOCR-en",
	"device": "cpu"
}
```

**GPU response:**

```json
{
	"status": "ok",
	"model": "Donut-base (escalation only)",
	"paddle_model": "PaddleOCR-en",
	"device": "cuda"
}
```

> PaddleOCR runs CPU-only in both images — `paddle_model` is always on CPU regardless of device.

---

## Postman

Import `donut.postman_collection.json` into Postman.

**Recommended testing order:**

1. `GET /health` — confirm both models are loaded and check active device
2. `POST /debug/compare` — inspect which model extracted what and view raw OCR text
3. `POST /verify/id` with a clean adult ID — expect `tier1_high` or `tier1_low`, `success: true`
4. `POST /verify/id` with a minor's ID — expect `age_below_minimum`
5. `POST /verify/id` with a blurry/low-quality ID — expect `tier2_*` + an `escalation_reason`

---

## Understanding the Tier Decision

After a request, check these fields in the response:

| Field               | Meaning                                                                    |
| ------------------- | -------------------------------------------------------------------------- |
| `tier_used`         | `"tier1"` = PaddleOCR resolved it; `"tier2"` = Donut was invoked          |
| `escalation_reason` | Why Tier 2 was triggered (`null` if Tier 1 resolved)                       |
| `confidence_tier`   | Full resolution detail — see table below                                   |

**Confidence tiers:**

| Value             | Meaning                                                      |
| ----------------- | ------------------------------------------------------------ |
| `tier1_high`      | Paddle found keyword-anchored DoB — very reliable            |
| `tier1_low`       | Paddle found an unanchored single DoB — proceed, lower trust |
| `tier2_confirmed` | Donut confirmed Paddle's extracted DoB                       |
| `tier2_independent`| Donut independently extracted DoB (Paddle found none)       |
| `tier2_conflict`  | Both found DoB but disagreed — Donut used as ground truth    |
| `tier2_failed`    | Neither model found DoB — request fails                      |

**Escalation reasons:**

| Value                            | Trigger                                                   |
| -------------------------------- | --------------------------------------------------------- |
| `ocr_text_too_short`             | Raw OCR text was < 20 chars — likely blurry/cropped image |
| `no_candidates`                  | No date patterns found in OCR text                        |
| `multiple_unanchored_candidates` | > 1 dates found, none near a DoB keyword                  |

---

## Reading the Logs

```
[INFO] donut.verify — [coordinator] tier=1  dob=1989-04-12  candidates=1  keyword_anchored=True  text_len=142
[INFO] donut.verify — [coordinator] Tier 1 resolved: tier1_high  dob=1989-04-12
[INFO] donut.verify — [verify/id] COORDINATOR  session_id=test-001  tier=tier1  confidence_tier=tier1_high  escalation_reason=None  dob=1989-04-12
```

**Escalated example:**

```
[INFO] donut.verify — [coordinator] tier=1  dob=None  candidates=0  keyword_anchored=False  text_len=8
[INFO] donut.verify — [coordinator] Escalating to Tier 2. reason=no_candidates
[INFO] donut.verify — [coordinator] tier=2  donut_dob=1989-04-12  paddle_dob=None
```

---

## Makefile Reference

| Target                 | Description                 |
| ---------------------- | --------------------------- |
| `make build`           | Build the CPU Docker image  |
| `make build-gpu`       | Build the GPU Docker image  |
| `make serve`           | Start the service (CPU)     |
| `make serve-gpu`       | Start the service (GPU)     |
| `make serve-build`     | Rebuild and start (CPU)     |
| `make serve-build-gpu` | Rebuild and start (GPU)     |
| `make serve-prod`      | Start production mode (CPU) |
| `make serve-prod-gpu`  | Start production mode (GPU) |
