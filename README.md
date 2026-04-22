# ID Verification Service

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/release/python-3110/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

A cost-optimised OCR microservice specialised for extracting **Date of Birth (DoB)** from government-issued ID documents, using a **tiered pipeline** to minimise VLM invocations.

---

## 🚀 Key Features

- **Tiered Pipeline**: PaddleOCR handles ~70–90% of clean IDs in < 2s. Donut only invoked for ambiguous cases.
- **Cost-Efficient**: Tier 2 model is [Donut](https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa) (~500 MB) replacing Florence-2-large-ft (1.54 GB) — 68% smaller, comparable accuracy on document extraction tasks.
- **Confidence Tiers**: Results categorised as `tier1_high`, `tier1_low`, `tier2_confirmed`, `tier2_independent`, `tier2_conflict`, or `tier2_failed`.
- **Age Validation**: Verifies extracted age meets the minimum requirement (18+).
- **Containerised**: Fully Dockerised with CPU and GPU support.
- **Developer Friendly**: Full Swagger/OpenAPI docs, `/debug/compare` endpoint, and a Postman collection.

---

## 🛠 Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Models**:
    - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — Tier 1, always runs
    - [Donut (naver-clova-ix/donut-base-finetuned-docvqa)](https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa) — Tier 2, escalation only
- **Processing**: PyTorch, OpenCV, Pillow
- **Server**: Uvicorn

---

## ⚙️ Pipeline

```
Request
  │
  ▼
[Tier 1] PaddleOCR + regex + spatial heuristics
  │
  ├── keyword-anchored match ──► tier1_high → return ✅ (no Donut, < 2s)
  ├── single unanchored match ──► tier1_low → return ✅ (no Donut, < 2s)
  └── ambiguous / no match ───► [Tier 2] Donut DocVQA
                                    │
                                    ├── DoB found ──► tier2_confirmed / tier2_independent / tier2_conflict
                                    └── DoB not found ──► tier2_failed ❌
```

**Escalation triggers (any one → escalate to Donut):**
- Raw OCR text < 20 characters (blurry / cropped image)
- No date patterns found
- Multiple date candidates, none near a DoB keyword ("birth", "dob", etc.)

---

## 📡 API Endpoints

### `POST /verify/id`

| Parameter | Type | Description |
|---|---|---|
| `session_id` | string | UUID for response correlation |
| `id_image` | file | (Optional) JPEG / PNG / WEBP image of the ID |
| `id_image_url` | string | (Optional) Direct URL to the ID image |

**Example response (Tier 1 resolved):**
```json
{
    "success": true,
    "data": {
        "is_valid": true,
        "age_verified": true,
        "extracted_dob": "1989-04-12",
        "extracted_age": 36,
        "confidence_tier": "tier1_high",
        "tier_used": "tier1",
        "escalation_reason": null
    },
    "error": null
}
```

### `POST /debug/compare`

Shows full tier decision metadata including `raw_paddle_text`, `dob_paddle`, and `dob_donut`. Development only (`ENABLE_SWAGGER=true`).

### `GET /health`

```json
{
    "status": "ok",
    "donut_model": "Donut-base (escalation only)",
    "paddle_model": "PaddleOCR-en",
    "device": "cpu"
}
```

---

## 📦 Getting Started

```bash
# CPU (default)
make build
make serve

# GPU
make build-gpu
make serve-gpu
```

Service available at `http://localhost:8001`. Swagger UI at `http://localhost:8001/docs`.

On first startup, Donut downloads from HuggingFace (~500 MB) and PaddleOCR CDN (~50 MB) and cache in Docker volumes.

---

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `ENABLE_SWAGGER` | `true` | Set `false` in production to disable Swagger and `/debug/*` |

---

## 👨‍💻 Development

- Import `donut.postman_collection.json` into Postman for endpoint testing.
- See `CLAUDE.md` for a quick architecture reference designed for AI-assisted development.
- See `docs/` for full implementation, deployment, and cost analysis documentation.
