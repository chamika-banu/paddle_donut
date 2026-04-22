from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile

from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("donut.verify")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import httpx

async def download_image(url: str, suffix: str) -> str:
    """Helper to download an image from a URL into an ephemeral file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=15.0)
                response.raise_for_status()
                tmp.write(response.content)
            return tmp.name
        except Exception as e:
            os.unlink(tmp.name)
            raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {e}")

from coordinator import DoBCoordinator
from services.donut_service import DonutService
from services.paddle_service import PaddleService
from schemas import (
    DebugCompareData,
    DebugCompareResponse,
    ErrorDetail,
    HealthResponse,
    IDVerificationData,
    IDVerificationResponse,
)

service: DonutService | None = None
coordinator: DoBCoordinator | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models once at container startup; clean up on shutdown."""
    global service, coordinator

    service = DonutService()
    paddle = PaddleService()

    coordinator = DoBCoordinator(donut=service, paddle=paddle)

    yield

SWAGGER_ENABLED = os.getenv("ENABLE_SWAGGER", "true").lower() == "true"

app = FastAPI(
    lifespan=lifespan,
    title="ID Verification Service (Tiered Pipeline)",
    description=(
        "Extracts Date of Birth from government-issued ID documents using a "
        "cost-optimised tiered pipeline.\n\n"
        "**Tier 1 (fast path):** PaddleOCR + regex + spatial heuristics handles ~70–90% "
        "of clean IDs with no VLM invocation.\n\n"
        "**Tier 2 (escalation):** Donut (NAVER CLOVA OCR) DocVQA is invoked only "
        "for ambiguous or difficult cases where Tier 1 cannot produce a confident result. "
        "Donut is optimized for document understanding and is 68% smaller than Florence-2.\n\n"
        "Returns age verification result including which tier resolved the request."
    ),
    version="2.0.0",
    docs_url="/docs" if SWAGGER_ENABLED else None,
    redoc_url=None,
    openapi_url="/openapi.json" if SWAGGER_ENABLED else None,
)

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Service health check",
)
def health():
    return {
        "status": "ok",
        "donut_model": "Donut-base (escalation only)",
        "paddle_model": "PaddleOCR-en",
        "device": service.device if service else "not_loaded",
    }

@app.post(
    "/verify/id",
    response_model=IDVerificationResponse,
    tags=["Verification"],
    summary="Verify ID document and extract age",
    responses={
        200: {"description": "Pipeline ran — check `success` field for pass/fail"},
        500: {"description": "Unhandled system error during inference"},
    },
)
async def verify_id(
    session_id: str = Form(
        ...,
        description="UUID passed through from NestJS for response correlation.",
    ),
    id_image: UploadFile | None = File(None, description="JPEG / PNG / WEBP image of the ID document"),
    id_image_url: str | None = Form(None, description="URL of the ID document image"),
):
    """
    Tiered ID verification pipeline:

    1. **Tier 1** — PaddleOCR + regex + spatial heuristics
       - Keyword-anchored match → high confidence, return immediately (no Donut)
       - Single unanchored match → low confidence, return immediately (no Donut)
       - No candidates, short text, or multiple unanchored candidates → escalate

    2. **Tier 2** (escalation only) — Donut DocVQA
       - Asked: "What is the date of birth?"
       - Result cross-checked with any Paddle result

    3. Age validation — extracted DoB must be ≥ 18 years ago
    """
    if not id_image and not id_image_url:
        raise HTTPException(status_code=400, detail="Either id_image or id_image_url must be provided.")

    if id_image:
        suffix = os.path.splitext(id_image.filename or "upload.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(id_image.file, tmp)
            tmp_path = tmp.name
    else:
        tmp_path = await download_image(id_image_url, ".jpg")

    try:
        if coordinator is None or service is None:
            raise HTTPException(
                status_code=503,
                detail="Service is still initializing or failed to load models."
            )

        coord_result = await coordinator.extract_and_compare(tmp_path)

        logger.info(
            "[verify/id] COORDINATOR  session_id=%s  tier=%s  confidence_tier=%s  "
            "escalation_reason=%s  dob=%s",
            session_id,
            coord_result.tier_used,
            coord_result.confidence_tier,
            coord_result.escalation_reason,
            coord_result.dob,
        )

        data = IDVerificationData(
            is_valid=False,
            age_verified=False,
            extracted_dob=coord_result.dob,
            extracted_age=None,
            failure_reason=None,
            confidence_tier=coord_result.confidence_tier,
            tier_used=coord_result.tier_used,
            escalation_reason=coord_result.escalation_reason,
        )

        if coord_result.dob is None:
            data.failure_reason = "dob_extraction_failed"
            return IDVerificationResponse(
                success=False,
                data=data,
                error=ErrorDetail(
                    code="dob_extraction_failed",
                    message="No Date of Birth pattern found in the extracted text.",
                    stage="dob_extraction",
                ),
            )

        age = service.compute_age(coord_result.dob)
        data.extracted_age = age

        if age < 18:
            data.failure_reason = "age_below_minimum"
            return IDVerificationResponse(
                success=False,
                data=data,
                error=ErrorDetail(
                    code="age_below_minimum",
                    message="Guardian does not meet the minimum age requirement of 18.",
                    stage="age_validation",
                ),
            )

        data.is_valid = True
        data.age_verified = True
        return IDVerificationResponse(success=True, data=data, error=None)

    except Exception as exc:
        logger.exception(
            "[verify/id] SYSTEM ERROR  session_id=%s  error=%s", session_id, exc
        )
        return JSONResponse(
            status_code=500,
            content=IDVerificationResponse(
                success=False,
                data=None,
                error=ErrorDetail(
                    code="inference_error",
                    message=f"Model inference failed unexpectedly. Check service logs. ({exc})",
                    stage="ocr",
                ),
            ).model_dump(),
        )

    finally:
        os.unlink(tmp_path)

@app.post(
    "/debug/compare",
    response_model=DebugCompareResponse,
    tags=["Debug"],
    summary="Run both models independently and compare their DoB extractions — dev/debug use",
    include_in_schema=SWAGGER_ENABLED,
)
async def debug_compare(
    id_image: UploadFile | None = File(None, description="JPEG / PNG / WEBP image of the ID document"),
    id_image_url: str | None = Form(None, description="URL of the ID document image"),
):
    """
    Always runs **both** PaddleOCR and Donut on the same image, regardless
    of what the production tiered pipeline would decide. Use this to:
    - Verify both models are working correctly
    - See what each model independently extracts
    - Diagnose why one model succeeded or failed

    This bypasses the escalation logic entirely — Donut is always invoked.
    Only available when ENABLE_SWAGGER=true (dev mode).
    """
    if not SWAGGER_ENABLED:
        raise HTTPException(status_code=404)

    if not id_image and not id_image_url:
        raise HTTPException(status_code=400, detail="Either id_image or id_image_url must be provided.")

    if id_image:
        suffix = os.path.splitext(id_image.filename or "upload.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(id_image.file, tmp)
            tmp_path = tmp.name
    else:
        tmp_path = await download_image(id_image_url, ".jpg")

    try:
        if coordinator is None or service is None:
            raise HTTPException(
                status_code=503,
                detail="Service is still initializing or failed to load models."
            )

        # Step 1: PaddleOCR
        paddle_result = await asyncio.to_thread(
            coordinator._paddle.extract_with_confidence, tmp_path
        )

        # Step 2: Donut (always, regardless of what Paddle found)
        dob_donut = await asyncio.to_thread(
            coordinator._donut.extract_dob_vqa, tmp_path
        )

        return DebugCompareResponse(
            success=True,
            data=DebugCompareData(
                dob_paddle=paddle_result.dob,
                dob_donut=dob_donut,
                raw_paddle_text=paddle_result.raw_text,
            ),
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content=DebugCompareResponse(
                success=False,
                data=None,
                error=ErrorDetail(
                    code="inference_error",
                    message=str(exc),
                    stage="ocr",
                ),
            ).model_dump(),
        )
    finally:
        os.unlink(tmp_path)
