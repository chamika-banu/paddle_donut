# These models are used by FastAPI to generate accurate OpenAPI/Swagger docs

from pydantic import BaseModel


class IDVerificationData(BaseModel):
    is_valid: bool
    age_verified: bool
    extracted_dob: str | None
    extracted_age: int | None
    failure_reason: str | None
    confidence_tier: str | None
    tier_used: str | None           # "tier1" or "tier2"
    escalation_reason: str | None   # None if Tier 1 resolved; explains why Donut was invoked


class ErrorDetail(BaseModel):
    code: str
    message: str
    stage: str


class IDVerificationResponse(BaseModel):
    success: bool
    data: IDVerificationData | None
    error: ErrorDetail | None


class DebugCompareData(BaseModel):
    dob_paddle: str | None
    dob_donut: str | None
    raw_paddle_text: str


class DebugCompareResponse(BaseModel):
    success: bool
    data: DebugCompareData | None
    error: ErrorDetail | None


class HealthResponse(BaseModel):
    status: str
    donut_model: str
    paddle_model: str
    device: str
