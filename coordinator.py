from __future__ import annotations

import asyncio

from dataclasses import dataclass
from typing import Literal

from services.donut_service import DonutService
from services.paddle_service import PaddleService
from services.constants import (
    ESCALATION_MIN_TEXT_LENGTH,
    ESCALATION_MAX_UNANCHORED_CANDIDATES,
)

ConfidenceTier = Literal[
    "tier1_high",        # Paddle found a keyword-anchored match — Donut not invoked
    "tier1_low",         # Paddle found an unanchored single match — Donut not invoked
    "tier2_confirmed",    # Donut confirmed Paddle's extracted DoB
    "tier2_independent", # Donut independently extracted a DoB (Paddle had None)
    "tier2_conflict",     # Donut and Paddle both found dates but they differ — Donut wins
    "tier2_failed",      # Donut was invoked but could not extract a DoB; Paddle had None too
]


@dataclass
class CoordinatorResult:
    dob: str | None
    confidence_tier: ConfidenceTier
    dob_donut: str | None   # populated only when Tier 2 was invoked
    dob_paddle: str | None
    tier_used: Literal["tier1", "tier2"]
    escalation_reason: str | None  # None if Tier 1 succeeded; explains why Tier 2 was triggered
    raw_paddle_text: str           # raw OCR text from PaddleOCR — exposed via /debug/compare


class DoBCoordinator:
    """
    Tiered DoB extraction coordinator.

    Tier 1 — PaddleOCR + regex + spatial heuristics (fast, CPU-only)
    ─────────────────────────────────────────────────────────────────
    Runs first on every request. Escalates to Tier 2 if ANY of:
      • No date candidates found in the OCR text
      • Raw OCR text is too short (likely blurry/cropped image)
      • Multiple date candidates found and NONE is keyword-anchored
        (i.e., cannot distinguish DoB from expiry/issue dates)

    Tier 2 — Donut DocVQA (escalation only, ~10–30% of requests)
    ──────────────────────────────────────────────────────────────
    Invoked only when Tier 1 cannot produce a confident result.
    Donut reads the document as a whole and directly answers
    "What is the date of birth?" — bypassing regex entirely.
    Donut is optimized for document understanding; ~68% smaller than Florence-2.

    Confidence tiers:
      tier1_high        — Paddle found a keyword-anchored DoB → high trust
      tier1_low         — Paddle found an unanchored single candidate → proceed, lower trust
      tier2_confirmed    — Donut confirmed Paddle's extracted DoB
      tier2_independent — Donut extracted a DoB while Paddle found nothing
      tier2_conflict     — Both extracted different DoBs → Donut used as ground truth
      tier2_failed      — Neither model found a DoB → request fails
    """

    def __init__(
        self,
        donut: DonutService,
        paddle: PaddleService,
    ) -> None:
        self._donut = donut
        self._paddle = paddle

    def _should_escalate(self, paddle_result) -> tuple[bool, str | None]:
        """
        Evaluate whether the Tier 1 result is confident enough to trust.
        Returns (should_escalate, reason_string).
        """
        raw_text = paddle_result.raw_text

        # Rule 1: OCR text too short — image quality likely too poor for regex
        if len(raw_text.strip()) < ESCALATION_MIN_TEXT_LENGTH:
            return True, "ocr_text_too_short"

        # Rule 2: No date candidates found at all
        if paddle_result.candidate_count == 0:
            return True, "no_candidates"

        # Rule 3: Paddle found a result, but it's keyword-anchored → high confidence
        if paddle_result.dob is not None and paddle_result.keyword_anchored:
            return False, None  # Tier 1 high confidence — no escalation

        # Rule 4: Multiple unanchored candidates — ambiguous (could be expiry, issue, dob)
        if (
            paddle_result.candidate_count > ESCALATION_MAX_UNANCHORED_CANDIDATES
            and not paddle_result.keyword_anchored
        ):
            return True, "multiple_unanchored_candidates"

        # Rule 5: Single unanchored candidate — low confidence but proceed without escalation
        # (escalating for every unanchored match would over-invoke Florence)
        return False, None

    async def extract_and_compare(self, image_path: str) -> CoordinatorResult:
        loop = asyncio.get_event_loop()

        # ── Tier 1: PaddleOCR + regex ─────────────────────────────────────────
        paddle_result = await loop.run_in_executor(
            None, self._paddle.extract_with_confidence, image_path
        )
        print(
            f"[coordinator] tier=1  dob={paddle_result.dob}  "
            f"candidates={paddle_result.candidate_count}  "
            f"keyword_anchored={paddle_result.keyword_anchored}  "
            f"text_len={len(paddle_result.raw_text.strip())}"
        )

        escalate, escalation_reason = self._should_escalate(paddle_result)

        if not escalate:
            # Tier 1 is confident enough — return without invoking Donut
            tier = "tier1_high" if paddle_result.keyword_anchored else "tier1_low"
            print(f"[coordinator] Tier 1 resolved: {tier}  dob={paddle_result.dob}")
            return CoordinatorResult(
                dob=paddle_result.dob,
                confidence_tier=tier,
                dob_donut=None,
                dob_paddle=paddle_result.dob,
                tier_used="tier1",
                escalation_reason=None,
                raw_paddle_text=paddle_result.raw_text,
            )

        # ── Tier 2: Donut DocVQA (escalation) ────────────────────────────────
        print(f"[coordinator] Escalating to Tier 2. reason={escalation_reason}")
        dob_donut = await loop.run_in_executor(
            None, self._donut.extract_dob_vqa, image_path
        )
        print(f"[coordinator] tier=2  donut_dob={dob_donut}  paddle_dob={paddle_result.dob}")

        # Sub-case A: Donut found a DoB
        if dob_donut is not None:
            if paddle_result.dob is not None and dob_donut == paddle_result.dob:
                # Both agree
                return CoordinatorResult(
                    dob=dob_donut,
                    confidence_tier="tier2_confirmed",
                    dob_donut=dob_donut,
                    dob_paddle=paddle_result.dob,
                    tier_used="tier2",
                    escalation_reason=escalation_reason,
                    raw_paddle_text=paddle_result.raw_text,
                )
            elif paddle_result.dob is not None and dob_donut != paddle_result.dob:
                # Both found dates but they differ — use Donut as ground truth
                return CoordinatorResult(
                    dob=dob_donut,
                    confidence_tier="tier2_conflict",
                    dob_donut=dob_donut,
                    dob_paddle=paddle_result.dob,
                    tier_used="tier2",
                    escalation_reason=escalation_reason,
                    raw_paddle_text=paddle_result.raw_text,
                )
            else:
                # Donut found it; Paddle had found nothing
                return CoordinatorResult(
                    dob=dob_donut,
                    confidence_tier="tier2_independent",
                    dob_donut=dob_donut,
                    dob_paddle=None,
                    tier_used="tier2",
                    escalation_reason=escalation_reason,
                    raw_paddle_text=paddle_result.raw_text,
                )

        # Sub-case B: Donut also failed — use Paddle's result if it had one (last resort)
        dob = paddle_result.dob  # may still be None
        return CoordinatorResult(
            dob=dob,
            confidence_tier="tier2_failed",
            dob_donut=None,
            dob_paddle=dob,
            tier_used="tier2",
            escalation_reason=escalation_reason,
            raw_paddle_text=paddle_result.raw_text,
        )

