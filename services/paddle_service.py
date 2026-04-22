from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, date
from dateutil import parser as dateutil_parser
from paddleocr import PaddleOCR

from .constants import (
    DOB_PATTERNS,
    DOB_CONTEXT_KEYWORDS,
    ESCALATION_MIN_TEXT_LENGTH,
    ESCALATION_MAX_UNANCHORED_CANDIDATES,
)


@dataclass
class PaddleExtractionResult:
    """
    Rich result from Tier 1 extraction.

    The coordinator uses these fields to decide whether to escalate to Tier 2:
    - dob: extracted date in 'YYYY-MM-DD' format, or None
    - candidate_count: total number of date-pattern matches found in the text
    - keyword_anchored: True if the winning match was preceded by a DoB keyword
                        (e.g. "birth", "dob") within 60 characters
    - raw_text: full OCR text — used for escalation length check and debug logging
    """
    dob: str | None
    candidate_count: int
    keyword_anchored: bool
    raw_text: str


class PaddleService:
    def __init__(self) -> None:
        print("[PaddleService] Downloading/Loading PaddleOCR models... (This is ~50MB and only happens on first run if not cached!)")
        # use_angle_cls=True handles rotated/tilted ID photos.
        # show_log=False suppresses PaddleOCR's verbose stdout.
        # rec_batch_num=1 prevents 'contiguous Tensor' crashes in rec_multi_head on CPU.
        self._ocr_en = PaddleOCR(
            use_angle_cls=True, 
            lang='en', 
            show_log=False, 
            rec_batch_num=1
        )
        print("[PaddleService] Ready.")

    def _run_ocr(self, image_path: str) -> str:
        """Run PaddleOCR and return all recognised text as a single string."""
        result = self._ocr_en.ocr(image_path, cls=True)
        if not result or not result[0]:
            return ""
        lines = [line[1][0] for block in result for line in block if line[1]]
        return "\n".join(lines)

    def _extract_candidates(self, text: str) -> list[tuple[int, str]]:
        """
        Run regex sweep over OCR text. Returns a list of (priority, date_str) tuples.
        Priority 1 = keyword-anchored (near a DoB context keyword), 0 = unanchored.
        """
        text_lower = text.lower()
        candidates: list[tuple[int, str]] = []

        for pattern in DOB_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start = match.start()
                window = text_lower[max(0, start - 60):start]
                priority = 0
                for kw in DOB_CONTEXT_KEYWORDS:
                    if kw in window:
                        priority = 1
                        break
                candidates.append((priority, match.group(1)))

        # Sort by priority descending — keyword-anchored matches first
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates

    def _parse_date(self, raw_date: str) -> str | None:
        """
        Attempt to parse a raw date string into 'YYYY-MM-DD'.
        Returns None if parsing fails or the date is implausible.
        """
        try:
            year_first = bool(re.match(r'^\d{4}', raw_date))
            parsed: datetime = dateutil_parser.parse(
                raw_date,
                dayfirst=not year_first,
                yearfirst=year_first,
            )
            today = date.today()
            if parsed.date() >= today:
                return None
            if today.year - parsed.year > 120:
                return None
            return parsed.strftime("%Y-%m-%d")
        except (ValueError, OverflowError):
            return None

    def extract_with_confidence(self, image_path: str) -> PaddleExtractionResult:
        """
        Run PaddleOCR on the image and return a rich PaddleExtractionResult.

        The coordinator uses this to decide whether to escalate to Tier 2 (Donut):
        - keyword_anchored == True  → Skip Donut (high trust)
        - candidate_count == 0      → Escalate to Donut
        - candidate_count > 1       → Escalate to Donut
        - Otherwise → Tier 1 result is trusted, Donut is not invoked
        """
        raw_text = self._run_ocr(image_path)
        candidates = self._extract_candidates(raw_text)

        dob: str | None = None
        keyword_anchored = False

        for priority, raw_date in candidates:
            parsed = self._parse_date(raw_date)
            if parsed:
                dob = parsed
                keyword_anchored = priority == 1
                break

        return PaddleExtractionResult(
            dob=dob,
            candidate_count=len(candidates),
            keyword_anchored=keyword_anchored,
            raw_text=raw_text,
        )

    def extract(self, image_path: str) -> str | None:
        """
        Convenience method — returns DoB as 'YYYY-MM-DD', or None.
        Used by the debug/compare endpoint.
        """
        return self.extract_with_confidence(image_path).dob
