import re

# DoB Extraction Constants and Patterns

# Covers the most common date formats found on government IDs:
#   12/04/1989   12-04-1989   12.04.1989
#   1989/04/12   1989-04-12
#   12 April 1989   12 Apr 1989
DOB_PATTERNS = [
    r'\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\b',
    r'\b(\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2})\b',
    r'\b(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
    r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
    r'\s+\d{4})\b',
]

# Anchor keywords indicating a date is likely a DoB (not expiry/issue).
DOB_CONTEXT_KEYWORDS = [
    'birth', 'dob', 'd.o.b', 'born', 'date of birth',
]

# Regex used by Paddle for quick date presence check
PATTERN_YYYY_MM_DD = re.compile(
    r"\d{4}[-/. \\]*(?:0[1-9]|1[0-2]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[-/. \\]*(?:0[1-9]|[12][0-9]|3[01])",
    re.IGNORECASE,
)

# Standard model refusal answers or common "empty" values
REFUSAL_ANSWERS = {"not found", "none", "n/a", "unknown", ""}

# ESCALATION THRESHOLDS
# Tier 2 (Donut). Tune these based on real-world accuracy data.

# If the total raw text from the ID is shorter than this, it's likely blurry,
# cropped, or low-resolution to trust Tier 1. Escalate to Donut.
ESCALATION_MIN_TEXT_LENGTH = 20

# If Paddle extracts MORE than this number of dates, and NONE of them are anchored
# by a keyword like "DOB", the document is considered too dense/ambiguous.
# Escalate to Donut to resolve.
ESCALATION_MAX_UNANCHORED_CANDIDATES = 1
