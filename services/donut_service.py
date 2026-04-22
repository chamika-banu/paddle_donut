from __future__ import annotations

import re
import torch
from datetime import date, datetime
from dateutil import parser as dateutil_parser
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

from transformers import AutoProcessor, VisionEncoderDecoderModel
from services.constants import REFUSAL_ANSWERS


class DonutService:
    """
    Donut (NAVER CLOVA OCR) service — used for Tier 2 escalation.

    Donut (~500 MB) is a vision-language model optimized for document
    understanding and OCR.
    """

    MODEL_ID = "naver-clova-ix/donut-base-finetuned-docvqa"
    # Pinned to a specific commit hash for reproducibility.
    # Source: huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa/commits/main
    MODEL_REVISION = "b19d2e332684b0e2d35d9144ce34047767335cf8"

    def __init__(self) -> None:
        """Initialize Donut model and processor."""
        if torch.cuda.is_available():
            self.device = "cuda"
            display_device = "GPU"
            print("[DonutService] GPU detected, using CUDA")
        else:
            self.device = "cpu"
            display_device = "CPU"
            print("[DonutService] No GPU available, using CPU")

        print(f"[DonutService] Loading model '{self.MODEL_ID}' on {display_device}...")
        print("[DonutService] This may take 1-2 minutes on first run (downloading ~500 MB)...")

        # Load processor (handles image preprocessing and tokenization)
        print("[DonutService] Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            revision=self.MODEL_REVISION,
            trust_remote_code=True,
        )

        # Load vision-encoder-decoder model
        print("[DonutService] Loading model (~500 MB)...")
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.MODEL_ID,
            revision=self.MODEL_REVISION,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        # Set to eval mode (disable dropout, batch norm)
        self.model.eval()

        print(f"[DonutService] Model loaded successfully on {self.device}")
        print(f"[DonutService] Revision: {self.MODEL_REVISION}")
        print("[DonutService] Model ready.")

    def _load_image(self, image_path: str) -> Image.Image:
        """
        Load and preprocess image.

        Args:
            image_path: Path to image file (JPEG, PNG, WebP)

        Returns:
            PIL Image in RGB format, with EXIF rotation applied
        """
        try:
            img = Image.open(image_path)
            # Apply EXIF-based rotation (handles phone photos)
            img = ImageOps.exif_transpose(img)
            # Ensure RGB (handles grayscale, RGBA, etc.)
            return img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_path}: {e}")

    def extract_dob_vqa(self, image_path: str) -> str | None:
        """
        Extract Date of Birth from ID document using visual Q&A.

        This method:
        1. Loads the document image
        2. Encodes it with Donut's vision encoder
        3. Prompts: "What is the date of birth?"
        4. Decodes the model's response
        5. Parses and normalizes the extracted date

        Args:
            image_path: Path to ID document image

        Returns:
            Date in YYYY-MM-DD format if found, else None
        """
        try:
            # Load and preprocess image
            image = self._load_image(image_path)

            # Prepare the task prompt for Donut DocVQA
            # naver-clova-ix/donut-base-finetuned-docvqa uses <s_docvqa><s_question>...</s_question><s_answer>
            question = "What is the date of birth?"
            task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"

            # Encode image — DonutProcessor requires images= keyword argument
            image_tensor = self.processor(
                images=image,             # ← must use keyword arg, not positional
                return_tensors="pt"
            ).pixel_values.to(self.device).to(self.model.dtype)

            # Prepare decoder input (task prompt tokens)
            decoder_input_ids = self.processor.tokenizer(
                task_prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids.to(self.device)

            # Run inference
            # Suppress gradients for faster inference
            with torch.no_grad():
                # Generate output tokens
                # early_stopping: Stop when EOS token is generated
                # use_cache: Use KV cache for faster decoding
                generated_ids = self.model.generate(
                    pixel_values=image_tensor,
                    decoder_input_ids=decoder_input_ids,
                    max_length=256,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,  # Greedy decoding (fastest)
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],  # Penalize UNK
                )

            # Decode tokens back to text
            # batch_decode: Handle batches (we have batch_size=1)
            # We MUST KEEP special tokens (skip_special_tokens=False),
            # otherwise <s_answer> is stripped out and the question
            # text merges with the answer text.
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )[0].strip()            

            # Validate extracted text
            if "<s_answer>" in generated_text:
                generated_text = generated_text.split("<s_answer>")[-1]
            if "</s_answer>" in generated_text:
                generated_text = generated_text.split("</s_answer>")[0]
            
            generated_text = generated_text.strip()            

            if not generated_text or generated_text.lower() in REFUSAL_ANSWERS:
                return None

            # Try to parse as a date
            return self._parse_date_string(generated_text)

        except Exception as e:
            print(f"[DonutService] Error during extraction: {e}")
            return None

    def _parse_date_string(self, date_string: str) -> str | None:
        """
        Parse a date string into YYYY-MM-DD format.

        Handles various date formats:
        - DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
        - YYYY/MM/DD, YYYY-MM-DD
        - DD Month YYYY, DD Mon YYYY (e.g., "12 April 1989")

        Args:
            date_string: Raw date string from model output

        Returns:
            YYYY-MM-DD string if parseable, else None
        """
        try:
            # Determine date format: Year first or day/month first?
            year_first = bool(re.match(r'^\d{4}', date_string.strip()))

            # Use dateutil parser (handles many formats automatically)
            parsed_date: datetime = dateutil_parser.parse(
                date_string,
                dayfirst=not year_first,
                yearfirst=year_first,
            )

            # Sanity checks
            today = date.today()
            parsed = parsed_date.date()

            # Reject future dates
            if parsed >= today:
                return None

            # Reject implausibly old dates (>120 years)
            if today.year - parsed.year > 120:
                return None

            # Convert to standard format
            normalized = parsed.strftime("%Y-%m-%d")
            return normalized

        except (ValueError, OverflowError, AttributeError):
            # dateutil couldn't parse the string
            return None

    def compute_age(self, dob_str: str) -> int:
        """
        Calculate age in complete years from a YYYY-MM-DD date string.

        Args:
            dob_str: Date of birth as YYYY-MM-DD

        Returns:
            Age as integer (number of complete years)

        Example:
            - DoB: "2005-03-15", Today: "2024-03-14" → Age: 18
            - DoB: "2005-03-15", Today: "2024-03-15" → Age: 19
        """
        dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
        today = date.today()
        age = today.year - dob.year

        # Subtract 1 if birthday hasn't occurred yet this year
        if (today.month, today.day) < (dob.month, dob.day):
            age -= 1

        return age

    @staticmethod
    def get_device_info() -> dict:
        """
        Get information about the device being used.

        Returns:
            Dict with device, cuda availability, etc.
        """
        return {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
