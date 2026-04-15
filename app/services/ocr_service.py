
import asyncio
import json
import re
import traceback
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image
from app.ml.detector import FieldDetector
from app.ml.recognizer import BengaliRecognizer, EnglishRecognizer
from app.utils.image_processing import crop_field, prepare_for_recognition
from app.utils.logger import get_logger

REVIEW_DIR = Path("data/flagged_human")
CONFIDENCE_THRESHOLD = 0.50
logger = get_logger(__name__)

# Only "name" field uses the Bengali (Qwen VL) recognizer.
# All other fields — partner_code, age, phn_number, brand, date — use TrOCR.
BENGALI_FIELDS = {"name"}
SPECIAL_FIXED_FIELDS = {
    "customer_copy": ("customer_copy", "customer copy"),
    "sign": ("sign", "signed"),
    "cheek_mark": ("cheek_mark", "checked"),
    "check_mark": ("cheek_mark", "checked"),
    "cheek_mark_0": ("cheek_mark", "unchecked"),
    "check_mark_0": ("cheek_mark", "unchecked"),
}


def _digits_only(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def _format_date_dd_mm_yyyy(text: str) -> str:
    digits = _digits_only(text)
    if len(digits) < 8:
        return ""
    digits = digits[:8]
    return f"{digits[0:2]} {digits[2:4]} {digits[4:8]}"


class OCRService:
    def __init__(self):
        logger.info("Initializing OCRService components")
        self.detector = FieldDetector("models/detector/cdr_book_modify_training_dataset_v1.pt")
        self.bengali_recognizer = None
        try:
            self.bengali_recognizer = BengaliRecognizer()
            logger.info("BengaliRecognizer initialized successfully")
        except Exception as e:
            # Keep service alive even if Bengali model fails to initialize.
            logger.exception("BengaliRecognizer initialization failed: %s", e)
            traceback.print_exc()

        self.english_recognizer = EnglishRecognizer()
        logger.info("EnglishRecognizer initialized successfully")
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Review directory ready at %s", REVIEW_DIR)

    async def process_form(
        self, image: Image.Image
    ) -> Tuple[Dict[str, Dict[str, object]], bool]:
        logger.info("OCRService.process_form started")
        detections = self.detector.detect_batch([image])[0]
        final_result: Dict[str, Dict[str, object]] = {}
        flag_for_human = False
        loop = asyncio.get_event_loop()

        detected_labels = {
            detections.names[int(box.cls[0])].lower().strip() for box in detections.boxes
        }
        logger.info(
            "Detection completed | boxes=%d | labels=%s",
            len(detections.boxes),
            sorted(detected_labels),
        )

        # Validate page type by required anchor label.
        if "customer_copy" not in detected_labels:
            logger.warning("Invalid page detected: 'customer_copy' label missing")
            return {
                "error": {
                    "text": "this is not valid customer page",
                    "confidence": 0.0,
                }
            }, True

        for box in detections.boxes:
            # Normalize to lowercase — detector returns "name", "partner_code" etc.
            field_name = detections.names[int(box.cls[0])].lower().strip()
            detector_confidence = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
            logger.info(
                "Processing field | name=%s | detector_confidence=%.2f",
                field_name,
                detector_confidence,
            )

            # Fixed-value labels (no OCR)
            if field_name in SPECIAL_FIXED_FIELDS:
                output_key, output_text = SPECIAL_FIXED_FIELDS[field_name]
                logger.info(
                    "Applying fixed output (no OCR) | field=%s | output_key=%s | output_text=%s",
                    field_name,
                    output_key,
                    output_text,
                )

                # Prefer "unchecked" over "checked" if both labels appear.
                if output_key == "cheek_mark":
                    if output_text == "unchecked" or output_key not in final_result:
                        final_result[output_key] = {
                            "text": output_text,
                            "confidence": round(detector_confidence, 2),
                        }
                else:
                    final_result[output_key] = {
                        "text": output_text,
                        "confidence": round(detector_confidence, 2),
                    }

                if detector_confidence < CONFIDENCE_THRESHOLD:
                    flag_for_human = True
                    logger.info(
                        "Flagged for human review due to low detector confidence on fixed field | field=%s",
                        field_name,
                    )
                continue

            raw_crop = crop_field(image, box.xyxy[0])
            processed_crop = prepare_for_recognition(raw_crop)

            try:
                if field_name in BENGALI_FIELDS and self.bengali_recognizer is not None:
                    logger.info("Using Bengali recognizer for field=%s", field_name)
                    text, confidence = await loop.run_in_executor(
                        None, self.bengali_recognizer.inference, processed_crop
                    )
                elif field_name in BENGALI_FIELDS and self.bengali_recognizer is None:
                    # Bengali model unavailable; continue with fallback but flag review.
                    flag_for_human = True
                    logger.warning(
                        "Bengali recognizer unavailable, falling back to English recognizer | field=%s",
                        field_name,
                        )
                    final_result[field_name] = {
                            "text": "",
                            "confidence": 0.0,
                        }
                    continue
                else:   
                    logger.info("Using English recognizer for field=%s", field_name)
                    text, confidence = await loop.run_in_executor(
                        None, self.english_recognizer.inference, processed_crop
                    )
            except Exception as e:
                # Don't let one bad crop crash the whole form
                text = ""
                confidence = 0.0
                flag_for_human = True
                logger.exception("Inference failed for field '%s': %s", field_name, e)

            # Field-specific normalization rules
            if field_name == "age":
                text = _digits_only(text)
            elif field_name == "date":
                text = _format_date_dd_mm_yyyy(text)

            logger.info(
                "Field result | field=%s | text=%s | confidence=%.2f",
                field_name,
                text,
                float(confidence),
            )

            if confidence < CONFIDENCE_THRESHOLD:
                flag_for_human = True
                logger.info(
                    "Flagged for human review due to low OCR confidence | field=%s | confidence=%.2f",
                    field_name,
                    float(confidence),
                )

            final_result[field_name] = {
                "text": text,
                "confidence": round(float(confidence), 2),
            }

        if flag_for_human:
            logger.info("Sample flagged for human review; saving artifacts")
            self._save_for_review(image, final_result)

        logger.info("OCRService.process_form finished | needs_review=%s", flag_for_human)
        return final_result, flag_for_human

    def _save_for_review(
        self, image: Image.Image, result: Dict[str, object]
    ) -> None:
        review_index = len(list(REVIEW_DIR.glob("review_*.png"))) + 1
        image_path = REVIEW_DIR / f"review_{review_index}.png"
        metadata_path = REVIEW_DIR / f"review_{review_index}.json"

        image.save(image_path, format="PNG")
        metadata_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Saved review artifacts | image=%s | metadata=%s", image_path, metadata_path)
