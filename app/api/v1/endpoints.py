
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image
import io

from app.services.ocr_service import OCRService
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Singleton — initialized once on first request, not on every call
_ocr_service: OCRService | None = None


def get_ocr_service() -> OCRService:
    global _ocr_service
    if _ocr_service is None:
        logger.info("Initializing OCRService singleton")
        _ocr_service = OCRService()
    return _ocr_service


@router.get("/health")
def health_check():
    logger.info("Health check endpoint called")
    return {"status": "service is running"}

@router.post("/process-form")
async def process_form_api(file: UploadFile = File(...)):
    logger.info(
        "Received /process-form request | filename=%s | content_type=%s",
        file.filename,
        file.content_type,
    )

    if not file.content_type.startswith("image/"):
        logger.warning("Rejected non-image upload | content_type=%s", file.content_type)
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        logger.info("Uploaded file read successfully | bytes=%d", len(contents))
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.info("Image decoded successfully | size=%s", image.size)
    except Exception as e:
        logger.exception("Failed to read/parse uploaded image")
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    try:
        logger.info("Starting OCR pipeline execution")
        data, needs_review = await get_ocr_service().process_form(image)
        logger.info("OCR pipeline completed | needs_review=%s", needs_review)
    except Exception as e:
        logger.exception("OCR processing failed")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")

    return {"data": data, "needs_review": needs_review}