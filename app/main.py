
import os
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import router
from app.services.ocr_service import OCRService
from app.utils.logger import get_logger

os.environ.setdefault("BNB_CUDA_VERSION", "124")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize models before any requests arrive
    logger.info("Server startup: initializing OCRService and loading models")
    app.state.ocr_service = OCRService()
    logger.info("OCRService initialized and models loaded successfully")
    yield
    # Shutdown
    logger.info("Server shutdown")



app = FastAPI(title="CDR OCR Service", lifespan=lifespan)

# CORS settings
origins = [
    "*", 
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/v1")



if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)