
import os

# Keep configurable from environment.
# If not provided, default to CUDA 12.4 build for bitsandbytes.
# To disable manually: export BNB_CUDA_VERSION=
os.environ.setdefault("BNB_CUDA_VERSION", "124")

from fastapi import FastAPI
from app.api.v1.endpoints import router

app = FastAPI(title="CDR OCR Service")
app.include_router(router, prefix="/v1")