import torch
from ultralytics import YOLO
from app.utils.logger import get_logger

logger = get_logger(__name__)

class FieldDetector:
    def __init__(self, model_path):
        logger.info("Loading YOLO detector model from %s", model_path)
        self.model = YOLO(model_path)
        logger.info("YOLO detector model loaded")
    
    def detect_batch(self, image_list):
        # image_list is a list of PIL or CV2 images
        # YOLOv8 automatically handles batches efficiently
        logger.info("Running detector batch inference | batch_size=%d", len(image_list))
        results = self.model.predict(image_list, stream=False, batch=16)
        logger.info("Detector batch inference completed")
        return results