# apps/detection/services.py
from .ml_logic.inference import AnomalyDetectionInference
import os


class DetectionService:
    def __init__(self):
        self._engine = None

    @property
    def engine(self):
        # Load model artifacts on first use, not at import time
        if self._engine is None:
            models_path = os.path.join(os.getcwd(), "data", "models")
            self._engine = AnomalyDetectionInference(models_path=models_path)
        return self._engine

    def predict(self, features: dict):
        return self.engine.predict_single(features)


detection_service = DetectionService()
