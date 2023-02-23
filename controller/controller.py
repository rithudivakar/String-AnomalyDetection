from typing import List
from entity.entity import Data
from service.service import AnomalyDetectionService

class AnomalyDetectionController:
    def __init__(self, k: int):
        self.service = AnomalyDetectionService(k)
    
    def load_data(self, filename: str) -> List[Data]:
        return self.service.load_data(filename)
    
    def detect_anomalies(self) -> List[Data]:
        return self.service.detect_anomalies()