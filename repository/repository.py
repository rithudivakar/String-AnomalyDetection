from typing import List
from entity.entity import Data

class DataRepository:
    def __init__(self):
        self.data = []

    def add_data(self, record):
        data = Data(record)
        self.data.append(data)
        return data

    def get_data(self) -> List[Data]:
        return self.data
