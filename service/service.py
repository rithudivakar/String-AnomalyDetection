from typing import List
from sklearn.neighbors import NearestNeighbors
from entity.entity import Data  
from repository.repository import DataRepository
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import Levenshtein

class AnomalyDetectionService:
    def __init__(self, k: int):
        self.repository = DataRepository()
        self.k = k

    # def load_data(self, filename: str, index_column: str) -> List[Data]:
    #     dfString = pd.read_csv(filename, index_col=index_column)
    #     dfint = dfString.apply(pd.to_numeric, errors='coerce')
    #     nan = np.nan
    #     # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    #     imputer = KNNImputer(n_neighbors=2, weights="uniform")
    #     dfimp = imputer.fit_transform(dfint)
    #     # numeric_data.to_csv('/home/rithuparnakd/projectData/AnomalyDetection/anomalyDetectionString/data/numeric_data.csv')
    #     # df = pd.read_csv('/home/rithuparnakd/projectData/AnomalyDetection/anomalyDetectionString/data/numeric_data.csv', index_col=False)
    #     df = pd.DataFrame(dfimp)
    #     for i in range(len(df)):
    #         record = list(df.iloc[i])
    #         self.repository.add_data(record)
    #     return self.repository.get_data()

    # def detect_anomalies(self) -> List[Data]:
    #     X = [d.record for d in self.repository.get_data()]
    #     dataneighbors = NearestNeighbors(n_neighbors=self.k)
    #     dataneighbors.fit(X)
    #     distances, indices = dataneighbors.kneighbors(X)
    #     anomalies = []
    #     for i in range(len(self.repository.get_data())):
    #         if distances[i][-1] > 10:
    #             self.repository.get_data()[i].is_anomaly = True
    #             anomalies.append(self.repository.get_data()[i])
    #     return anomalies

    def load_data(self, filename: str) -> List[Data]:
        df = pd.read_csv(filename)
        df_numeric = df.select_dtypes(include=[np.number])
        imputer = KNNImputer(n_neighbors=2, weights="uniform")
        df_imp = imputer.fit_transform(df_numeric)
        for i in range(len(df_imp)):
            record = list(df_imp[i])
            self.repository.add_data(record)
        return self.repository.get_data()

    def detect_anomalies(self) -> List[Data]:
        X = [d.record for d in self.repository.get_data()]
        dataneighbors = NearestNeighbors(n_neighbors=self.k, metric=Levenshtein.distance)
        dataneighbors.fit(X)
        distances, indices = dataneighbors.kneighbors(X)
        anomalies = []
        for i in range(len(self.repository.get_data())):
            if distances[i][-1] > 10:
                self.repository.get_data()[i].is_anomaly = True
                anomalies.append(self.repository.get_data()[i])
        return anomalies

