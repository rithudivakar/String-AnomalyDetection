from typing import List
from sklearn.neighbors import NearestNeighbors
from entity.entity import Data  
from repository.repository import DataRepository
import pandas as pd
import numpy as np
# from sklearn.impute import SimpleImputer
# from sklearn.impute import KNNImputer
# import Levenshtein
import gensim.downloader as api
import string
import pickle



class AnomalyDetectionService:
    def __init__(self, k: int):
        self.repository = DataRepository()
        self.k = k

        
    def word_embedding(self, text):
        embeddings_dict = {}
        with open("/home/rithuparnakd/projectData/AnomalyDetection/anomalyDetectionString/data/glove.twitter.27B.25d.pkl", "rb") as f:
            data = pickle.load(f, encoding='latin1')
            # data = data.reshape(data.shape[0], -1)
            
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Split the input text into individual words
        words = text.lower().split()
        
        # Calculate the vector representation of the input text by averaging the vectors of the individual words
        vectors = []
        for word in words:
            if word in data:
                vectors.append(data[word])
        if len(vectors) > 0:
            vector = sum(vectors) / len(vectors)
        else:
            vector = np.zeros(25)
        return vector



    def load_data(self, filename: str, index_column: str) -> List[Data]:
        df = pd.read_csv(filename, index_col=index_column)
        
        for i in range(len(df)):
            record = list(df.iloc[i])
            # text_data = df[record]
            encoded_data = [self.word_embedding(text) for text in record]
            self.repository.add_data(encoded_data)

        return self.repository.get_data()



    def detect_anomalies(self) -> List[Data]:
        X = [d.record for d in self.repository.get_data()]
        dataneighbors = NearestNeighbors(n_neighbors=self.k)
        dataneighbors.fit(X)
        distances, indices = dataneighbors.kneighbors(X)
        anomalies = []
        for i in range(len(self.repository.get_data())):
            if distances[i][-1] > 10:
                self.repository.get_data()[i].is_anomaly = True
                anomalies.append(self.repository.get_data()[i])
        return anomalies
