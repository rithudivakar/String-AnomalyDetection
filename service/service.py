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
import math



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
        # vectors.add(thisLine.split(";"))
        # String[][] array = new String[vectors.size()][0]
        # array = [["" for j in range(len(vectors))] for i in range(len(vectors))]

        # lines.toArray(array)
        # for (int i = 0; i < vectors.size(); i++):
        #     List<String> vector = vectors.get(i);
        #     array[i] = vector.toArray(new String[0]);

        if len(vectors) > 0:
            vector = sum(vectors) / len(vectors)
    
        else:
            vector = np.zeros(25)
        
        # vectors = np.reshape(vectors.shape[0], -1)
        return vector

    # def toBinary(self, a):
    #     l,m=[],[]
    #     for i in a:
    #         l.append(ord(i))
    #     for i in l:
    #         m.append(int(bin(i)[2:]))
    #     return m

    # def toBinary(self, a):
    #     l = [ord(i) for i in a]
    #     m = np.array([int(bin(i)[2:]) for i in l])
    #     returdef toBinary(self, s):
    
    
    # def toBinary(self, s):
    #     encoded = []
    #     for c in s:
    #         encoded.append(ord(c))
    #     return encoded



    # def load_data(self, filename: str, index_column: str) -> List[Data]:
    #     df = pd.read_csv(filename, index_col=index_column)
        
    #     for i in range(len(df)):
    #         record = list(df.iloc[i])
    #         # text_data = df[record]
    #         encoded_data = [self.word_embedding(text) for text in record]
    #         self.repository.add_data(encoded_data)

    #     return self.repository.get_data()


    # def load_data(self, filename: str, index_column: str) -> List[Data]:
    #     df = pd.read_csv(filename, index_col=index_column)

    #     X = []
    #     for i in range(len(df)):
    #         record = list(df.iloc[i])
    #         encoded_data = [self.word_embedding(text) for text in record]
    #         X.append(np.concatenate(encoded_data))
    #         self.repository.add_data(encoded_data)

    #     X = np.stack(X)

    #     self.repository.add_data([f"feature_{i}" for i in range(X.shape[1])])

    #     return self.repository.get_data()

    def load_data(self, filename: str, index_column: str) -> List[Data]:
        df = pd.read_csv(filename, index_col=index_column)

        X = []
        max_len = 0
        for i in range(len(df)):
            record = list(df.iloc[i])
            encoded_data = [self.word_embedding(text) for text in record]
            max_len = max(max_len, len(encoded_data))
            X.append(encoded_data)
            self.repository.add_data(encoded_data)

        # Pad the feature vectors with zeros so that they all have the same length
        for i in range(len(X)):
            pad_len = max_len - len(X[i])
            if pad_len > 0:
                X[i] = np.concatenate((X[i], np.zeros((pad_len, 25))), axis=0)
        
        X = np.stack([np.concatenate(x) for x in X])

        self.repository.add_data([f"feature_{i}" for i in range(X.shape[1])])

        return self.repository.get_data()


    def detect_anomalies(self) -> List[Data]:
        X = [d.record for d in self.repository.get_data()]
        # X = X.reshape(X.shape[0], -1)
        max_len = max(len(seq) for seq in X)
        X_padded = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')
        arr = np.array(X_padded)
        arr_reshaped = arr.reshape((len(X_padded), -1))
        dataneighbors = NearestNeighbors(n_neighbors=self.k)
        dataneighbors.fit(arr_reshaped)
        distances, indices = dataneighbors.kneighbors(X_padded)
        anomalies = []
        for i in range(len(self.repository.get_data())):
            if distances[i][-1] > 10:
                self.repository.get_data()[i].is_anomaly = True
                anomalies.append(self.repository.get_data()[i])
        return anomalies

    # def detect_anomalies(self) -> List[Data]:
    #     X = [d.record for d in self.repository.get_data()]
    #     # X = np.asarray(X)
    #     dataneighbors = NearestNeighbors(n_neighbors=self.k)
    #     dataneighbors.fit(X)
    #     distances, indices = dataneighbors.kneighbors(X)
    #     anomalies = []
    #     for i in range(len(self.repository.get_data())):
    #         if distances[i][-1] > 10:
    #             self.repository.get_data()[i].is_anomaly = True
    #             anomalies.append(self.repository.get_data()[i])
    #     return anomalies
