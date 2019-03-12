import pickle
import os
from os.path import isfile

from architectures import CATBOOST_ENSEMBLE

ARCHITECTURE = CATBOOST_ENSEMBLE.NAME

architecture_mapping = {
    CATBOOST_ENSEMBLE.NAME: CATBOOST_ENSEMBLE
}

import numpy as np
import pandas as pd

class Model:

    def __init__(self, datainfo, timeinfo):
        self._architecture = architecture_mapping[ARCHITECTURE](datainfo, timeinfo)
        
    def fit(self, F, y, datainfo, timeinfo):
        count = {}
        X1 = [
            ['apple', '5', 'dog'],
            ['orange', '2', 'cat'],
            ['pear', '3', 'dog'],
            ['pear', '1', 'dog'],
            ['apple', '1', 'cat'],
            ['apple', '2', 'cat'],
            ['apple', '3', 'dog'],
            ['pear', '4', 'dog']
        ]
        Y = [0, 1, 1, 1, 0, 0, 1, 0]
        X1 = np.array(X1)
        Y = np.array(Y)
        for i in range(X1.shape[1]):
            d0 = pd.DataFrame({'X': X1[:,i]})
            d1 = d0.groupby('X',as_index=True)
            d2 = pd.DataFrame({},index=[])
            d2['COUNT'] = d1.count()
            d3 = d0.join(d2, on='X')[['COUNT']].values
            count[i] = d3
            print(d2)
            print(d3)

        self._architecture.fit(F, y, datainfo, timeinfo)

    def predict(self, F, datainfo, timeinfo):
        return self._architecture.predict(F, datainfo, timeinfo)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self