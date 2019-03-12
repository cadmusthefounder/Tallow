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
from collections import Counter

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
        X1 = np.array(X1)
        for i in range(X1.shape[1]):
            df= pd.DataFrame({'X1': X1[:,i]})
            count_i = Counter(df['X1'])
            count[i] = dict(count_i)

        print(count)
        X2 = [
            ['apple', '1', 'cat'],
            ['orange', '2', 'cat'],
            ['pear', '5', 'dog'],
            ['pear', '5', 'cat'],
            ['orange', '1', 'cat'],
            ['orange', '5', 'cat'],
            ['orange', '3', 'dog'],
            ['grapes', '2', 'dog']
        ]
        X2 = np.array(X2)
        for i in range(X2.shape[1]):
            df= pd.DataFrame({'X2': X2[:,i]})
            count_i = Counter(df['X2'])
            count[i].update(count_i)
        print(count)
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