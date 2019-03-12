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
from utils import *

class Model:

    def __init__(self, datainfo, timeinfo):

        # X1 = [
        #     ['apple', '5', '1'],
        #     ['apple', '5', '1'],
        #     ['pear', '2', '0'],
        #     ['apple', '2', '1'],
        #     ['apple', '0', '1'],
        #     ['pear', '3', '0']
        # ]
        # df = pd.DataFrame(X1)
        # counts = count_frequency(X1)
        # print(df)
        # print(counts)
        
        # new_df = pd.DataFrame()
        # for x in df.columns:
        #     new_df[x] = df[x].apply(lambda row: counts[x][row])
        # print(new_df)
        self._architecture = architecture_mapping[ARCHITECTURE](datainfo, timeinfo)
        
    def fit(self, F, y, datainfo, timeinfo):
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