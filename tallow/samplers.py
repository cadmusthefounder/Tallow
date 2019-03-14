from imblearn.under_sampling import RandomUnderSampler
from collections import Counter, OrderedDict
from math import pow
from random import random
import numpy as np
from sklearn.utils import check_random_state, safe_indexing
from sklearn.model_selection import train_test_split

class RandomMajorityUnderSampler:

    def __init__(self, random_state, replacement=False):
        self._sampler = RandomUnderSampler(random_state=random_state, replacement=replacement)

    def sample(self, X, y):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('samplers.py', 'RandomMajorityUnderSampler', 'sample', 'Start'))
        sampled_X, sampled_y = self._sampler.fit_resample(X, y)

        print('File: {} Class: {} Function: {} State: {} \n'.format('samplers.py', 'RandomMajorityUnderSampler', 'sample', 'End'))
        return sampled_X, sampled_y 

class StratifiedRandomSampler:

    def __init__(self, max_data, random_state):
        self._max_data = max_data
        self._random_state = random_state

    def sample(self, X, y):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('samplers.py', 'StratifiedRandomSampler', 'sample', 'Start'))
        
        if len(X) < self._max_data:
            return X, y

        ratio = float(len(X)) / float(self._max_data)
        train_data, test_data, train_labels, test_labels = train_test_split(
            X, 
            y, 
            test_size=ratio, 
            random_state=self._random_state,
            shuffle=True, 
            stratify=y
        )
        print('File: {} Class: {} Function: {} State: {} \n'.format('samplers.py', 'StratifiedRandomSampler', 'sample', 'End'))
        return train_data, train_labels
