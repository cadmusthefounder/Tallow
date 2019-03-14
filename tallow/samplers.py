from imblearn.under_sampling import RandomUnderSampler
from collections import Counter, OrderedDict
from math import pow
from random import random
import numpy as np
from sklearn.utils import check_random_state, safe_indexing
from sklearn.model_selection import train_test_split

class OldRandomMajorityUnderSampler:

    def __init__(self, random_state, fraction=3):
        self._random_state = random_state
        self._fraction = fraction

    def sample(self, X, y):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('samplers.py', 'OldRandomMajorityUnderSampler', 'sample', 'Start'))
        class_0_freq = len(y[y==0])
        class_1_freq = len(y[y==1])
        majority_class = 0
        if class_1_freq>class_0_freq:
            majority_class = 1
            minority_count = class_0_freq
        else:
            minority_count = class_1_freq

        minority_class = int(not majority_class)
        indices = np.array(range(len(y)))
        majority_ind = indices[y==majority_class]
        minority_index = indices[y==minority_class]
        
        np.random.seed(self._random_state)
        if int(minority_count * self._fraction) > len(majority_ind):
            size = len(majority_ind)
        else:
            size = int(minority_count * self._fraction)
        majority_index = np.random.choice(indices[y==majority_class],size=size,replace=False)
        sorted_index = sorted(np.concatenate([minority_index,majority_index]))
        print('File: {} Class: {} Function: {} State: {} \n'.format('samplers.py', 'OldRandomMajorityUnderSampler', 'sample', 'End'))
        return X[sorted_index],y[sorted_index]

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
