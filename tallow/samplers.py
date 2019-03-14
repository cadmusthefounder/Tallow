from imblearn.over_sampling import SMOTE
from collections import Counter, OrderedDict
from math import pow
from random import random
import numpy as np
from sklearn.utils import check_random_state, safe_indexing

class SMOTESampler:

    def __init__(self):
        self._sampler = SMOTE()

    def sample(self, X, y):
        return self._sampler.fit_resample(X, y)

class RandomOverSampler:
    
    def __init__(self, random_state):
        self._random_state = random_state
    
    def sample(self, X, y):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('samplers.py', 'RandomOverSampler', 'sample', 'Start'))
        random_state = check_random_state(self._random_state)
        target_stats = Counter(y)

        sample_indices = range(X.shape[0])
        sampling_strategy_ = self._check_sampling_strategy(y)
        for class_sample, num_samples in sampling_strategy_.items():
            target_class_indices = np.flatnonzero(y == class_sample)
            indices = random_state.randint(
                low=0, high=target_stats[class_sample], size=num_samples)

            sample_indices = np.append(sample_indices,
                                       target_class_indices[indices])

        print('File: {} Class: {} Function: {} State: {} \n'.format('samplers.py', 'RandomOverSampler', 'sample', 'End'))
        return (safe_indexing(X, sample_indices),
                safe_indexing(y, sample_indices))

    def _check_sampling_strategy(self, y):
        return OrderedDict(sorted(self._sampling_strategy_not_majority(y).items()))

    def _count_class_sample(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))
    
    def _sampling_strategy_not_majority(self, y):
        target_stats = self._count_class_sample(y)
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items() if key != class_majority
        }
        return sampling_strategy

class RandomUnderSampler:

    def __init__(self, random_state, replacement=False):
        self._random_state = random_state
        self._replacement = replacement

    def sample(self, X, y):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('samplers.py', 'RandomUnderSampler', 'sample', 'Start'))
        random_state = check_random_state(self._random_state)
        idx_under = np.empty((0, ), dtype=int)

        sampling_strategy_ = self._check_sampling_strategy(y)
        for target_class in np.unique(y):
            if target_class in sampling_strategy_.keys():
                n_samples = sampling_strategy_[target_class]
                index_target_class = random_state.choice(
                    range(np.count_nonzero(y == target_class)),
                    size=n_samples,
                    replace=self._replacement)
            else:
                index_target_class = slice(None)

            idx_under = np.concatenate(
                (idx_under,
                 np.flatnonzero(y == target_class)[index_target_class]),
                axis=0)

        print('File: {} Class: {} Function: {} State: {} \n'.format('samplers.py', 'RandomUnderSampler', 'sample', 'End'))
        return safe_indexing(X, idx_under), safe_indexing(y, idx_under)

    def _check_sampling_strategy(self, y):
        return OrderedDict(sorted(self._sampling_strategy_not_minority(y).items()))

    def _count_class_sample(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))

    def _sampling_strategy_not_minority(self, y):
        target_stats = self._count_class_sample(y)
        n_sample_minority = min(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_minority
            for key in target_stats.keys() if key != class_minority
        }
        return sampling_strategy

class RandomSampler:

    def __init__(self, max_data):
        self.max_data = max_data

    def sample(self, X, y):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('samplers.py', 'RandomSampler', 'sample', 'Start'))
        
        if len(X) < self.max_data:
            return X, y
        indices = np.sort(np.random.choice(len(X), self.max_data, replace=False))
        print('File: {} Class: {} Function: {} State: {} \n'.format('samplers.py', 'RandomSampler', 'sample', 'End'))
        return X[indices,:], y[indices]

class BiasedReservoirSampler:

    def __init__(self, capacity, bias_rate, info):
        self._capacity = capacity
        self._bias_rate = bias_rate
        self._size = 10000
        self._p_in = self._capacity * self._bias_rate
        self._p_in_index = 0
        self._p_in_array = self._generate_p_in_array()
        
        self._current_index = 0
        self._indices = self._generate_indices()
        
        self._current_capacity = 0
        self._current_reservoir_data = np.empty([self._capacity, info['total_no_of_features']], dtype=object)
        self._current_reservoir_label = np.empty(self._capacity)
        
    def sample(self, incoming_data, incoming_labels):
        print('\nsample')
        print('incoming_data.shape: {}'.format(incoming_data.shape))
        print('incoming_labels.shape: {}'.format(incoming_labels.shape))
        
        for i in range(len(incoming_data)):
            if self._current_capacity < self._capacity or self._triggered():
                if self._current_index >= len(self._indices):
                    self._current_index = 0
                    self._indices = self._generate_indices()

                if self._indices[self._current_index] < self._current_capacity:
                    self._current_reservoir_data[self._indices[self._current_index]] = incoming_data[i]
                    self._current_reservoir_label[self._indices[self._current_index]] = incoming_labels[i]
                else:
                    self._current_reservoir_data[self._current_capacity] = incoming_data[i]
                    self._current_reservoir_label[self._current_capacity] = incoming_labels[i]
                    self._current_capacity += 1
                
                self._current_index += 1

        actual_reservoir_data = self._current_reservoir_data
        actual_reservoir_labels = self._current_reservoir_label
        if self._current_capacity < self._capacity:
            actual_reservoir_data = actual_reservoir_data[:self._current_capacity,:]
            actual_reservoir_labels = actual_reservoir_labels[:self._current_capacity]

        print('actual_reservoir_data.shape: {}'.format(actual_reservoir_data.shape))
        print('actual_reservoir_labels.shape : {}\n'.format(actual_reservoir_labels.shape))
        return actual_reservoir_data, actual_reservoir_labels

    def _triggered(self):
        if self._p_in_index >= len(self._p_in_array):
            self._p_in_index = 0
            self._p_in_array = self._generate_p_in_array()

        triggered = self._p_in_array[self._p_in_index] <= self._p_in
        self._p_in_index += 1
        return triggered

    def _generate_indices(self):
        return np.random.randint(self._capacity, size=self._size)

    def _generate_p_in_array(self):
        return np.random.random_sample(size=self._size)