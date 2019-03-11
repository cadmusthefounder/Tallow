from collections import Counter, OrderedDict
import numpy as np
from sklearn.utils import check_random_state, safe_indexing

class RandomOverSampler:
    
    def __init__(self, random_state):
        self._random_state = random_state
    
    def sample(self, X, y):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('samplers.py', 'RandomOverSampler', 'sample', 'Start'))
        random_state = check_random_state(self._random_state)
        target_stats = Counter(y)

        sample_indices = range(X.shape[0])
        sampling_strategy_ = self._check_sampling_strategy(y, )
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

class RandomSampler:

    def __init__(self, max_data):
        self._max_data = max_data

    def sample(self, X, y):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('samplers.py', 'RandomSampler', 'sample', 'Start'))
        indices = np.sort(np.random.choice(len(X), self._max_data, replace=False))
        print('File: {} Class: {} Function: {} State: {} \n'.format('samplers.py', 'RandomSampler', 'sample', 'End'))
        return X[indices,:], y[indices]
