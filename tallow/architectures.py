from utils import *
pip_install('catboost')
pip_install('lightgbm')
pip_install('hyperopt')

import numpy as np
from math import pow
from catboost import Pool
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from hyperparameters_tuner import HyperparametersTuner
from profiles import Profile
from samplers import RandomOverSampler, RandomUnderSampler, RandomSampler

class DataType:
    TRAIN = 'TRAIN'
    VALIDATION = 'VALIDATION'
    TEST = 'TEST'

class Original:
    NAME = 'Original'

    def __init__(self, datainfo, timeinfo):
        self._info = extract(datainfo, timeinfo)
        print_data_info(self._info)
        print_time_info(self._info)

        self._use_validation = True
        self._large_dataset_validation_ratio = 0 if not self._use_validation else 0.1
        self._small_dataset_validation_ratio = 0 if not self._use_validation else 0.25
        self._early_stopping_rounds = 0 if not self._use_validation else 20

        self._dataset_size_threshold = 400000
        self._large_dataset_max_data = 300000
        self._small_dataset_max_data = 400000

        self._iteration = 0
        self._random_state = 13
        self._max_evaluations = 2
        self._dataset_budget_threshold = 0.8

        self._category_indices = None
        self._categorical_frequency_map = {}
        self._mvc_frequency_map = {}
        
        self._best_hyperparameters = None
        
        self._train_data = np.array([])
        self._train_labels = np.array([])

        self._under_sampler = RandomUnderSampler(self._random_state)
        self._sampler = None
        self._profile = Profile.LGBM_ORIGINAL_NAME

    def fit(self, F, y, datainfo, timeinfo):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('architectures.py', 'Original', 'fit', 'Start'))

        info = extract(datainfo, timeinfo)
        self._info.update(info)
        print_time_info(self._info)

        data = get_data(F, self._info)
        y = y.ravel()
        print('data.shape: {}'.format(data.shape))
        print('y.shape: {}'.format(y.shape))

        bincount = np.bincount(y.astype(int))
        print('Number of 0 label: {}'.format(bincount[0]))
        print('Number of 1 label: {}'.format(bincount[1]))

        max_data = self._large_dataset_max_data if is_large_dataset(len(data), self._dataset_size_threshold) else self._small_dataset_max_data
        validation_ratio = self._large_dataset_validation_ratio if is_large_dataset(len(data), self._dataset_size_threshold) else self._small_dataset_validation_ratio
        self._sampler = RandomSampler(max_data) if self._sampler is None else self._sampler

        train_data, self._validation_data, train_labels, self._validation_labels = train_test_split(
            data,
            y,
            test_size=validation_ratio,
            random_state=self._random_state,
            shuffle=True,
            stratify=y
        )
        print('train_data.shape: {}'.format(train_data.shape))
        print('train_labels.shape: {}'.format(train_labels.shape))
        print('self._validation_data.shape: {}'.format(self._validation_data.shape))
        print('self._validation_labels.shape: {}'.format(self._validation_labels.shape))

        train_data, train_labels = self._under_sampler.sample(train_data, train_labels)
        print('train_data.shape: {}'.format(train_data.shape))
        print('train_labels.shape: {}'.format(train_labels.shape))

        self._train_data = train_data if len(self._train_data) == 0 else np.concatenate((self._train_data, train_data), axis=0)
        self._train_labels = train_labels if len(self._train_labels) == 0 else np.concatenate((self._train_labels, train_labels), axis=0)
        self._train_data, self._train_labels = self._sampler.sample(self._train_data, self._train_labels)
        print('self._train_data.shape: {}'.format(self._train_data.shape))
        print('self._train_labels.shape: {}'.format(self._train_labels.shape))

        print('File: {} Class: {} Function: {} State: {} \n'.format('architectures.py', 'Original', 'fit', 'End'))
        
    def predict(self, F, datainfo, timeinfo):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('architectures.py', 'Original', 'predict', 'Start'))
    
        info = extract(datainfo, timeinfo)
        self._info.update(info)
        print_time_info(self._info)

        test_data = get_data(F, self._info)
        print('test_data.shape: {}'.format(test_data.shape))

        classification_class, fixed_hyperparameters, search_space = Profile.parse_profile(self._profile, self._early_stopping_rounds)

        test_data = self._transform(test_data, DataType.TEST)
        train_data = self._transform(self._train_data, DataType.TRAIN)
        validation_data = self._transform(self._validation_data, DataType.VALIDATION)

        print('test_data.shape: {}'.format(test_data.shape))
        print('train_data.shape: {}'.format(train_data.shape))
        print('self._train_labels.shape: {}'.format(self._train_labels.shape))
        print('validation_data.shape: {}'.format(validation_data.shape))
        print('self._validation_labels.shape: {}'.format(self._validation_labels.shape))
        
        train_pool = Pool(train_data, self._train_labels)
        validation_pool = Pool(validation_data, self._validation_labels)
        validation_set = (validation_data, self._validation_labels)
        
        if self._best_hyperparameters is None:
            tuner = HyperparametersTuner(classification_class, fixed_hyperparameters, search_space, self._max_evaluations)
            self._best_hyperparameters = tuner.get_best_hyperparameters(train_pool, validation_pool)
            print('self._best_hyperparameters: {}'.format(self._best_hyperparameters))

        if has_sufficient_time(self._dataset_budget_threshold, self._info):
            classifier = classification_class(**self._best_hyperparameters)
            
            if isinstance(classifier, LGBMClassifier):                
                classifier.fit(training_data, training_labels, eval_set=validation_set)
            else:
                classifier.fit(train_pool, eval_set=validation_pool)

            # if len(self._classifiers) > 1:
            #     probabilities = np.zeros(validation_pool.num_row())
            #     for i in range(len(self._classifiers)):
            #         if i == 0:
            #             probabilities = self._classifiers[i].predict_proba(validation_pool)[:,1]
            #         else:
            #             probabilities = np.vstack((probabilities, self._classifiers[i].predict_proba(validation_pool)[:,1]))
            #     probabilities = np.transpose(probabilities)
            #     self._lr = LogisticRegression()
            #     self._lr.fit(probabilities, validation_pool.get_label())
        else:
            print('Time budget exceeded.')

        probabilities = self._classifier.predict_proba(test_data)[:,1]
        print('probabilities.shape: {}'.format(probabilities.shape))
        print('File: {} Class: {} Function: {} State: {} \n'.format('architectures.py', 'Original', 'predict', 'End'))
        
        self._iteration += 1
        return probabilities

    def _transform(self, data, datatype):
        transformed_data = np.array([])
        time_data, numerical_data, categorical_data, mvc_data = split_data_by_type(data, self._info)
        if len(time_data) > 0:
            transformed_data = subtract_min_time(time_data)
            transformed_data = np.concatenate((transformed_data, difference_between_time_columns(time_data)), axis=1)
        if len(numerical_data) > 0:
            transformed_data = numerical_data if len(transformed_data) == 0 else \
                                np.concatenate((transformed_data, numerical_data), axis=1)
        if len(categorical_data) > 0:
            if (datatype == DataType.TRAIN and self._iteration == 0) or datatype == DataType.TEST:
                count_frequency(self._categorical_frequency_map, categorical_data)
            encoded_categorical_data = encode_frequency(self._categorical_frequency_map, categorical_data)
            transformed_data = np.concatenate((transformed_data, encoded_categorical_data), axis=1)
        if len(mvc_data) > 0: 
            if (datatype == DataType.TRAIN and self._iteration == 0) or datatype == DataType.TEST:
                count_frequency(self._mvc_frequency_map, mvc_data)
            
            encoded_mvc_data = encode_frequency(self._mvc_frequency_map, mvc_data)
            transformed_data = np.concatenate((transformed_data, encoded_mvc_data), axis=1)
        return transformed_data
