from utils import *
pip_install('catboost')
pip_install('hyperopt')

import numpy as np
from math import pow
from catboost import CatBoostClassifier, Pool
from hyperopt import hp
from hyperopt.pyll.base import scope
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from hyperparameters_tuner import HyperparametersTuner
from samplers import RandomOverSampler, RandomSampler

class CATBOOST_ENSEMBLE:
    NAME = 'CATBOOST_ENSEMBLE'

    def __init__(self, datainfo, timeinfo):
        self._info = extract(datainfo, timeinfo)
        print_data_info(self._info)
        print_time_info(self._info)

        self._classifier_class = CatBoostClassifier
        self._classifiers = []
        self._max_classifiers = 5
        self._validation_size = 0.3
        self._random_state = 13
        self._max_data = 400000
        self._max_evaluations = 5
        self._dataset_budget_threshold = 0.8
        self._category_indices = None
        self._categorical_frequency_map = {}
        self._mvc_frequency_map = {}
        self._iteration = 0
        self._over_sampler = RandomOverSampler(self._random_state)
        self._sampler = RandomSampler(self._max_data)
        self._fixed_hyperparameters = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC:hints=skip_train~false',
            'use_best_model': True,
            'best_model_min_trees': 100,
            'early_stopping_rounds': 7,
            'depth': 8,
            'random_strength': 1,
            'bagging_temperature': 1,
            'boosting_type': 'Plain',
            'max_ctr_complexity': 2,
            'verbose': True,
            'random_state': self._random_state,
            'has_time': True
        }
        self._search_space = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC:hints=skip_train~false',
            'use_best_model': True,
            'best_model_min_trees': scope.int(hp.quniform('best_model_min_trees', 100, 400, 50)),
            'early_stopping_rounds': 7,
            'depth': scope.int(hp.quniform('depth', 6, 10, 1)),
            'random_strength': hp.loguniform('random_strength', np.log(1), np.log(2)),
            'bagging_temperature': hp.loguniform('bagging_temperature', np.log(0.1), np.log(3)),
            'boosting_type': 'Plain',
            'max_ctr_complexity': 2,
            'verbose': False,
            'random_state': self._random_state,
            'has_time': True
        }
        self._best_hyperparameters = None
        self._lr = None

    def fit(self, F, y, datainfo, timeinfo):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('architectures.py', 'CATBOOST_ENSEMBLE', 'fit', 'Start'))
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

        validation_size = 0.1 if len(data) > self._max_data else self._validation_size
        self._sampler.max_data = 300000 if len(data) > self._max_data else self._max_data 
        
        transformed_data = np.array([])
        cat_start_index = None
        time_data, numerical_data, categorical_data, mvc_data = split_data_by_type(data, self._info)
        if len(time_data) > 0:
            transformed_data = subtract_min_time(time_data)
            transformed_data = np.concatenate((transformed_data, difference_between_time_columns(time_data)), axis=1)
        if len(numerical_data) > 0:
            self._info['transformed_numerical_data_starting_index'] = len(transformed_data)
            transformed_data = numerical_data if len(transformed_data) == 0 else \
                                np.concatenate((transformed_data, numerical_data), axis=1)
        if len(categorical_data) > 0:
            self._info['transformed_categorical_data_starting_index'] = len(transformed_data)
            if self._iteration == 0:
                count_frequency(self._categorical_frequency_map, categorical_data)
            cat_start_index = transformed_data.shape[1]
            transformed_data = np.concatenate((transformed_data, categorical_data), axis=1)
        if len(mvc_data) > 0: 
            self._info['transformed_mvc_data_starting_index'] = len(transformed_data)
            if self._iteration == 0:
                count_frequency(self._mvc_frequency_map, mvc_data)
            cat_start_index = transformed_data.shape[1] if cat_start_index is None else cat_start_index
            transformed_data = np.concatenate((transformed_data, mvc_data), axis=1)
       
        print('transformed_data.shape: {}'.format(transformed_data.shape))
        self._category_indices = None if cat_start_index is None else list(range(cat_start_index, transformed_data.shape[1]))
        
        self._train_data, self._validation_data, self._train_labels, self._validation_labels = train_test_split(
            transformed_data,
            y,
            test_size=validation_size,
            random_state=self._random_state,
            stratify=y
        )
        print('self._train_data.shape: {}'.format(self._train_data.shape))
        print('self._train_labels.shape: {}'.format(self._train_labels.shape))
        print('self._validation_data.shape: {}'.format(self._validation_data.shape))
        print('self._validation_labels.shape: {}'.format(self._validation_labels.shape))
        
        self._train_data, self._train_labels = self._over_sampler.sample(self._train_data, self._train_labels)
        print('train_data.shape: {}'.format(self._train_data.shape))
        print('train_labels.shape: {}'.format(self._train_labels.shape))

        self._train_data, self._train_labels = self._sampler.sample(self._train_data, self._train_labels)
        print('self._train_data.shape: {}'.format(self._train_data.shape))
        print('self._train_labels.shape: {}'.format(self._train_labels.shape))
        
        self._iteration += 1
        print('File: {} Class: {} Function: {} State: {} \n'.format('architectures.py', 'CATBOOST_ENSEMBLE', 'fit', 'End'))
    
    def predict(self, F, datainfo, timeinfo):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('architectures.py', 'CATBOOST_ENSEMBLE', 'predict', 'Start'))
        info = extract(datainfo, timeinfo)
        self._info.update(info)
        print_time_info(self._info)

        data = get_data(F, self._info)
        print('data.shape: {}'.format(data.shape))
        
        # transform and encode prediction data
        transformed_data = np.array([])
        encoded_categorical_data = np.array([])
        encoded_mvc_data = np.array([])
        cat_start_index = None
        time_data, numerical_data, categorical_data, mvc_data = split_data_by_type(data, self._info)
        if len(time_data) > 0:
            transformed_data = subtract_min_time(time_data)
            transformed_data = np.concatenate((transformed_data, difference_between_time_columns(time_data)), axis=1)
        if len(numerical_data) > 0:
            transformed_data = numerical_data if len(transformed_data) == 0 else \
                                np.concatenate((transformed_data, numerical_data), axis=1)
        if len(categorical_data) > 0:
            count_frequency(self._categorical_frequency_map, categorical_data)
            transformed_data = np.concatenate((transformed_data, categorical_data), axis=1)
            encoded_categorical_data = encode_frequency(self._categorical_frequency_map, categorical_data)
        if len(mvc_data) > 0: 
            count_frequency(self._mvc_frequency_map, mvc_data)
            transformed_data = np.concatenate((transformed_data, mvc_data), axis=1)
            encoded_mvc_data = encode_frequency(self._mvc_frequency_map, mvc_data)
        transformed_data = np.concatenate((transformed_data, encoded_categorical_data), axis=1) if len(encoded_categorical_data) > 0 else transformed_data
        transformed_data = np.concatenate((transformed_data, encoded_mvc_data), axis=1) if len(encoded_mvc_data) > 0 else transformed_data
        print('transformed_data.shape: {}'.format(transformed_data.shape))

        # encode training data
        transformed_time_data, transformed_numerical_data, \
        transformed_categorical_data, transformed_mvc_data = split_data_by_type(self._train_data, self._info, transformed=True)
        encoded_transformed_categorical_data = np.array([])
        encoded_transformed_mvc_data = np.array([])
        if len(transformed_categorical_data) > 0:
            encoded_transformed_categorical_data = encode_frequency(self._categorical_frequency_map, transformed_categorical_data)
        if len(transformed_mvc_data) > 0:
            encoded_transformed_mvc_data = encode_frequency(self._mvc_frequency_map, transformed_mvc_data)
        self._train_data = np.concatenate((self._train_data, encoded_transformed_categorical_data), axis=1) if len(encoded_transformed_categorical_data) > 0 : self._train_data
        self._train_data = np.concatenate((self._train_data, encoded_transformed_mvc_data), axis=1) if len(encoded_transformed_mvc_data) > 0 : self._train_data

        # encode validation data
        transformed_time_data, transformed_numerical_data, \
        transformed_categorical_data, transformed_mvc_data = split_data_by_type(self._validation_data, self._info, transformed=True)
        encoded_transformed_categorical_data = np.array([])
        encoded_transformed_mvc_data = np.array([])
        if len(transformed_categorical_data) > 0:
            encoded_transformed_categorical_data = encode_frequency(self._categorical_frequency_map, transformed_categorical_data)
        if len(transformed_mvc_data) > 0:
            encoded_transformed_mvc_data = encode_frequency(self._mvc_frequency_map, transformed_mvc_data)
        self._validation_data = np.concatenate((self._validation_data, encoded_transformed_categorical_data), axis=1) if len(encoded_transformed_categorical_data) > 0 : self._validation_data
        self._validation_data = np.concatenate((self._validation_data, encoded_transformed_mvc_data), axis=1) if len(encoded_transformed_mvc_data) > 0 : self._validation_data

        print('self._train_data.shape: {}'.format(self._train_data.shape))
        print('self._train_labels.shape: {}'.format(self._train_labels.shape))
        print('self._validation_data.shape: {}'.format(self._validation_data.shape))
        print('self._validation_labels.shape: {}'.format(self._validation_labels.shape))
        train_pool = Pool(self._train_data, self._train_labels, cat_features=self._category_indices)
        validation_pool = Pool(self._validation_data, self._validation_labels, cat_features=self._category_indices)
        
        if self._best_hyperparameters is None:
            tuner = HyperparametersTuner(self._classifier_class, self._fixed_hyperparameters, self._search_space, self._max_evaluations)
            self._best_hyperparameters = tuner.get_best_hyperparameters(train_pool, validation_pool)
            print('self._best_hyperparameters: {}'.format(self._best_hyperparameters))

        if has_sufficient_time(self._dataset_budget_threshold, self._info) or len(self._classifiers) == 0:
            classifier = self._classifier_class(**self._best_hyperparameters)
            classifier.fit(train_pool, eval_set=validation_pool)   
            self._classifiers.append(classifier)

            if len(self._classifiers) > self._max_classifiers:
                self._classifiers.pop(0)

            if len(self._classifiers) > 1:
                probabilities = np.zeros(len(validation_data))
                for i in range(len(self._classifiers)):
                    if i == 0:
                        probabilities = self._classifiers[i].predict_proba(validation_pool)[:,1]
                    else:
                        probabilities = np.vstack((probabilities, self._classifiers[i].predict_proba(validation_pool)[:,1]))

                probabilities = np.transpose(probabilities)
                self._lr = LogisticRegression()
                self._lr.fit(probabilities, validation_labels)
        else:
            print('Time budget exceeded.')

        test_pool = Pool(transformed_data, cat_features=self._category_indices)
        if len(self._classifiers) == 1:
            probabilities = self._classifiers[0].predict_proba(transformed_data)[:,1]
        else:
            probabilities = np.zeros(len(transformed_data))
            for i in range(len(self._classifiers)):
                if i == 0:
                    probabilities = self._classifiers[i].predict_proba(transformed_data)[:,1]
                else:
                    probabilities = np.vstack((probabilities, self._classifiers[i].predict_proba(transformed_data)[:,1]))

            probabilities = np.transpose(probabilities)
            probabilities = self._lr.predict_proba(probabilities)[:,1]
        print('probabilities.shape: {}'.format(probabilities.shape))
        print('File: {} Class: {} Function: {} State: {} \n'.format('architectures.py', 'CATBOOST_ENSEMBLE', 'predict', 'End'))
        return probabilities

    