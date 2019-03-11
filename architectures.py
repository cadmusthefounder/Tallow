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
        info = extract(datainfo, timeinfo)
        print_data_info(info)
        print_time_info(info)

        self._classifier_class = CatBoostClassifier
        self._classifiers = []
        self._validation_size = 0.3
        self._random_state = 42
        self._max_data = 400000
        self._max_evaluations = 15
        self._dataset_budget_threshold = 0.8
        self._over_sampler = RandomOverSampler(self._random_state)
        self._sampler = RandomSampler(self._max_data)
        self._fixed_hyperparameters = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC:hints=skip_train~false',
            'use_best_model': True,
            'od_pval': pow(10, -2),
            'n_estimators': 700,
            'depth': 8,
            'random_strength': 1,
            'bagging_temperature': 1,
            'has_time': True,
            'boosting_type': 'Plain',
            'max_ctr_complexity': 2,
            'verbose': True
        }
        self._search_space = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC:hints=skip_train~false',
            'use_best_model': True,
            'od_pval': pow(10, -2),
            'n_estimators': scope.int(hp.quniform('n_estimators', 400, 1000, 100)),
            'depth': scope.int(hp.quniform('depth', 6, 10, 1)),
            'random_strength': scope.int(hp.quniform('random_strength', 1, 5, 1)),
            'bagging_temperature': hp.loguniform('bagging_temperature', np.log(0.1), np.log(3)),
            'has_time': True,
            'boosting_type': 'Plain',
            'max_ctr_complexity': 2,
            'verbose': False
        }
        self._best_hyperparameters = None
        self._lr = None

    def fit(self, F, y, datainfo, timeinfo):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('architectures.py', 'CATBOOST_ENSEMBLE', 'fit', 'Start'))
        info = extract(datainfo, timeinfo)
        print_time_info(info)

        data = get_data(F, info)
        y = y.ravel()
        print('data.shape: {}'.format(data.shape))
        print('y.shape: {}'.format(y.shape))

        bincount = np.bincount(y.astype(int))
        print('Number of 0 label: {}'.format(bincount[0]))
        print('Number of 1 label: {}'.format(bincount[1]))

        transformed_data = np.array([])
        cat_start_index = None
        time_data, numerical_data, categorical_data, mvc_data = split_data_by_type(data, info)
        if len(time_data) > 0:
            transformed_data = subtract_min_time(time_data)
            transformed_data = np.concatenate((transformed_data, difference_between_time_columns(time_data)), axis=1)
        if len(numerical_data) > 0:
            transformed_data = numerical_data if len(transformed_data) == 0 else \
                                np.concatenate((transformed_data, numerical_data), axis=1)
        if len(categorical_data) > 0:
            cat_start_index = transformed_data.shape[1]
            transformed_data = np.concatenate((transformed_data, categorical_data), axis=1)
        if len(mvc_data) > 0: 
            cat_start_index = transformed_data.shape[1] if cat_start_index is None else cat_start_index
            transformed_data = np.concatenate((transformed_data, mvc_data), axis=1)
        print('transformed_data.shape: {}'.format(transformed_data.shape))

        train_data, validation_data, train_labels, validation_labels = train_test_split(
            transformed_data,
            y,
            test_size=self._validation_size,
            random_state=self._random_state,
            shuffle=False
        )
        print('train_data.shape: {}'.format(train_data.shape))
        print('train_labels.shape: {}'.format(train_labels.shape))
        print('validation_data.shape: {}'.format(validation_data.shape))
        print('validation_labels.shape: {}'.format(validation_labels.shape))
        
        train_data, train_labels = self._over_sampler.sample(train_data, train_labels)
        print('train_data.shape: {}'.format(train_data.shape))
        print('train_labels.shape: {}'.format(train_labels.shape))

        train_data, train_labels = self._sampler.sample(train_data, train_labels)
        print('train_data.shape: {}'.format(train_data.shape))
        print('train_labels.shape: {}'.format(train_labels.shape))
        
        category_indices = None if cat_start_index is None else list(range(cat_start_index, transformed_data.shape[1]))
        train_pool = Pool(train_data, train_labels, cat_features=category_indices)
        validation_pool = Pool(validation_data, validation_labels, cat_features=category_indices)

        if self._best_hyperparameters is None:
            tuner = HyperparametersTuner(self._classifier_class, self._fixed_hyperparameters, self._search_space, self._max_evaluations)
            self._best_hyperparameters = tuner.get_best_hyperparameters(train_pool, validation_pool)
            print('self._best_hyperparameters: {}'.format(self._best_hyperparameters))

        if has_sufficient_time(self._dataset_budget_threshold, info) or len(self._classifiers) == 0:
            classifier = self._classifier_class(**self._best_hyperparameters)
            classifier.fit(train_pool, eval_set=validation_pool)   
            self._classifiers.append(classifier)

            probabilities = np.zeros(len(validation_data))
            for i in range(len(self._classifiers)):
                if i == 0:
                    probabilities = self._classifiers[i].predict_proba(validation_pool)[:,1]
                else:
                    probabilities = np.vstack((probabilities, self._classifiers[i].predict_proba(validation_pool)[:,1]))

            probabilities = np.transpose(probabilities)
            self._lr = LogisticRegression()
            self._lr.fit(probabilities, validation_labels)

        print('File: {} Class: {} Function: {} State: {} \n'.format('architectures.py', 'CATBOOST_ENSEMBLE', 'fit', 'End'))
    
    def predict(self, F, datainfo, timeinfo):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('architectures.py', 'CATBOOST_ENSEMBLE', 'predict', 'Start'))
        info = extract(datainfo, timeinfo)
        print_time_info(info)

        data = get_data(F, info)
        print('data.shape: {}'.format(data.shape))
        
        transformed_data = np.array([])
        cat_start_index = None
        time_data, numerical_data, categorical_data, mvc_data = split_data_by_type(data, info)
        if len(time_data) > 0:
            transformed_data = subtract_min_time(time_data)
            transformed_data = np.concatenate((transformed_data, difference_between_time_columns(time_data)), axis=1)
        if len(numerical_data) > 0:
            transformed_data = numerical_data if len(transformed_data) == 0 else \
                                np.concatenate((transformed_data, numerical_data), axis=1)
        if len(categorical_data) > 0:
            cat_start_index = transformed_data.shape[1]
            transformed_data = np.concatenate((transformed_data, categorical_data), axis=1)
        if len(mvc_data) > 0: 
            cat_start_index = transformed_data.shape[1] if cat_start_index is None else cat_start_index
            transformed_data = np.concatenate((transformed_data, mvc_data), axis=1)
        print('transformed_data.shape: {}'.format(transformed_data.shape))

        category_indices = None if cat_start_index is None else list(range(cat_start_index, transformed_data.shape[1]))
        test_pool = Pool(transformed_data, cat_features=category_indices)
        
        probabilities = np.zeros(len(transformed_data))
        for i in range(len(self._classifiers)):
            if i == 0:
                probabilities = self._classifiers[i].predict_proba(transformed_data)[:,1]
            else:
                probabilities = np.vstack((probabilities, self._classifiers[i].predict_proba(transformed_data)[:,1]))

        probabilities = self._lr.predict_proba(probabilities)
        print('probabilities.shape: {}'.format(probabilities.shape))
        print('File: {} Class: {} Function: {} State: {} \n'.format('architectures.py', 'CATBOOST_ENSEMBLE', 'predict', 'End'))
        return probabilities

    