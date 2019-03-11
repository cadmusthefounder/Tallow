from utils import *
pip_install('catboost')
pip_install('hyperopt')

import numpy as np
from math import pow
from catboost import CatBoostClassifier, Pool
from hyperopt import hp
from hyperopt.pyll.base import scope
from sklearn.model_selection import train_test_split

class CATBOOST_ENSEMBLE:
    NAME = 'CATBOOST_ENSEMBLE'

    def __init__(self, datainfo, timeinfo):
        info = extract(datainfo, timeinfo)
        print_data_info(info)
        print_time_info(info)

        self._classifier_class = CatBoostClassifier
        self._classifier = None
        self._validation_size = 0.3
        self._fixed_hyperparameters = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC:hints=skip_train~false',
            'use_best_model': True,
            'od_type': 'IncToDec',
            'od_pval': pow(10, -5),
            'n_estimators': 500,
            'depth': 8,
            'random_strength': 1,
            'bagging_temperature': 1,
            'has_time': True,
            'boosting_type': 'Plain',
            'bootstrap_type' 'Bernoulli',
            'max_ctr_complexity': 2
        }
        self._search_space = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC:hints=skip_train~false',
            'use_best_model': True,
            'od_type': 'IncToDec',
            'od_pval': hp.loguniform('od_pval', np.log(pow(10, -10)), np.log(pow(10, -2))),
            'n_estimators': scope.int(hp.quniform('n_estimators', 400, 700, 50)),
            'depth': scope.int(hp.quniform('depth', 6, 10, 1)),
            'random_strength': scope.int(hp.quniform('random_strength', 1, 5, 1)),
            'bagging_temperature': hp.loguniform('bagging_temperature', np.log(0), np.log(3)),
            'has_time': True,
            'boosting_type': 'Plain',
            'bootstrap_type' 'Bernoulli',
            'max_ctr_complexity': 2
        }

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
            random_state=42,
            shuffle=False
        )
        category_indices = None if cat_start_index is None else list(range(cat_start_index, transformed_data.shape[1]))
        train_pool = Pool(train_data, train_labels, cat_features=category_indices)
        validation_pool = Pool(validation_data, validation_labels, cat_features=category_indices)

        self._classifier = self._classifier_class(**self._fixed_hyperparameters)
        self._classifier.fit(train_pool, eval_set=validation_pool)
        
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
        probabilities = self._classifier.predict_proba(test_pool)[:,1]
        print('probabilities.shape: {}'.format(probabilities.shape))
        print('File: {} Class: {} Function: {} State: {} \n'.format('architectures.py', 'CATBOOST_ENSEMBLE', 'predict', 'End'))
        return probabilities

    