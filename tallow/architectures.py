from math import pow
from copy import deepcopy
from utils import *

# pip_install('sklearn')
pip_install('lightgbm')
pip_install('hyperopt')
# pip_install('imbalanced-learn')

import numpy as np
from math import pow
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from hyperparameters_tuner import HyperparametersTuner
from profiles import Profile
from samplers import StratifiedRandomSampler, OldRandomMajorityUnderSampler

class DataType:
    TRAIN = 'TRAIN'
    TEST = 'TEST'

class OriginalEnsemble:
    NAME = 'OriginalEnsemble'

    def __init__(self, datainfo, timeinfo):
        self._info = extract(datainfo, timeinfo)
        print_data_info(self._info)
        print_time_info(self._info)

        self._validation_ratio = 0.25
        self._max_data = 400000

        self._iteration = 0
        self._random_state = 13
        self._max_evaluations = 25
        self._dataset_budget_threshold = 0.8
        self._should_correct = False
        self._correction_threshold = 0.75
        self._correction_n_splits = 20
        self._epsilon = 0.001
        self._ensemble_size = 3

        self._categorical_frequency_map = {}
        self._mvc_frequency_map = {}
        self._train_data = []
        self._train_labels = []
        
        self._best_hyperparameters = None
        self._classifiers = np.array([])
        self._imbalanced_sampler = OldRandomMajorityUnderSampler(self._random_state)
        self._too_much_data_sampler = StratifiedRandomSampler(self._max_data, self._random_state)
        self._profile = Profile.LGBM_ORIGINAL_NAME

    def fit(self, F, y, datainfo, timeinfo):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('architectures.py', 'OriginalEnsemble', 'fit', 'Start'))

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

        self._train_data = np.concatenate((self._train_data, data), axis=0) if len(self._train_data) > 0 else data
        self._train_labels = np.concatenate((self._train_labels, y), axis=0) if len(self._train_labels) > 0 else y
        self._train_data, self._train_labels = self._too_much_data_sampler.sample(data, y)
        print('self._train_data.shape: {}'.format(self._train_data.shape))
        print('self._train_labels.shape: {}'.format(self._train_labels.shape))
        print('File: {} Class: {} Function: {} State: {} \n'.format('architectures.py', 'OriginalEnsemble', 'fit', 'End'))
        
    def predict(self, F, datainfo, timeinfo):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('architectures.py', 'OriginalEnsemble', 'predict', 'Start'))
    
        info = extract(datainfo, timeinfo)
        self._info.update(info)
        print_time_info(self._info)

        test_data = get_data(F, self._info)
        print('test_data.shape: {}'.format(test_data.shape))

        transformed_test_data = self._transform(test_data, DataType.TEST)
        transformed_train_data = self._transform(self._train_data, DataType.TRAIN)
        print('transformed_test_data.shape: {}'.format(transformed_test_data.shape))
        print('transformed_train_data.shape: {}'.format(transformed_train_data.shape))
       
        train_data, validation_data, train_labels, validation_labels = train_test_split(
            transformed_train_data,
            self._train_labels,
            test_size=self._validation_ratio,
            random_state=self._random_state,
            shuffle=True,
            stratify=self._train_labels
        )
        print('train_data.shape: {}'.format(train_data.shape))
        print('train_labels.shape: {}'.format(train_labels.shape))
        print('validation_data.shape: {}'.format(validation_data.shape))
        print('validation_labels.shape: {}'.format(validation_labels.shape))

        train_data, train_labels = self._imbalanced_sampler.sample(train_data, train_labels)
        print('train_data.shape: {}'.format(train_data.shape))
        print('train_labels.shape: {}'.format(train_labels.shape))

        train_weights = correct_covariate_shift(
            train_data, 
            transformed_test_data, 
            self._random_state, 
            self._correction_threshold, 
            self._correction_n_splits
        ) if self._should_correct else None
        validation_weights =  correct_covariate_shift(
            validation_data, 
            transformed_test_data, 
            self._random_state, 
            self._correction_threshold, 
            self._correction_n_splits
        ) if self._should_correct else None
        train_dataset = lgbm.Dataset(train_data, train_labels, weight=train_weights, free_raw_data=False)
        validation_dataset = train_dataset.create_valid(validation_data, validation_labels, weight=validation_weights)

        fixed_hyperparameters, search_space = Profile.parse_profile(self._profile)
        if self._best_hyperparameters is None:
            tuner = HyperparametersTuner(fixed_hyperparameters, search_space, self._max_evaluations)
            self._best_hyperparameters = tuner.get_best_hyperparameters(train_dataset, validation_dataset)
            self._best_hyperparameters_clone = deepcopy(self._best_hyperparameters)
            self._best_hyperparameters_clone.pop('num_iterations', None)
            self._best_hyperparameters_clone.pop('early_stopping_round', None)
            print('self._best_hyperparameters: {}'.format(self._best_hyperparameters))    

        if has_sufficient_time(self._dataset_budget_threshold, self._info) or len(self._classifiers) == 0:
            new_classifier = lgbm.train(
                params=self._best_hyperparameters, 
                train_set=train_dataset, 
                valid_sets=[validation_dataset], 
                keep_training_booster=True,
                init_model=None
            )
            new_predictions = new_classifier.predict(validation_data)
            new_weight =  compute_weight(
                new_predictions, 
                validation_labels,
                validation_weights,
                self._epsilon
            )

            validation_train_data, validation_train_labels = self._imbalanced_sampler.sample(validation_data, validation_labels, free_raw_data=False)
            validation_train_dataset = lgbm.Dataset(validation_train_data, validation_train_labels)
            new_classifier = lgbm.train(
                params=self._best_hyperparameters_clone, 
                train_set=validation_train_dataset, 
                keep_training_booster=True,
                init_model=new_classifier
            )

            dummy_classifier = DummyClassifier(random_state=self._random_state)
            dummy_classifier.fit(train_data, train_labels, sample_weight=train_weights)
            dummy_predictions = dummy_classifier.predict(validation_data)
            dummy_weight =  compute_weight(
                dummy_predictions, 
                validation_labels,
                validation_weights,
                self._epsilon
            )

            self._ensemble_weights = np.array([])
            for i in range(len(self._classifiers)):
                currrent_classifier = self._classifiers[i]
                currrent_classifier_predictions = currrent_classifier.predict(validation_data)
                currrent_classifier_weight =  compute_weight(
                    currrent_classifier_predictions, 
                    validation_labels,
                    validation_weights,
                    self._epsilon
                )
                self._ensemble_weights = np.append(self._ensemble_weights, currrent_classifier_weight)

                if currrent_classifier_weight > dummy_weight:
                    currrent_classifier = lgbm.train(
                        params=self._best_hyperparameters, 
                        train_set=validation_train_dataset, 
                        valid_sets=[validation_dataset], 
                        keep_training_booster=True,
                        init_model=currrent_classifier
                    )
                    currrent_classifier = lgbm.train(
                        params=self._best_hyperparameters_clone, 
                        train_set=validation_train_dataset, 
                        keep_training_booster=True,
                        init_model=currrent_classifier
                    )
            self._classifiers = np.append(self._classifiers, new_classifier)
            self._ensemble_weights = np.append(self._ensemble_weights, new_weight)
            print('self._ensemble_weights: {}'.format(self._ensemble_weights))

            if len(self._classifiers) > self._ensemble_size:
                i = remove_worst_classifier(self._classifiers, validation_data, validation_labels)
                print('Removed classifier: {}'.format(i))
                self._classifiers = np.delete(self._classifiers, i)
                self._ensemble_weights = np.delete(self._ensemble_weights, i)
        else:
            print('Time budget exceeded.')

        self._iteration += 1
        predictions = np.zeros(len(transformed_test_data))
        for i in range(len(self._classifiers)):
            predictions = np.add(predictions, self._ensemble_weights[i] * self._classifiers[i].predict(transformed_test_data))
        predictions = np.divide(predictions, np.sum(self._ensemble_weights))        
        print('predictions.shape: {}'.format(predictions.shape))
        print('File: {} Class: {} Function: {} State: {} \n'.format('architectures.py', 'OriginalEnsemble', 'predict', 'End'))
        return predictions

    def _transform(self, data, datatype):
        transformed_data = np.array([])
        time_data, numerical_data, categorical_data, mvc_data = split_data_by_type(data, self._info)
        if len(time_data) > 0:
            transformed_data = subtract_min_time(time_data)
            transformed_data = np.concatenate((transformed_data, difference_between_time_columns(time_data)), axis=1)
            transformed_data = np.concatenate((transformed_data, extract_detailed_time(time_data)), axis=1)
        if len(numerical_data) > 0:
            transformed_data = numerical_data if len(transformed_data) == 0 else \
                                np.concatenate((transformed_data, numerical_data), axis=1)
        if len(categorical_data) > 0:
            if (datatype == DataType.TRAIN and self._iteration == 0) or datatype == DataType.TEST:
                self._categorical_frequency_map = count_frequency(self._categorical_frequency_map, categorical_data)
            encoded_categorical_data = encode_frequency(self._categorical_frequency_map, categorical_data)
            transformed_data = np.concatenate((transformed_data, encoded_categorical_data), axis=1)
        if len(mvc_data) > 0: 
            if (datatype == DataType.TRAIN and self._iteration == 0) or datatype == DataType.TEST:
                self._mvc_frequency_map = count_frequency(self._mvc_frequency_map, mvc_data)
            
            encoded_mvc_data = encode_frequency(self._mvc_frequency_map, mvc_data)
            transformed_data = np.concatenate((transformed_data, encoded_mvc_data), axis=1)
        return transformed_data