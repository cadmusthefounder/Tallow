from math import pow
from utils import *
pip_install('catboost')
pip_install('lightgbm')
pip_install('hyperopt')

import numpy as np
from math import pow
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from hyperparameters_tuner import HyperparametersTuner
from profiles import Profile
from samplers import RandomOverSampler, RandomUnderSampler, RandomSampler, BiasedReservoirSampler

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
        self._early_stopping_rounds = 0 if not self._use_validation else 0

        self._dataset_size_threshold = 400000
        self._large_dataset_max_data = 300000
        self._small_dataset_max_data = 400000

        self._iteration = 0
        self._random_state = 13
        self._max_evaluations = 25
        self._dataset_budget_threshold = 0.8

        self._category_indices = None
        self._categorical_frequency_map = {}
        self._mvc_frequency_map = {}
        
        self._best_hyperparameters = None
        
        self._train_data = np.array([])
        self._train_labels = np.array([])

        self._classifier = None
        self._lr = None
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
        validation_pool = Pool(validation_data, self._validation_labels) if self._use_validation else None
        validation_set = (validation_data, self._validation_labels) if self._use_validation else None
        
        if self._best_hyperparameters is None:
            tuner = HyperparametersTuner(classification_class, fixed_hyperparameters, search_space, self._max_evaluations)
            self._best_hyperparameters = tuner.get_best_hyperparameters(train_pool, validation_pool)
            print('self._best_hyperparameters: {}'.format(self._best_hyperparameters))

        if has_sufficient_time(self._dataset_budget_threshold, self._info) or self._classifier is None:
            self._classifier = classification_class(**self._best_hyperparameters)
            
            if isinstance(self._classifier, LGBMClassifier):                
                self._classifier.fit(train_data, self._train_labels)
            else:
                self._classifier.fit(train_pool)

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
            transformed_data = np.concatenate((transformed_data, extract_detailed_time(time_data)), axis=1)
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

class OriginalEnsemble:
    NAME = 'OriginalEnsemble'

    def __init__(self, datainfo, timeinfo):
        self._info = extract(datainfo, timeinfo)
        print_data_info(self._info)
        print_time_info(self._info)

        self._use_validation = True
        self._large_dataset_validation_ratio = 0 if not self._use_validation else 0.1
        self._small_dataset_validation_ratio = 0 if not self._use_validation else 0.25
        self._early_stopping_rounds = 0 if not self._use_validation else 0

        self._dataset_size_threshold = 400000
        self._large_dataset_max_data = 300000
        self._small_dataset_max_data = 400000

        self._iteration = 0
        self._random_state = 13
        self._max_evaluations = 25
        self._dataset_budget_threshold = 0.8

        self._category_indices = None
        self._categorical_frequency_map = {}
        self._mvc_frequency_map = {}
        
        self._best_hyperparameters = None
        
        self._train_data = np.array([])
        self._train_labels = np.array([])

        self._classifiers = []
        self._under_sampler = RandomUnderSampler(self._random_state)
        self._sampler = None
        self._bias_rate = pow(10, -6)
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

        max_data = self._large_dataset_max_data if is_large_dataset(len(data), self._dataset_size_threshold) else self._small_dataset_max_data
        validation_ratio = self._large_dataset_validation_ratio if is_large_dataset(len(data), self._dataset_size_threshold) else self._small_dataset_validation_ratio
        self._sampler = RandomSampler(max_data) if self._sampler is None else self._sampler
        # self._sampler = BiasedReservoirSampler(max_data, self._bias_rate, self._info) if self._sampler is None else self._sampler

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

        # self._train_data = train_data if len(self._train_data) == 0 else np.concatenate((self._train_data, train_data), axis=0)
        # self._train_labels = train_labels if len(self._train_labels) == 0 else np.concatenate((self._train_labels, train_labels), axis=0)
        # self._train_data, self._train_labels = self._sampler.sample(self._train_data, self._train_labels)
        self._train_data, self._train_labels = self._sampler.sample(train_data, train_labels)
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

        classification_class, fixed_hyperparameters, search_space = Profile.parse_profile(self._profile, self._early_stopping_rounds)

        test_data = self._transform(test_data, DataType.TEST)
        train_data = self._transform(self._train_data, DataType.TRAIN)
        validation_data = self._transform(self._validation_data, DataType.VALIDATION)

        print('test_data.shape: {}'.format(test_data.shape))
        print('train_data.shape: {}'.format(train_data.shape))
        print('self._train_labels.shape: {}'.format(self._train_labels.shape))
        print('validation_data.shape: {}'.format(validation_data.shape))
        print('self._validation_labels.shape: {}'.format(self._validation_labels.shape))

        weights = self._correct_covariate_shift(train_data, test_data)
        print('weights.shape: {}'.format(weights.shape))

        train_pool = Pool(train_data, self._train_labels)
        validation_pool = Pool(validation_data, self._validation_labels) if self._use_validation else None
        validation_set = (validation_data, self._validation_labels) if self._use_validation else None
        
        if self._best_hyperparameters is None:
            tuner = HyperparametersTuner(classification_class, fixed_hyperparameters, search_space, self._max_evaluations)
            self._best_hyperparameters = tuner.get_best_hyperparameters(train_pool, validation_pool, weights)
            print('self._best_hyperparameters: {}'.format(self._best_hyperparameters))

        if has_sufficient_time(self._dataset_budget_threshold, self._info) or len(self._classifiers) == 0:
            classifier = classification_class(**self._best_hyperparameters)
            
            if isinstance(classifier, LGBMClassifier):                
                classifier.fit(train_data, self._train_labels, sample_weight=weights)
            else:
                classifier.fit(train_pool)

            self._classifiers.append(classifier)

            if len(self._classifiers) > 1:
                for i in range(len(self._classifiers)):
                    if i == 0:
                        probabilities = self._classifiers[i].predict_proba(validation_data)[:,1]
                    else:
                        probabilities = np.vstack((probabilities, self._classifiers[i].predict_proba(validation_data)[:,1]))
                probabilities = np.transpose(probabilities)
                self._lr = LogisticRegression()
                self._lr.fit(probabilities, self._validation_labels)
        else:
            print('Time budget exceeded.')

        if len(self._classifiers) == 1:
            probabilities = self._classifiers[0].predict_proba(test_data)[:,1]
        else:
            for i in range(len(self._classifiers)):
                if i == 0:
                    probabilities = self._classifiers[i].predict_proba(test_data)[:,1]
                else:
                    probabilities = np.vstack((probabilities, self._classifiers[i].predict_proba(test_data)[:,1]))
            probabilities = np.transpose(probabilities)
            probabilities = self._lr.predict_proba(probabilities)[:,1]
        print('probabilities.shape: {}'.format(probabilities.shape))
        print('File: {} Class: {} Function: {} State: {} \n'.format('architectures.py', 'OriginalEnsemble', 'predict', 'End'))
        
        self._iteration += 1
        return probabilities

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
                count_frequency(self._categorical_frequency_map, categorical_data)
            encoded_categorical_data = encode_frequency(self._categorical_frequency_map, categorical_data)
            transformed_data = np.concatenate((transformed_data, encoded_categorical_data), axis=1)
        if len(mvc_data) > 0: 
            if (datatype == DataType.TRAIN and self._iteration == 0) or datatype == DataType.TEST:
                count_frequency(self._mvc_frequency_map, mvc_data)
            
            encoded_mvc_data = encode_frequency(self._mvc_frequency_map, mvc_data)
            transformed_data = np.concatenate((transformed_data, encoded_mvc_data), axis=1)
        return transformed_data

    def _correct_covariate_shift(self, train_data, test_data):
        X = pd.DataFrame(test_data)
        Z = pd.DataFrame(train_data)
        X['is_z'] = 0 # 0 means test set
        Z['is_z'] = 1 # 1 means training set
        XZ = pd.concat( [X, Z], ignore_index=True, axis=0 )

        labels = XZ['is_z'].values
        XZ = XZ.drop('is_z', axis=1).values
        X, Z = X.values, Z.values

        clf = RandomForestClassifier(max_depth=2)
        predictions = np.zeros(labels.shape)
        skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=self._random_state)
        for fold, (train_idx, test_idx) in enumerate(skf.split(XZ, labels)):
            print('Training discriminator model for fold {}'.format(fold))
            X_train, X_test = XZ[train_idx], XZ[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
                
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            predictions[test_idx] = probs

        print('ROC-AUC for X and Z distributions: {}'.format(roc_auc_score(labels, predictions)))
        predictions_Z = predictions[len(X):]
        weights = (1./predictions_Z) - 1. 
        weights /= np.mean(weights) # we do this to re-normalize the computed log-loss
        return weights
