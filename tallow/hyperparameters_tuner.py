import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, space_eval, STATUS_OK, Trials
from lightgbm import LGBMClassifier

class HyperparametersTuner:

    def __init__(self, fixed_hyperparameters, search_space, max_evaluations):
        self._fixed_hyperparameters = fixed_hyperparameters
        self._search_space = search_space
        self._max_evaluations = max_evaluations


    def get_best_hyperparameters(self, train_data, train_labels, validation_ratio, random_state):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('hyperparameters_tuner.py', 'HyperparametersTuner', 'get_best_hyperparameters', 'Start'))

        self._train_data, self._validation_data, self._train_labels, self._validation_labels = train_test_split(
            train_data,
            train_labels,
            test_size=validation_ratio,
            random_state=random_state,
            shuffle=True,
            stratify=train_labels
        )
        classifier = LGBMClassifier()
        classifier.set_params(**self._fixed_hyperparameters)
        classifier.fit(self._train_data, self._train_labels)

        predictions = classifier.predict_proba(self._validation_data)[:,1]
        labels = self._validation_labels
        fixed_hyperparameters_score = roc_auc_score(labels, predictions)
        print('labels.shape: {}'.format(labels.shape))
        print('predictions.shape: {}'.format(predictions.shape))
        
        trials = Trials()
        best = fmin(
            fn=self.objective,
            space=self._search_space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=self._max_evaluations
        )
        best_trial_hyperparameters = space_eval(self._search_space, best)
        best_trial_hyperparameters_score = 1 - np.min([x['loss'] for x in trials.results])

        if fixed_hyperparameters_score > best_trial_hyperparameters_score:
            print('best auc score: {}'.format(fixed_hyperparameters_score))
            return self._fixed_hyperparameters
        else:
            print('best auc score: {}'.format(best_trial_hyperparameters_score))
            return best_trial_hyperparameters

        print('File: {} Class: {} Function: {} State: {} \n'.format('hyperparameters_tuner.py', 'HyperparametersTuner', 'get_best_hyperparameters', 'End'))

    def objective(self, trial_hyperparameters):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('hyperparameters_tuner.py', 'HyperparametersTuner', 'objective', 'Start'))
        print('trial_hyperparameters: {}'.format(trial_hyperparameters))
        
        classifier = LGBMClassifier()
        classifier.set_params(**self._fixed_hyperparameters)
        classifier.fit(self._train_data, self._train_labels)

        predictions = classifier.predict_proba(self._validation_data)[:,1]
        labels = self._validation_labels
        trial_score = roc_auc_score(labels, predictions)
        print('labels.shape: {}'.format(labels.shape))
        print('predictions.shape: {}'.format(predictions.shape))

        print('File: {} Class: {} Function: {} State: {} \n'.format('hyperparameters_tuner.py', 'HyperparametersTuner', 'objective', 'End'))
        return {'loss': (1 - trial_score), 'status': STATUS_OK }