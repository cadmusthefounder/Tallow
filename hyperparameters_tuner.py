import numpy as np
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, space_eval, STATUS_OK, Trials

class HyperparametersTuner:

    def __init__(self, classifier_class, fixed_hyperparameters, search_space, max_evaluations):
        self._classifier_class = classifier_class
        self._fixed_hyperparameters = fixed_hyperparameters
        self._search_space = search_space
        self._max_evaluations = max_evaluations

    def get_best_hyperparameters(self, train_pool, validation_pool):
        print('\nFile: {} Class: {} Function: {} State: {}'.format('hyperparameters_tuner.py', 'HyperparametersTuner', 'get_best_hyperparameters', 'Start'))
        self._train_pool = train_pool
        self._validation_pool = validation_pool

        # Try fixed hyperparameters
        classifier = self._classifier_class()
        classifier.set_params(**self._fixed_hyperparameters)
        classifier.fit(self._train_pool, eval_set=self._validation_pool)
        predictions = classifier.predict(self._validation_pool)

        print('Fixed hyperparameters')
        print('self._validation_pool.get_label().shape: {}'.format(self._validation_pool.get_label().shape))
        print('predictions.shape: {}'.format(predictions.shape))
        
        fixed_hyperparameters_score = roc_auc_score(self._validation_pool.get_label(), predictions)

        # Find best trial hyperparameters
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

        classifier = self._classifier_class()
        classifier.set_params(**trial_hyperparameters)
        classifier.fit(self._train_pool, eval_set=self._validation_pool)
        predictions = classifier.predict(self._validation_pool)

        print('self._validation_pool.get_label().shape: {}'.format(self._validation_pool.get_label().shape))
        print('predictions.shape: {}'.format(predictions.shape))
        
        trial_score = roc_auc_score(self._validation_pool.get_label(), predictions)
        print('File: {} Class: {} Function: {} State: {} \n'.format('hyperparameters_tuner.py', 'HyperparametersTuner', 'objective', 'End'))
        return {'loss': (1 - trial_score), 'status': STATUS_OK }