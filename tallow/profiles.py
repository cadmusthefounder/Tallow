import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope

class Profile:

    LGBM_ORIGINAL_NAME = 'LGBM_ORIGINAL'

    LGBM_ORIGINAL = {
        'fixed_hyperparameters': {
            'learning_rate': 0.01, 
            'num_leaves': 60, 
            'feature_fraction': 0.6, 
            'bagging_fraction': 0.6, 
            'bagging_freq': 2, 
            'min_data_in_leaf': 20,
            'num_iterations': 600, 
            'early_stopping_round': 50,
            'boosting_type': 'gbdt', 
            'objective': 'binary', 
            'metric': 'auc'
        },
        'search_space': {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.01)),
            'num_leaves': scope.int(hp.quniform('num_leaves', 10, 80, 5)), 
            'feature_fraction': hp.loguniform('feature_fraction', np.log(0.6), np.log(0.9)), 
            'bagging_fraction': hp.loguniform('bagging_fraction', np.log(0.6), np.log(0.9)), 
            'bagging_freq': scope.int(hp.quniform('bagging_freq', 2, 10, 1)), 
            'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 10, 120, 10)),
            'num_iterations': 600, 
            'early_stopping_round': 30,
            'boosting_type': 'gbdt', 
            'objective': 'binary',
            'metric': 'auc'
        }
    }

    NAME_PROFILE_MAP = {
        LGBM_ORIGINAL_NAME: LGBM_ORIGINAL
    }

    @staticmethod
    def parse_profile(profile_name):
        profile = Profile.NAME_PROFILE_MAP[profile_name]
        return profile['fixed_hyperparameters'], profile['search_space']