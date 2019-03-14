import pip
import time
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error

def pip_install(package):
    pip.main(['install', package])

def pip_uninstall(package):
    pip.main(['uninstall', '-y', package])

def extract(datainfo, timeinfo):
    time_budget = datainfo['time_budget']
    no_of_time_features = datainfo['loaded_feat_types'][0]
    no_of_numerical_features = datainfo['loaded_feat_types'][1]
    no_of_categorical_features = datainfo['loaded_feat_types'][2]
    no_of_mvc_features = datainfo['loaded_feat_types'][3]
    total_no_of_features = no_of_time_features + no_of_numerical_features + \
                        no_of_categorical_features + no_of_mvc_features

    numerical_data_starting_index = no_of_time_features
    categorical_data_starting_index = numerical_data_starting_index + no_of_numerical_features
    mvc_data_starting_index = categorical_data_starting_index + no_of_categorical_features

    current_time = time.time() 
    overall_time_spent = current_time - timeinfo[0]
    dataset_time_spent = current_time- timeinfo[1]

    return {
        'time_budget': time_budget,
        'no_of_time_features': no_of_time_features,
        'no_of_numerical_features': no_of_numerical_features,
        'no_of_categorical_features': no_of_categorical_features,
        'no_of_mvc_features': no_of_mvc_features,
        'total_no_of_features': total_no_of_features,
        'numerical_data_starting_index': numerical_data_starting_index,
        'categorical_data_starting_index': categorical_data_starting_index,
        'mvc_data_starting_index': mvc_data_starting_index,
        'overall_time_spent': overall_time_spent,
        'dataset_time_spent': dataset_time_spent
    }

def print_data_info(info):
    print('\nFile: {} Class: {} Function: {} State: {}'.format('utils.py', 'None', 'print_data_info', 'Start'))
    print('Dataset time budget: {0:f} seconds'.format(info['time_budget']))
    print('No. of time features: {0:d}'.format(info['no_of_time_features']))
    print('No. of numerical features: {0:d}'.format(info['no_of_numerical_features']))
    print('No. of categorical features: {0:d}'.format(info['no_of_categorical_features']))
    print('No. of mvc features: {0:d}'.format(info['no_of_mvc_features']))
    print('File: {} Class: {} Function: {} State: {} \n'.format('utils.py', 'None', 'print_data_info', 'End'))

def print_time_info(info):
    print('\nFile: {} Class: {} Function: {} State: {}'.format('utils.py', 'None', 'print_time_info', 'Start'))
    print('Overall time spent: {0:5.2f} seconds'.format(info['overall_time_spent']))
    print('Dataset time spent: {0:5.2f} seconds'.format(info['dataset_time_spent'])) 
    print('Time budget: {0:5.2f} seconds'.format(info['time_budget'])) 
    print('File: {} Class: {} Function: {} State: {} \n'.format('utils.py', 'None', 'print_time_info', 'End'))

def get_data(F, info):
    print('\nFile: {} Class: {} Function: {} State: {}'.format('utils.py', 'None', 'get_data', 'Start'))
    data = np.array([])
    if info['no_of_time_features'] > 0 or info['no_of_numerical_features'] > 0:
        data = np.nan_to_num(F['numerical'])

    if info['no_of_categorical_features'] > 0:
        data = F['CAT'].fillna('nan').values if len(data) == 0 else \
                np.concatenate((data, F['CAT'].fillna('nan').values), axis=1)

    if info['no_of_mvc_features'] > 0:
        data = F['MV'].fillna('nan').values if len(data) == 0 else \
                np.concatenate((data, F['MV'].fillna('nan').values), axis=1)

    print('data.shape: {}'.format(data.shape))
    print('File: {} Class: {} Function: {} State: {} \n'.format('utils.py', 'None', 'get_data', 'End'))
    return data

def split_data_by_type(data, info, transformed=False):
    print('\nFile: {} Class: {} Function: {} State: {}'.format('utils.py', 'None', 'split_data_by_type', 'Start'))
    
    if not transformed:
        time_data = np.array([]) if info['no_of_time_features'] == 0 else data[:,:info['numerical_data_starting_index']]
        numerical_data = np.array([]) if info['no_of_numerical_features'] == 0 else \
                        data[:,info['numerical_data_starting_index']:info['categorical_data_starting_index']]
        categorical_data = np.array([]) if info['no_of_categorical_features'] == 0 else \
                        data[:,info['categorical_data_starting_index']:info['mvc_data_starting_index']]
        mvc_data = np.array([]) if info['no_of_mvc_features'] == 0 else \
                        data[:,info['mvc_data_starting_index']:]

        print('time_data.shape :{}'.format(time_data.shape))
        print('numerical_data.shape :{}'.format(numerical_data.shape))
        print('categorical_data.shape :{}'.format(categorical_data.shape))
        print('mvc_data.shape :{}\n'.format(mvc_data.shape))
        print('File: {} Class: {} Function: {} State: {} \n'.format('utils.py', 'None', 'split_data_by_type', 'End'))
        return time_data, numerical_data, categorical_data, mvc_data
    else:
        time_data = np.array([]) if info['no_of_time_features'] == 0 else data[:,:info['transformed_numerical_data_starting_index']]
        numerical_data = np.array([]) if info['no_of_numerical_features'] == 0 else \
                        data[:,info['transformed_numerical_data_starting_index']:info['transformed_categorical_data_starting_index']]
        categorical_data = np.array([]) if info['no_of_categorical_features'] == 0 else \
                        data[:,info['transformed_categorical_data_starting_index']:info['transformed_mvc_data_starting_index']]
        mvc_data = np.array([]) if info['no_of_mvc_features'] == 0 else \
                        data[:,info['transformed_mvc_data_starting_index']:]

        print('time_data.shape :{}'.format(time_data.shape))
        print('numerical_data.shape :{}'.format(numerical_data.shape))
        print('categorical_data.shape :{}'.format(categorical_data.shape))
        print('mvc_data.shape :{}\n'.format(mvc_data.shape))
        print('File: {} Class: {} Function: {} State: {} \n'.format('utils.py', 'None', 'split_data_by_type', 'End'))
        return time_data, numerical_data, categorical_data, mvc_data

def subtract_min_time(time_data):
    print('\nFile: {} Class: {} Function: {} State: {}'.format('utils.py', 'None', 'subtract_min_time', 'Start'))
    print('time_data.shape: {}'.format(time_data.shape))
    result = np.apply_along_axis(
        lambda x: x.astype(float) - np.min(x[np.flatnonzero(x)]) if len(np.flatnonzero(x)) != 0 else x, 
        0, 
        time_data
    )
    print('result.shape: {}'.format(result.shape)) 
    print('File: {} Class: {} Function: {} State: {} \n'.format('utils.py', 'None', 'subtract_min_time', 'End'))
    return result

def difference_between_time_columns(time_data):
    print('\nFile: {} Class: {} Function: {} State: {}'.format('utils.py', 'None', 'difference_between_time_columns', 'Start'))
    no_of_rows, no_of_cols = time_data.shape
    print('time_data.shape: {}'.format((no_of_rows, no_of_cols)))
    result = np.array([])
    for i in range(no_of_cols):
            for j in range(i+1, no_of_cols):
                if len(np.nonzero(time_data[:,i])) > 0 and len(np.nonzero(time_data[:,j])) > 0:
                    difference = time_data[:,i] - time_data[:,j]
                    difference = difference.reshape((-1, 1))
                    result = difference if len(result) == 0 else np.concatenate((result, difference), axis=1)
    print('result.shape: {}'.format(result.shape)) 
    print('File: {} Class: {} Function: {} State: {} \n'.format('utils.py', 'None', 'difference_between_time_columns', 'End'))
    return result

def extract_detailed_time(time_data):
    print('\nFile: {} Class: {} Function: {} State: {}'.format('utils.py', 'None', 'extract_detailed_time', 'Start'))
    no_of_rows, no_of_cols = time_data.shape
    print('time_data.shape: {}'.format((no_of_rows, no_of_cols)))
    result = np.array([])
    for i in range(no_of_cols):
        dates = pd.DatetimeIndex(time_data[:,i])
        dayofweek = dates.dayofweek.values.reshape((-1, 1))
        dayofyear = dates.dayofyear.values.reshape((-1, 1))
        month = dates.month.values.reshape((-1, 1))
        weekofyear = dates.weekofyear.values.reshape((-1, 1))
        day = dates.day.values.reshape((-1, 1))
        hour = dates.hour.values.reshape((-1, 1))
        minute = dates.minute.values.reshape((-1, 1))
        year = dates.year.values.reshape((-1, 1))

        result = dayofweek if len(result) == 0 else np.concatenate((result, dayofweek), axis=1)
        result = dayofyear if len(result) == 0 else np.concatenate((result, dayofyear), axis=1)
        result = month if len(result) == 0 else np.concatenate((result, month), axis=1)
        result = day if len(result) == 0 else np.concatenate((result, day), axis=1)
        result = hour if len(result) == 0 else np.concatenate((result, hour), axis=1)
        result = minute if len(result) == 0 else np.concatenate((result, minute), axis=1)
        result = year if len(result) == 0 else np.concatenate((result, year), axis=1)

    print('result.shape: {}'.format(result.shape)) 
    print('File: {} Class: {} Function: {} State: {} \n'.format('utils.py', 'None', 'extract_detailed_time', 'End'))
    return result

def count_frequency(frequency_map, categorical_or_mvc_data):
    print('\nFile: {} Class: {} Function: {} State: {}'.format('utils.py', 'None', 'count_frequency', 'Start'))
    for i in range(categorical_or_mvc_data.shape[1]):
        count = Counter(categorical_or_mvc_data[:,i])
        frequency_map[i] = count if not i in frequency_map else frequency_map[i] + count
    print('File: {} Class: {} Function: {} State: {} \n'.format('utils.py', 'None', 'count_frequency', 'End'))
    return frequency_map

def encode_frequency(frequency_map, categorical_or_mvc_data):
    print('\nFile: {} Class: {} Function: {} State: {}'.format('utils.py', 'None', 'encode_frequency', 'Start'))
    result = np.array([])
    for i in range(categorical_or_mvc_data.shape[1]):
        counts = dict(frequency_map[i])
        encoded_col = pd.Series(categorical_or_mvc_data[:,i]).map(counts).values.reshape(-1,1)
        result = encoded_col if i == 0 else np.concatenate((result, encoded_col), axis=1)
    print('File: {} Class: {} Function: {} State: {} \n'.format('utils.py', 'None', 'encode_frequency', 'End'))
    return result

def correct_covariate_shift(train_data, test_data, random_state, threshold, n_splits):
    print('\nFile: {} Class: {} Function: {} State: {}'.format('utils.py', 'None', 'correct_covariate_shift', 'Start'))

    # split = len(train_data)
    # scaler = StandardScaler()
    # concat_data = np.concatenate((train_data, test_data), axis=0)
    # transformed_data = scaler.fit_transform(concat_data)
    # train_data, test_data = transformed_data[:split,:], transformed_data[split:,:]

    X = pd.DataFrame(test_data).fillna(0)
    Z = pd.DataFrame(train_data).fillna(0)
    X['is_z'] = 0 # 0 means test set
    Z['is_z'] = 1 # 1 means training set
    XZ = pd.concat( [X, Z], ignore_index=True, axis=0 )

    labels = XZ['is_z'].values
    XZ = XZ.drop('is_z', axis=1).values
    X, Z = X.values, Z.values

    clf = RandomForestClassifier(max_depth=3)
    predictions = np.zeros(labels.shape)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, test_idx) in enumerate(skf.split(XZ, labels)):
        X_train, X_test = XZ[train_idx], XZ[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
            
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:,1]
        predictions[test_idx] = probs

    score = roc_auc_score(labels, predictions)
    print('AUC: {}'.format(score))

    if score <= threshold:
        return None
    predictions_Z = predictions[len(X):]
    weights = (1./predictions_Z) - 1. 
    weights /= np.mean(weights) # we do this to re-normalize the computed log-loss
    print('weights.shape: {}'.format(weights.shape))
    print('File: {} Class: {} Function: {} State: {} \n'.format('utils.py', 'None', 'correct_covariate_shift', 'End'))
    return weights

def compute_weight(predictions, labels, epsilon):
    return 1 / (mean_squared_error(predictions, labels) + epsilon)

def compute_q_statistic(classifiers, data, labels):
    print('\nFile: {} Class: {} Function: {} State: {}'.format('utils.py', 'None', 'compute_q_statistic', 'Start'))
    normalise = q = 0
    for i in range(len(classifiers)):
        for j in range(i + 1, len(classifiers)):
            predictions_1 = np.where(classifiers[i].predict(data) > 0.5, 1, 0)
            predictions_2 = np.where(classifiers[j].predict(data) > 0.5, 1, 0)
            n00, n01, n10, n11 = compute_confusion_matrix(predictions_1, predictions_2, labels)
            q += float((n11 * n00) -  (n01 * n10)) / float((n11 * n00) + (n01 * n10))
            normalise += 1

    div = 1 - (q / float(normalise))
    print('div: {}'.format(div))
    print('File: {} Class: {} Function: {} State: {} \n'.format('utils.py', 'None', 'compute_q_statistic', 'End'))
    return div

def remove_worst_classifier(classifiers, data, labels):
    print('\nFile: {} Class: {} Function: {} State: {}'.format('utils.py', 'None', 'remove_worst_classifier', 'Start'))
    index = 0
    max_q = 0
    for i in range(len(classifiers)):
        new_classifiers = np.delete(classifiers, i)
        q = compute_q_statistic(new_classifiers, data, labels)
        if q > max_q:
            index = i
    print('File: {} Class: {} Function: {} State: {} \n'.format('utils.py', 'None', 'remove_worst_classifier', 'End'))
    return index

def compute_confusion_matrix(predictions_1, predictions_2, labels):
    print('\nFile: {} Class: {} Function: {} State: {}'.format('utils.py', 'None', 'compute_confusion_matrix', 'Start'))
    n00 = n01 = n10 = n11 = 0
    for i in range(len(labels)):
        if (predictions_1[i] == predictions_2[i]) and predictions_1[i] == labels[i]:
            n11 += 1
        elif (predictions_1[i] == predictions_2[i]) and predictions_1[i] != labels[i]:
            n00 += 1
        elif (predictions_1[i] != predictions_2[i]) and predictions_1[i] == labels[i]:
            n10 += 1
        else:
            n01 += 1
    print('n00: {} n01: {} n10: {} n11: {}'.format(n00, n01, n10, n11))
    print('File: {} Class: {} Function: {} State: {} \n'.format('utils.py', 'None', 'compute_confusion_matrix', 'End'))
    return n00, n01, n10, n11

def is_large_dataset(data_size, dataset_size_threshold):
    return data_size > dataset_size_threshold

def has_sufficient_time(dataset_budget_threshold, info):
    return info['dataset_time_spent'] < info['time_budget'] * dataset_budget_threshold
