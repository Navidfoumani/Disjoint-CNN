import math

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# Most of the code here are taken from https://github.com/hfawaz/dl-4-tsc

def prepare_inputs_deep_learning(train_inputs, test_inputs, window_len=40, stride=20,
                                 val_size=1, random_state=1234, verbose=1):
    # This function prepare the inputs to have the right shape for deep learning models.
    # The shape we are after is (n_series, series_len, series_dim)
    # Inputs are df with data and label columns
    # Inputs:
    #   train_inputs:   training dataset
    #   test_inputs:    test dataset
    #   window_len:     subsequence window size
    #   stride:         stride
    #   val_size:       number of series to be used as validation
    #   random_state:   random seed
    #   binary:         whether we convert to binary case
    #   verbose:        verbosity
    if verbose > 0:
        print('[ClassifierTools] Preparing inputs')

    if len(train_inputs) > val_size:
        train_series, val_series = train_test_split([x for x in range(len(train_inputs))],
                                                    test_size=val_size,
                                                    random_state=random_state)
    else:
        train_series = range(len(train_inputs))
        val_series = None

    X_train = []
    y_train = []
    for i in train_series:
        this_series = train_inputs.data[i]
        this_series_labels = train_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=window_len,
                                                       stride=stride)
        [X_train.append(x) for x in subsequences]
        [y_train.append(x) for x in sub_label]
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    if val_series is None:
        X_val = None
        y_val = None
    else:
        X_val = []
        y_val = []
        for i in val_series:
            this_series = train_inputs.data[i]
            this_series_labels = train_inputs.label[i]
            subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                           window_size=window_len,
                                                           stride=stride)
            [X_val.append(x) for x in subsequences]
            [y_val.append(x) for x in sub_label]
        X_val = np.array(X_val)
        y_val = np.array(y_val)

    X_test = []
    y_test = []
    for i in range(len(test_inputs)):
        this_series = test_inputs.data[i]
        this_series_labels = test_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=window_len,
                                                       stride=stride)
        [X_test.append(x) for x in subsequences]
        [y_test.append(x) for x in sub_label]
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_inputs_cnn_lstm(train_inputs, test_inputs, window_len=40, stride=20,
                            val_size=1, random_state=1234, n_subs=4, verbose=1):
    # This function prepare the inputs to have the right shape for deep learning models specifically CNN-LSTM models.
    # The idea is to get n_subs subsequences of length=window_len, pass each of them to a CNN for features and
    # learn the relationship with past subsequences using LSTM.
    # The shape we are after is (n_series, n_subs, series_len, series_dim)
    # Inputs are df with data and label columns
    # Inputs:
    #   train_inputs:   training dataset
    #   test_inputs:    test dataset
    #   window_len:     subsequence window size
    #   stride:         stride
    #   val_size:       number of series to be used as validation
    #   random_state:   random seed
    #   n_subs:         number of subsequences used to learn the relationship
    #   binary:         whether we convert to binary case
    #   verbose:        verbosity

    n_length = window_len
    larger_window = window_len * n_subs
    if verbose > 0:
        print('[ClassifierTools] Preparing inputs')

    train_series, val_series = train_test_split([x for x in range(len(train_inputs))],
                                                test_size=val_size,
                                                random_state=random_state)
    X_train = []
    y_train = []
    for i in train_series:
        this_series = train_inputs.data[i]
        this_series_labels = train_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=larger_window,
                                                       stride=stride)
        [X_train.append(x) for x in subsequences]
        [y_train.append(x) for x in sub_label]
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train = X_train.reshape((X_train.shape[0], n_subs, n_length, X_train.shape[2]))

    X_val = []
    y_val = []
    for i in val_series:
        this_series = train_inputs.data[i]
        this_series_labels = train_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=larger_window,
                                                       stride=stride)
        [X_val.append(x) for x in subsequences]
        [y_val.append(x) for x in sub_label]
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    X_val = X_val.reshape((X_val.shape[0], n_subs, n_length, X_val.shape[2]))

    X_test = []
    y_test = []
    for i in range(len(test_inputs)):
        this_series = test_inputs.data[i]
        this_series_labels = test_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=larger_window,
                                                       stride=stride)
        [X_test.append(x) for x in subsequences]
        [y_test.append(x) for x in sub_label]
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_test = X_test.reshape((X_test.shape[0], n_subs, n_length, X_test.shape[2]))

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_inputs(train_inputs, test_inputs, window_len=40, stride=20, verbose=1):
    # This function prepare the inputs to have the right shape for ML models without validation set.
    # The shape we are after is (n_series, series_len, series_dim)
    # Inputs are df with data and label columns
    # Inputs:
    #   train_inputs:   training dataset
    #   test_inputs:    test dataset
    #   window_len:     subsequence window size
    #   stride:         stride
    #   binary:         whether we convert to binary case
    #   class_one:      the classes to be used as class one
    #   verbose:        verbosity
    if verbose > 0:
        print('[ClassifierTools] Preparing inputs')

    X_train = []
    y_train = []
    for i in range(len(train_inputs)):
        this_series = train_inputs.data[i]
        this_series_labels = train_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=window_len,
                                                       stride=stride)
        [X_train.append(x) for x in subsequences]
        [y_train.append(x) for x in sub_label]
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = []
    y_test = []
    for i in range(len(test_inputs)):
        this_series = test_inputs.data[i]
        this_series_labels = test_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=window_len,
                                                       stride=stride)
        [X_test.append(x) for x in subsequences]
        [y_test.append(x) for x in sub_label]
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test


def extract_subsequences(X_data, y_data, window_size=30, stride=1, norm=False):
    # This function extract subsequences from a long time series.
    # Assumes that each timestamp has a label represented by y_data.
    # The label for each subsequence is taken with the majority class in that segment.
    data_len, data_dim = X_data.shape
    subsequences = []
    labels = []
    count = 0
    for i in range(0, data_len, stride):
        end = i + window_size
        if end > data_len:
            break
        tmp = X_data[i:end, :]
        if norm:
            # usually z-normalisation is required for TSC
            scaler = StandardScaler()
            tmp = scaler.fit_transform(tmp)
        subsequences.append(tmp)
        label = stats.mode(y_data[i:end]).mode[0]
        labels.append(label)
        count += 1
    return np.array(subsequences), np.array(labels)


def create_class_weight(labels, mu=2):
    labels_dict = {}
    total = 0
    for i in range(labels.shape[1]):
        labels_dict.update({i: sum(labels[:, i])})
        total += sum(labels[:, i])

    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

'''
if binary:
    label = make_binary(label, class_one=class_one)

def make_binary(y, class_one):
    if y in class_one:
        return 1
    else:
        return 0
'''
