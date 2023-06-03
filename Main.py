import os
import numpy as np
import pandas as pd
from utils.tools import create_directory
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from utils.data_loader import process_ts_data

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def fit_classifier(all_labels, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=16):
    nb_classes = len(np.unique(all_labels))
    # Create Classifier --------------------------------------------------------
    if classifier_name == "FCN" or classifier_name == "ResNet":
        input_shape = (X_train.shape[1], X_train.shape[2])
    elif classifier_name == "lstm_dcnn" or classifier_name == "MLSTM_FCN":
        input_shape = (X_train.shape[1], X_train.shape[2])
    else:
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    # Call Classifier ----------------------------------------------------------
    classifier = create_classifier(classifier_name, input_shape, nb_classes, verbose=True)
    # Train Classifier ----------------------------------------------------------
    if X_val is None:
        classifier.fit(X_train, y_train, None, None, epochs, batch_size)
    else:
        classifier.fit(X_train, y_train, X_val, y_val, epochs, batch_size)
    return classifier


def create_classifier(classifier_name, input_shape, nb_classes, verbose=False):

    # Networks ------------------------------------------------------------------------------
    if classifier_name == "T_CNN":
        from classifiers import T_CNN
        return T_CNN.Classifier_T_CNN(sub_output_directory, input_shape, nb_classes, verbose)
    if classifier_name == "S_CNN":
        from classifiers import S_CNN
        return S_CNN.Classifier_S_CNN(sub_output_directory, input_shape, nb_classes, verbose)
    if classifier_name == "ST_CNN":
        from classifiers import ST_CNN
        return ST_CNN.Classifier_ST_CNN(sub_output_directory, input_shape, nb_classes, verbose)

    if classifier_name == "DCNN_2L":
        from classifiers import DCNN_2L
        return DCNN_2L.Classifier_DCNN_2L(sub_output_directory, input_shape, nb_classes, verbose)
    if classifier_name == "DCNN_3L":
        from classifiers import DCNN_3L
        return DCNN_3L.Classifier_DCNN_3L(sub_output_directory, input_shape, nb_classes, verbose)
    if classifier_name == "DCNN_4L":
        from classifiers import DCNN_4L
        return DCNN_4L.Classifier_DCNN_4L(sub_output_directory, input_shape, nb_classes, verbose)
    
    # ------------------------------------------------------------------------------------------------------------------
    # Component Analysis -----------------------------------------------------------------------------------------------
    if classifier_name == "FCN":
        from classifiers import FCN
        return FCN.Classifier_FCN(sub_output_directory, input_shape, nb_classes, verbose)
    if classifier_name == "D_FCN":
        from classifiers import D_FCN
        return D_FCN.Classifier_D_FCN(sub_output_directory, input_shape, nb_classes, verbose)

    if classifier_name == "ResNet":
        from classifiers import ResNet
        return ResNet.Classifier_ResNet(sub_output_directory, input_shape, nb_classes, verbose)
    if classifier_name == "D_ResNet":
        from classifiers import D_ResNet
        return D_ResNet.Classifier_D_ResNet(sub_output_directory, input_shape, nb_classes, verbose)

    if classifier_name == "MC_CNN":
        from classifiers import MC_CNN
        return MC_CNN.Classifier_MC_CNN(sub_output_directory, input_shape, nb_classes, verbose)
    if classifier_name == "MLSTM_FCN":
        from classifiers import MLSTM_FCN
        return MLSTM_FCN.Classifier_MLSTM_FCN(sub_output_directory, input_shape, nb_classes, verbose)
    elif classifier_name == "lstm_dcnn":
        from classifiers import lstm_dcnn
        return lstm_dcnn.Classifier_LSTM_DCNN(sub_output_directory, input_shape, nb_classes, verbose)


def s_length(train_df, test_df):
    train_lengths = train_df.applymap(lambda x: len(x)).values
    test_lengths = test_df.applymap(lambda x: len(x)).values

    train_vert_diffs = np.abs(train_lengths - np.expand_dims(train_lengths[0, :], 0))

    if np.sum(train_vert_diffs) > 0:  # if any column (dimension) has varying length across samples
        train_max_seq_len = int(np.max(train_lengths[:, 0]))
        test_max_seq_len = int(np.max(test_lengths[:, 0]))
        max_seq_len = np.max([train_max_seq_len, test_max_seq_len])
    else:
        max_seq_len = train_lengths[0, 0]
    return max_seq_len
# Problem Setting -----------------------------------------------------------------------------------------------------
'''
Disjoint CNN :'DCNN_2L', 'DCNN_3L', 'DCNN_4L'
Temporal CNN : 'T_CNN'  ,  Spatial CNN: 'S_CNN' , Spatial-Temporal CNN : 'ST_CNN' 
Fully Convolutional Network: 'FCN' , Disjoint FCN : 'D_FCN' , Residual Network: 'ResNet', Disjoint ResNet: 'D_ResNet' 
Multivariate LSTM-FCN : 'MLSTM_FCN', Multi-Channel Deep CNN: 'MC_CNN'
'''
# ----------------------------------------------------------------------------------------------------------------------
ALL_Results = pd.DataFrame()
ALL_Results_list = []
problem_index = 0
data_path = os.getcwd() + '/Multivariate_ts/'
# Hyper-Parameter Setting ----------------------------------------------------------------------------------------------
classifier_name = "DCNN_2L"  # Choose the classifier name from aforementioned List
epochs = 500
Resample = 1  # Set to '1' for default Train and Test Sets, and '30' for running on all resampling
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

for problem in os.listdir(data_path):
    # Load Data --------------------------------------------------------------------------------------------------------
    output_directory = os.getcwd() + '/Results_'
    output_directory = output_directory + classifier_name + '/' + problem + '/'
    create_directory(output_directory)
    print("[Main] Problem: {}".format(problem))
    itr_result = [problem]
    # load --------------------------------------------------------------------------
    # set data folder
    data_folder = data_path + problem + "/"
    train_file = data_folder + problem + "_TRAIN.ts"
    test_file = data_folder + problem + "_TEST.ts"
    # Load Train and Test using sktime.utils.load_data function
    X_train, Y_train = load_from_tsfile_to_dataframe(train_file)
    X_test, Y_test = load_from_tsfile_to_dataframe(test_file)
    max_length = s_length(X_train, X_test)
    X_train = process_ts_data(X_train, max_length, normalise=False)
    X_test = process_ts_data(X_test, max_length,  normalise=False)

    all_data = np.vstack((X_train, X_test))
    all_labels = np.hstack((Y_train, Y_test))
    all_indices = np.arange(len(all_data))

    for itr in range(0, Resample):
        sub_output_directory = output_directory + str(itr + 1) + '/'
        create_directory(sub_output_directory)
        # Default Train and Test Set
        if itr == 0:
            x_train = X_train
            x_test = X_test
            y_train = Y_train
            y_test = Y_test
        else:
            training_indices = np.loadtxt("multi_indices/{}_INDICES_TRAIN.txt".format(problem),
                                          skiprows=itr,
                                          max_rows=1).astype(np.int32)
            test_indices = np.loadtxt("multi_indices/{}_INDICES_TEST.txt".format(problem),
                                      skiprows=itr,
                                      max_rows=1).astype(np.int32)
            x_train, y_train = all_data[training_indices, :], all_labels[training_indices]
            x_test, y_test = all_data[test_indices, :], all_labels[test_indices]

        # Making Consistent with Keras Output -------------------------------------------------
        all_labels_new = np.concatenate((y_train, y_test), axis=0)
        print("[Main] All labels: {}".format(np.unique(all_labels_new)))
        tmp = pd.get_dummies(all_labels_new).values
        y_train = tmp[:len(y_train)]
        y_test = tmp[len(y_train):]

        # Making Consistent with Keras Input ---------------------------------------------------
        if classifier_name == "FCN" or classifier_name == "ResNet" or classifier_name == "MLSTM_FCN":
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1])
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[1])
        else:
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        # classifier-----------------------------------------------------------------
        # Dynamic Batch-size base on Data
        if problem == 'EigenWorms' or problem == 'DuckDuck':
            batch_size = 1
        else:
            # batch_size = np.ceil(x_train.shape[0] / (8 * (np.max(y_train.shape[1]) + 1)))
            batch_size = 8
            
        val_index = np.random.randint(0, np.int(x_train.shape[0]), np.int(x_train.shape[0] / 10), dtype=int)
        x_val = x_train[val_index, :]
        y_val = y_train[val_index, :]

    classifier = fit_classifier(all_labels_new, x_train, y_train, x_val, y_val, epochs, batch_size)
    metrics_test, conf_mat = classifier.predict(x_test, y_test, best=True)
    metrics_test2, conf_mat2 = classifier.predict(x_test, y_test, best=False)

    metrics_test['train/val/test/test2'] = 'test'
    metrics_test2['train/val/test/test2'] = 'test2'
    metrics = pd.concat([metrics_test, metrics_test2]).reset_index(drop=True)

    print("[Main] Problem: {}".format(problem))
    print(metrics.head())

    metrics.to_csv(sub_output_directory + 'classification_metrics.csv')
    np.savetxt(sub_output_directory + 'confusion_matrix.csv', conf_mat, delimiter=",")
    itr_result.append(metrics.accuracy[0])
    itr_result.append(metrics.accuracy[1])
    sub_output_directory = []

    if len(ALL_Results_list) == 0:
        ALL_Results_list = np.hstack((ALL_Results_list, itr_result))
    else:
        ALL_Results_list = np.vstack((ALL_Results_list, itr_result))

    problem_index = problem_index + 1

ALL_Results = pd.DataFrame(ALL_Results_list)
ALL_Results.to_csv(os.getcwd() + '/Results_' + classifier_name + '/'+'All_results1.csv')
