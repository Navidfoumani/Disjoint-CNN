import numpy as np
import pandas as pd

import sklearn.metrics as m
from sklearn.metrics import confusion_matrix



def get_binary_labels_softmax(y):
    # get the driving labels as a binary 1D array
    y_1 = np.asarray([1 * (y[i][1] > y[i][0]) for i in range(len(y))])
    # get the distracted labels as a binary 1D array
    y_0 = abs(y_1 - 1)
    return y_0, y_1


def get_binary_labels(y):
    # get the driving labels as a binary 1D array
    y_1 = y
    # get the distracted labels as a binary 1D array
    y_0 = abs(y_1 - 1)

    return y_0, y_1


def get_predictions_from_softmax(y):
    return np.argmax(y, axis=1)


def predict_model(model, X, y, output_dir=''):

    # make a prediction with the model
    if output_dir.find('EigenWorms') or output_dir.find('DuckDuck'):
        batch_size = np.ceil(X.shape[0] / (8 * (np.max(y.shape[1]) + 1)))
        yhat = model.predict(X, np.int(batch_size))
    else:
        yhat = model.predict(X)

    ytrue = get_predictions_from_softmax(y)
    yhat = get_predictions_from_softmax(yhat)

    result_dic = {}
    result_dic["accuracy"] = []
    result_dic["accuracy"].append(m.accuracy_score(ytrue, yhat))
    conf_mat = confusion_matrix(ytrue, yhat)
    #print(conf_mat)
    # ---------------------------------------------------------
    pd.DataFrame(yhat).to_csv(output_dir + 'yhat.csv')
    pd.DataFrame(ytrue).to_csv(output_dir + 'y_true.csv')
    # ---------------------------------------------------------
    # put the model predictions into a dataframe
    model_metrics = pd.DataFrame.from_dict(result_dic)
    # calculate summary stats
    return model_metrics, conf_mat, ytrue, yhat

