import numpy as np
from sklearn.preprocessing import StandardScaler
import random


def process_ts_data(X, vary_len: str = "suffix-noise", normalise: bool = False):
    """
    This is a function to process the data, i.e. convert dataframe to numpy array
    :param X:
    :param normalise:
    :return:
    """
    num_instances, num_dim = X.shape
    columns = X.columns
    max_len = np.max([len(X[columns[0]][i]) for i in range(num_instances)])
    output = np.zeros((num_instances, max_len, num_dim), dtype=np.float64)

    for i in range(num_dim):
        for j in range(num_instances):
            output[j, :, i] = X[columns[i]][j].values
        output[:, :, i] = fill_missing(output[:, :, i],
                                       max_len,
                                       vary_len,
                                       normalise)

    return output

def fill_missing(x: np.array,
                 max_len: int,
                 vary_len: str = "suffix-noise",
                 normalise: bool = True):
    if vary_len == "zero":
        if normalise:
            x = StandardScaler().fit_transform(x)
        x = np.nan_to_num(x)
    elif vary_len == 'prefix-suffix-noise':
        for i in range(len(x)):
            series = list()
            for a in x[i, :]:
                if np.isnan(a):
                    break
                series.append(a)
            series = np.array(series)
            seq_len = len(series)
            diff_len = int(0.5 * (max_len - seq_len))

            for j in range(diff_len):
                x[i, j] = random.random() / 1000

            for j in range(diff_len, seq_len):
                x[i, j] = series[j - seq_len]

            for j in range(seq_len, max_len):
                x[i, j] = random.random() / 1000

            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]
    elif vary_len == 'uniform-scaling':
        for i in range(len(x)):
            series = list()
            for a in x[i, :]:
                if np.isnan(a):
                    break
                series.append(a)
            series = np.array(series)
            seq_len = len(series)

            for j in range(max_len):
                scaling_factor = int(j * seq_len / max_len)
                x[i, j] = series[scaling_factor]
            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]
    else:
        for i in range(len(x)):
            for j in range(len(x[i])):
                if np.isnan(x[i, j]):
                    x[i, j] = random.random() / 1000

            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]

    return x


