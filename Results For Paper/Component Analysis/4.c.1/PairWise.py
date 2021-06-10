
# Wilcoxon Signed-Rank Test
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def pairwise_comparison(df, model_a, model_b):
    #df = df[["dataset_name", model_a, model_b]].copy()
    win = len(df.loc[df[model_a] > df[model_b]])
    tie = len(df.loc[df[model_a] == df[model_b]])
    loss = len(df.loc[df[model_a] < df[model_b]])

    plt.figure(figsize=(7 ,7))
    plt.scatter(df[model_b], df[model_a])
    plt.plot(np.arange(0 ,2 ,0.1), np.arange(0 ,2 ,0.1), 'r')
    plt.ylabel(model_a, fontsize=15)
    plt.xlabel(model_b, fontsize=15)
    plt.title("{} vs {} {}/{}/{}".format(model_a, model_b, win, tie, loss), fontsize=15)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.text(0.55, 0.05, "{} is better here".format(model_b))
    plt.text(0.05, 0.95, "{} is better here".format(model_a))
    plt.savefig('{} vs {}.eps'.format(model_a, model_b), bbox_inches='tight')
    plt.show()

    n_dataset = df.dataset_name.nunique()
    df2 = df.melt(id_vars="dataset_name", var_name="classifier_name", value_name="accuracy")

    rank_data = np.array(df2['accuracy']).reshape(2, n_dataset)
    # create the data frame containg the accuracies
    df_ranks = pd.DataFrame(data=rank_data, index=[model_a, model_b], columns=np.unique(df2['dataset_name']))

    # number of wins
    dfff = df_ranks.rank(ascending=False)
    print("Number of wins")
    print(dfff[dfff == 1.0].sum(axis=1))
    print()
    # average the ranks
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    print("Average Rank")
    print(average_ranks)
    print()

    stat, p = wilcoxon(df[model_a], df[model_b], zero_method='pratt')
    print("Wilcoxon Signed-Rank Test")
    print("stat=%.3f, p=%.3e" % (stat, p))
    if p > 0.05:
        if win > loss:
            print('{} is not significant more accurate than {}'.format(model_a, model_b))
        elif win < loss:
            print('{} is not significant more accurate than {}'.format(model_b, model_a))
        else:
            print('Difference between {} and {} is not significant'.format(model_a, model_b))
    else:
        if win > loss:
            print('{} is significantly more accurate than {}'.format(model_a, model_b))
        else:
            print('{} is significantly more accurate than {}'.format(model_b, model_a))


df_all = pd.read_csv('Temporal-spatial.csv',index_col=False)
#df_Temporal = pd.read_csv('Temporal.csv',index_col=False)
#df_Spatial = pd.read_csv('Spatial.csv',index_col=False)
pairwise_comparison(df_all, 'Temporal', 'Spatial')