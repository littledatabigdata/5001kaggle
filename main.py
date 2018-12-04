import numpy as np
import pandas as pd
# from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
import datetime
import time

df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

NUM_IT = 100

# TODO: things to try tomorrow
# try to train a clf with > 0.9 accuracy
# idea: train a bunch of other classifiers
# idea: automate parameter tweaking, brute force all possible combos and just train and see the accuracy

def proc_data(src):
    # dropping random_state because it's RNG
    # dropping scale because it's irrelevant
    # dropping alpha because it's a constant multiplication
    src.drop(['random_state', 'scale', 'alpha', 'id'], axis=1, inplace=True)

    # fill missing values in penalty with l2 because it's the default
    # one hot encode penalty because it's nominal
    cat = src['penalty'].astype('category', categories=['l2', 'elasticnet', 'l1'])
    cat = cat.fillna('l2')

    encoded = pd.get_dummies(data=cat, dummy_na=False)
    src.drop(['penalty'], axis=1, inplace=True)

    # For decreasing model complexity
    # dropping penalty because it's not too important
    # multiply class and clusters per class then drop them
    # src['clusters'] = src['n_classes'] * src['n_clusters_per_class']
    # src.drop(['penalty', 'n_clusters_per_class', 'n_classes'], axis=1, inplace=True)

    # n_jobs replace -1 with the max number of cores 16 (from test data)
    src.replace(-1, 16, inplace=True)

    return pd.concat([src, encoded], axis=1)

processed = proc_data(df)
t_processed = proc_data(test)


for it in range(NUM_IT):
    best = 0
    for bag in range(10):
        # bagging, use 20% as validation
        train = processed.sample(frac=0.8, replace=True)
        validation = processed.drop(train.index)

        # multiple label by 10 then drop the label
        label = train['time'] * 10
        train.drop(['time'], axis=1, inplace=True)

        v_label = validation['time'] * 10
        validation.drop(['time'], axis=1, inplace=True)

        X = train.values
        Y = label.values

        v_X = validation.values
        v_Y = v_label.values

        # TODO: fiddle with params
        clf = GradientBoostingRegressor(learning_rate=0.11, n_estimators=200, subsample=0.8, max_depth=5, warm_start=False)

        clf.fit(X, Y)

        # validation
        # print train.columns
        # print [x for x in clf.feature_importances_]
        score = clf.score(v_X, v_Y)
        if score > best:
            best = score
            best_clf = clf

    # print best
    if best >= 0.9:
        prediction = best_clf.predict(t_processed.values) / float(10)

        pp = pd.DataFrame(data=prediction, columns=['time'])
        pp.to_csv('./good_res/results' + str(best) + '.csv')