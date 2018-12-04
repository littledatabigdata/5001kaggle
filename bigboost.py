import xgboost as xgb
import pandas as pd

df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

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
dtest = xgb.DMatrix(t_processed.values)

for bag in range(1):
    # bagging, use 20% as validation
    train = processed.sample(frac=0.75, replace=True)
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

    dtrain = xgb.DMatrix(X, label=Y)
    dvali = xgb.DMatrix(v_X, label=v_Y)

    param = {
        'booster': 'gbtree',
        'eta': 0.05,
        'gamma': 1,
        'max_depth': 4,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'min_child_weight': 1,
        # 'lambda': 1.35,
        # 'alpha': 0.15,
        'objective': 'gpu:reg:linear',
        'eval_metric': 'rmse'
    }

    evallist = [(dvali, 'eval'), (dtrain, 'train')]

    num_round = 1500
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=85, verbose_eval=85)

    ypred = bst.predict(dtest) / 10
    pp = pd.DataFrame(data=ypred, columns=['time'])
    pp.to_csv('./xgbpred/xgb' + str(bag) + '.csv')