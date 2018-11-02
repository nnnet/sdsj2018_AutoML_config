import pandas as pd
import numpy as np
import lightgbm as lgb
import hyperopt
from hyperopt import hp, tpe, STATUS_OK, space_eval, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from lib.util import timeit, log, Config
from typing import List, Dict
from lib.model_other import train_h2o, predict_h2o


# @timeit
# def train(X: pd.DataFrame, y: pd.Series, config: Config):
    
#     if 'models' in config.data and 'lst_models' in config['models']:
#         if config['models']['lst_models'][0]=='h2o':
#             train_h2o(X, y, config)
#         elif config['models']['lst_models'][0]=='lightgbm':
#             train_lightgbm(X, y, config)
#         else:
#             raise ValueError('Train: Unknow model name "{0}"'.format(config['models']['lst_models'][0]))
            
#     elif config["nrows"] < 1000:
#         train_h2o(X, y, config)
#     else:
#         train_lightgbm(X, y, config)


# @timeit
# def predict(X: pd.DataFrame, config: Config) -> List:
# #    preds = predict_lightgbm(X, config)
#     if 'models' in config.data and 'lst_models' in config['models']:
#         if config['models']['lst_models'][0]=='h2o':
#             preds = predict_h2o(X, config)
#         elif config['models']['lst_models'][0]=='lightgbm':
#             preds = predict_lightgbm(X, config)
#         else:
#             raise ValueError('Predict: Unknow model name "{0}"'.format(config['models']['lst_models'][0]))
            
#     elif config["nrows"] < 1000:
#         preds = predict_h2o(X, config)
#     else:
#         preds = predict_lightgbm(X, config)

#     if config["non_negative_target"]:
#         preds = [max(0, p) for p in preds]

#     return preds


@timeit
def validate(preds: pd.DataFrame, target_csv: str, mode: str) -> np.float64:
    # .rename(columns={'prediction':'target'})
    df = pd.merge(preds, pd.read_csv(target_csv), on="line_id", left_index=True)
    score = roc_auc_score(df.target.values, df.prediction.values) if mode == "classification" else \
        np.sqrt(mean_squared_error(df.target.values, df.prediction.values))
    log("Score: {:0.4f}".format(score))
    return score
# roc_auc_score(df.target.values.reshape(-1,1), df.prediction.values.reshape(-1,1))
    
@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Config):
    params = {
        "objective": "regression" if config["mode"] == "regression" else "binary",
        "metric": "rmse" if config["mode"] == "regression" else "auc",
        "verbosity": -1,
        "seed": 1,
    }

    X_sample, y_sample = data_sample(X, y)
    hyperparams = hyperopt_lightgbm(X_sample, y_sample, params, config)

    X_train, X_val, y_train, y_val = data_split(X, y)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    config['params']['pipeline'][config["stage"]]["model"] = \
            lgb.train({**params, **hyperparams}, train_data, 3000, valid_data, \
                          early_stopping_rounds=50, verbose_eval=100)


@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:
    return config['params']['pipeline'][config["stage"]]['model'].predict(X)


@timeit
def hyperopt_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config):
    X_train, X_val, y_train, y_val = data_split(X, y, test_size=0.5)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.05),
        "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
        "reg_alpha": hp.uniform("reg_alpha", 0, 30),
        "reg_lambda": hp.uniform("reg_lambda", 0, 30),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
    }

    def objective(hyperparams):
        model = lgb.train({**params, **hyperparams}, train_data, 300, valid_data,
                          early_stopping_rounds=100, verbose_eval=100)

        score = model.best_score["valid_0"][params["metric"]]
        if config.is_classification():
            score = -score

        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials, algo=tpe.suggest, max_evals=50, verbose=1,
                         rstate=np.random.RandomState(1))

    hyperparams = space_eval(space, best)
    log("{:0.4f} {}".format(trials.best_trial['result']['loss'], hyperparams))
    return hyperparams



def data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    return train_test_split(X, y, test_size=test_size, random_state=1)


def data_sample(X: pd.DataFrame, y: pd.Series, nrows: int=5000) -> (pd.DataFrame, pd.Series):
    if len(X) > nrows:
        X_sample = X.sample(nrows, random_state=1)
        y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample
