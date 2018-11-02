import pandas as pd
import lightgbm as lgb
from boruta import BorutaPy
from typing import List, Optional


class LGBMFeatureEstimator():
    def __init__(self, params, n_estimators:int=50):
        self.params = params
        self.n_estimators = n_estimators

    def get_params(self):
        return self.params

    def set_params(self, n_estimators:int=None, random_state:int=None):
        if n_estimators is not None:
            self.n_estimators = n_estimators
        if random_state is not None:
            self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series, importance_type:str="gain"):
        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(self.params, train_data, self.n_estimators)
        self.feature_importances_ = model.feature_importance(importance_type=importance_type)


def select_features(X: pd.DataFrame, y: pd.Series, mode: str, n_estimators: int=50, 
                    max_iter: int=100, perc: int=75,
                    learning_rate:float=0.01, verbosity:int=-1,
                    seed:int=1, max_depth:int=-1,
                    random_state:int=1, verbose:int=2
                    ) -> List[str]:
    feat_estimator = LGBMFeatureEstimator({
        "objective": "regression" if mode == "regression" else "binary",
        "metric": "rmse" if mode == "regression" else "auc",
        "learning_rate": learning_rate,
        "verbosity": verbosity,
        "seed": seed,
        "max_depth": max_depth,
    }, n_estimators)

    feat_selector = BorutaPy(feat_estimator, n_estimators=n_estimators, max_iter=max_iter, 
                             verbose=verbose, random_state=random_state, perc=perc)

    try:
        feat_selector.fit(X.values, y.values.ravel())
    except TypeError:
        pass

    return X.columns[feat_selector.support_].tolist()
