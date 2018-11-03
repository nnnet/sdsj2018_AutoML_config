import copy
import datetime
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lib.features import select_features
from lib.util import timeit, log, Config

from copy import deepcopy
# import gc, psutil
from typing import Optional
from lib.model import \
        train_lightgbm, predict_lightgbm
from lib.model_other import \
        train_vw, predict_vw, \
        train_h2o, predict_h2o, \
        train_lm, predict_lm, \
        train_rand_forest, predict_rand_forest, \
        train_linear_cv, predict_linear_cv, \
        train_bayesian, predict_bayesian, \
        train_arima, predict_arima
import time


@timeit
def pipeline(df: pd.DataFrame, config: Config
             , train_csv: str=None
             , test_csv: str=None, prediction_csv: str=None) -> (pd.DataFrame, Optional[np.float64]):

    if config.is_train():
        config['stages'] = {}
        
    for ids, stage in enumerate(config['graph']):
        if len(stage)==0 or stage[0] is None or stage[0]=='':
            config["stage"] = '{0}/n{1}'.format(config["stage"],
                               'Error value stage "{0}" in pipeline'.format(stage))
            raise ValueError(config["stage"])
            
        config["stage"] = stage[0]
        config["stage_nb"] = ids
        
        if config.is_train():
            config['stages'][config["stage"]] = {}
        
        config['stages'][config["stage"]]['time'] = 0
        start_time = time.time()
        
        if stage[0] == 'Start':
            continue
        # elif stage[0] == 'End':
        #     break

        elif not stage[0] in config['params']['pipeline']:
            config["stage"] = '{0}/n{1}'.format(config["stage"],
                               'Unknow node "{0}" in pipeline'.format(stage[0]))
            raise ValueError(config["stage"])

        elif not config['params']['pipeline'][stage[0]]['node'] in _node_map:
            config["stage"] = '{0}/n{1}'.format(config["stage"],
                               'Unknow node "{0}" in _node_map'.format(
                                       config['params']['pipeline'][stage[0]]['node']))
            raise ValueError(config["stage"])
            
        node = _node_map[config['params']['pipeline'][stage[0]]['node']]
        if node.name == 'read_df':
            if config.is_train():
                df = node(train_csv, config)

        elif 'args' in config['params']['pipeline'][stage[0]] \
                and len(config['params']['pipeline'][stage[0]]['args'])!=0:
            node.function(df, config, **config['params']['pipeline'][stage[0]]['args'])
        else:
            node(df, config)
        
        stage_time_inc(config, start_time, stage[0])


def stage_time_inc(config, start_time, stage_name):
    if 'stages_time' in config:
        if stage_name in config['stages_time']:
            config['stages_time'][stage_name] += (time.time() - start_time)
        else:
            config['stages_time'][stage_name] = time.time() - start_time


@timeit
def model(df: pd.DataFrame, config: Config, models: list): #  -> (Optional[pd.DataFrame], Optional[pd.Series])
    if config.is_train():
        X, y = split_X_y(df, config)
        if config.is_train():
            # train(X, y, config)
    
            if 'args' in config['params']['pipeline'][config["stage"]] and \
                'models' in config['params']['pipeline'][config["stage"]]['args']:
                if config['params']['pipeline'][config["stage"]]['args']['models'][0]=='h2o':
                    train_h2o(X, y, config)
                elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='lightgbm':
                    train_lightgbm(X, y, config)
                elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='vw':
                    train_vw(X, y, config)
                elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='lm':
                    train_lm(X, y, config)
                elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='rf':
                    train_rand_forest(X, y, config)
                elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='lcv':
                    train_linear_cv(X, y, config)
                elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='bayes':
                    train_bayesian(X, y, config)
                elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='arima':
                    train_arima(X, y, config)
                else:
                    raise ValueError('Train: Unknow model name "{0}"'.format(config['params']['pipeline'][config["stage"]]['args']['models'][0]))
                    
            else:
                config['params']['pipeline'][config["stage"]]['models'] = []
                if config["nrows"] < 1000:
                    train_h2o(X, y, config)
                    config['params']['pipeline'][config["stage"]]['models'].append('h2o')
                else:
                    train_lightgbm(X, y, config)
                    config['params']['pipeline'][config["stage"]]['models'].append('lightgbm')

        if config['params']['pipeline'][config["stage"]]['args']['models'][0]=='h2o':
            df[config["stage"]] = predict_h2o(X, config)
        elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='lightgbm':
            df[config["stage"]] = predict_lightgbm(X, config)
        elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='vw':
            df[config["stage"]] = predict_vw(X, y, config)
        elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='lm':
            df[config["stage"]] = predict_lm(X, config)
        elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='rf':
            df[config["stage"]] = predict_rand_forest(X, config)
        elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='lcv':
            df[config["stage"]] = predict_linear_cv(X, config)
        elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='bayes':
            df[config["stage"]] = predict_bayesian(X, config)
        elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='arima':
            df[config["stage"]] = predict_arima(X, config)
        
    else:
        if config['params']['pipeline'][config["stage"]]['args']['models'][0]=='h2o':
            df[config["stage"]] = predict_h2o(df, config)
        elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='lightgbm':
            df[config["stage"]] = predict_lightgbm(df, config)
        elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='vw':
            df[config["stage"]] = predict_vw(df, y, config)
        elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='lm':
            df[config["stage"]] = predict_lm(df, config)
        elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='rf':
            df[config["stage"]] = predict_rand_forest(df, config)
        elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='lcv':
            df[config["stage"]] = predict_linear_cv(df, config)
        elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='bayes':
            df[config["stage"]] = predict_bayesian(df, config)
        elif config['params']['pipeline'][config["stage"]]['args']['models'][0]=='arima':
            df[config["stage"]] = predict_arima(df, config)

        

    if config["non_negative_target"]:
        df[config["stage"]] = df[config["stage"]].apply(lambda p: max(0, p))


@timeit
def split_X_y(df: pd.DataFrame, config: Config) -> (pd.DataFrame, Optional[pd.Series]):
    if config['params']['field_target_name'] in df.columns:
        return df.drop(config['params']['field_target_name'], axis=1), df[config['params']['field_target_name']]
    else:
        return df, None

@timeit
def columns_float64_to_32(df: pd.DataFrame, config: Config):
    columns_float64 = df.select_dtypes(include=['float64']).columns
    df[columns_float64] = df[columns_float64].astype(np.float32)
    

@timeit
def check_columns_exists(df: pd.DataFrame, config: Config
                         , key_stage: str
                         , drop_columns_test: bool =True):
    field_target_name=config['params']['field_target_name'] 
    if config.is_train():
        if not 'columns_exists' in config['params']['pipeline'][config["stage"]]:
            config['params']['pipeline'][config["stage"]]['columns_exists'] = {}
            if not field_target_name in df.columns:
                raise ValueError('Column y="{0}" not exists in train dataset'.format(field_target_name))
            
        config['params']['pipeline'][config["stage"]]['columns_exists'][key_stage] = \
                                    set([x for x in df.columns if x!=field_target_name])

    elif 'columns_exists' in config['params']['pipeline'][config["stage"]]:
        if key_stage in config['params']['pipeline'][config["stage"]]['columns_exists']:
            set_columns = config['params']['pipeline'][config["stage"]]['columns_exists'][key_stage] - set(df.columns)
            if len(set_columns)!=0:
                raise ValueError('Columns "{0}" not exists in test dataset on stage {1}'.format(str(set_columns), key_stage))
    
            set_columns = set(df.columns) - config['params']['pipeline'][config["stage"]]['columns_exists'][key_stage]
            if len(set_columns)!=0:
                if drop_columns_test:
                    df.drop(columns=[x for x in set_columns], inplace=True)
                else:
                    raise ValueError('Columns "{0}" not exists in train dataset on stage {1}'.format(str(set_columns), key_stage))
        else:
            raise ValueError('Preprocess stage "{0}" not exists'.format(key_stage))
             

@timeit
def drop_columns(df: pd.DataFrame, config: Config):
    df.drop([c for c in ["is_test", "line_id"] if c in df], axis=1, inplace=True)
    drop_constant_columns(df, config)


def calc_columns_metric(df: pd.DataFrame, columns: list, metric: str, value=None):
    if not value is None:
        return pd.Series([value]*len(columns), index=columns)
    elif metric is None:
        return pd.Series([-1]*len(columns), index=columns)
    else:
        if columns[0].startswith('datetime_'):
            return pd.Series([pd.to_datetime(df[x].dropna().astype(np.int64).agg(metric)) for x in columns]
                             , index = columns)
        else:
            return df.loc[:, columns].agg(metric, axis=0)

def fillna_columns(df: pd.DataFrame, columns_metric: pd.Series):
    for c in [x for x in list(columns_metric.index) if x in df.columns ]:
        df[c].fillna(columns_metric[c], inplace=True)

    return df

@timeit
def fillna(df: pd.DataFrame, config: Config, args: dict={}):

    if len(args)!=0:

        for k, v in args.items():

            if config.is_train():
                lst_columns = [c for c in df if c.startswith(k)]
                config['stages'][config["stage"]][k] = {'lst_columns': lst_columns}
                    
                if len(lst_columns) != 0:
                    if 'agg' in v or 'value' in v:
    
                        if config.is_train():
                            s_fillna_values = calc_columns_metric(df, lst_columns
                                , metric = v['agg'] if 'agg' in v else None
                                , value = v['value'] if 'value' in v else None)
                            
                            config['stages'][config["stage"]][k]['fillna_values'] = deepcopy(s_fillna_values)

            if len(config['stages'][config["stage"]][k]['lst_columns'])!=0:    
                fillna_columns(df, config['stages'][config["stage"]][k]['fillna_values'])
        
    else:

        for c in [c for c in df if c.startswith("number_")]:
            df[c].fillna(-1, inplace=True)
    
        for c in [c for c in df if c.startswith("string_")]:
            df[c].fillna("", inplace=True)
    
        for c in [c for c in df if c.startswith("datetime_")]:
            df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)

def get_constant_columns(df: pd.DataFrame
                         , limit_unique_float=1
                         , limit_unique_object=1
                         , limit_unique_datetime64=1
                         ):
    constant_columns = [c for c in df.select_dtypes(include=['float32']) \
                            if len(np.unique(df[ np.isfinite(df[c]) ][c].values)) <= limit_unique_float]

    constant_columns += [c for c in df.select_dtypes(include=['object'])
                            if len(np.unique(df[ ~pd.isnull(df[c]) ][c].values)) <= limit_unique_object]

    constant_columns += [c for c in df.select_dtypes(include=['datetime64[ns]'])
                          if len(np.unique(df[ ~pd.isnull(df[c]) ][c].values)) <= limit_unique_datetime64]

    return constant_columns
    
@timeit
def drop_constant_columns(df: pd.DataFrame, config: Config):
    if "constant_columns" not in config:
        config["constant_columns"] = get_constant_columns(df)
        
        log("Constant columns: " + ", ".join(config["constant_columns"]), config.verbose)

    if len(config["constant_columns"]) > 0:
        df.drop(config["constant_columns"], axis=1, inplace=True)


@timeit
def transform_datetime(df: pd.DataFrame, config: Config):
    date_parts = ["year", "weekday", "month", "day", "hour"]

    if "date_columns" not in config:
        config["date_columns"] = {}

        for c in [c for c in df if c.startswith("datetime_")]:
            config["date_columns"][c] = []
            for part in date_parts:
                part_col = c + "_" + part
                df[part_col] = getattr(df[c].dt, part).astype(np.uint16 if part == "year" else np.uint8).values

                if not (df[part_col] != df[part_col].iloc[0]).any():
                    log(part_col + " is constant", config.verbose)
                    df.drop(part_col, axis=1, inplace=True)
                else:
                    config["date_columns"][c].append(part)

            df.drop(c, axis=1, inplace=True)
    else:
        for c, parts in config["date_columns"].items():
            for part in parts:
                part_col = c + "_" + part
                df[part_col] = getattr(df[c].dt, part)
            df.drop(c, axis=1, inplace=True)


@timeit
def transform_categorical(df: pd.DataFrame, config: Config):
    if "categorical_columns" not in config:
        # https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
        prior = config["categorical_prior"] = df["target"].mean()
        min_samples_leaf = int(0.01 * len(df))
        smoothing = 0.5 * min_samples_leaf

        config["categorical_columns"] = {}
        for c in [c for c in df if c.startswith("string_")]:
            averages = df[[c, "target"]].groupby(c)["target"].agg(["mean", "count"])
            smooth = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
            averages["target"] = prior * (1 - smooth) + averages["mean"] * smooth
            config["categorical_columns"][c] = averages["target"].to_dict()

        log(list(config["categorical_columns"].keys()), config.verbose)

    for c, values in config["categorical_columns"].items():
        df.loc[:, c] = df[c].apply(lambda x: values[x] if x in values else config["categorical_prior"])


@timeit
def scale(df: pd.DataFrame, config: Config):
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    scale_columns = [c for c in df if c.startswith("number_") 
                            and df[c].dtype != np.int8 
                            and c not in config["categorical_columns"]]

    if len(scale_columns) > 0:

        if config.is_train():
            config['stages'][config["stage"]]['scale_columns'] = deepcopy(scale_columns)

            config['stages'][config["stage"]]['model'] = StandardScaler(copy=False)
            config['stages'][config["stage"]]['scale_columns'] = deepcopy(scale_columns)
            config['stages'][config["stage"]]['model'].fit(df[scale_columns].astype(np.float32))

        df[config['stages'][config["stage"]]['scale_columns']] = \
            config['stages'][config["stage"]]['model'].transform( \
                       df[config['stages'][config["stage"]]['scale_columns']].astype(np.float32) ).astype(np.float32)


@timeit
def to_int8(df: pd.DataFrame, config: Config):
    config['stages'][config["stage"]]['lst_columns'] = []
    vals = [-1, 0, 1]

    for c in [c for c in df if c.startswith("number_")]:
        if (~df[c].isin(vals)).any():
            continue
        config['stages'][config["stage"]]['lst_columns'].append(c)

    log(config['stages'][config["stage"]]['lst_columns'], config.verbose)

    if len(config['stages'][config["stage"]]['lst_columns']) > 0:
        df.loc[:, config['stages'][config["stage"]]['lst_columns']] = \
            df.loc[:, config['stages'][config["stage"]]['lst_columns']].astype(np.int8)


def get_sample_rows(df: pd.DataFrame, config: Config):
    df_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    mem_per_row = df_size_mb / len(df)
    sample_rows = int(config['params']['memory']['max_size_mb'] / mem_per_row)
    
    return df_size_mb, sample_rows


@timeit
def subsample(df: pd.DataFrame, config: Config):
    if config.is_train():
        # df_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        df_size_mb, sample_rows = get_sample_rows(df, config)

        if df_size_mb > config['params']['memory']['max_size_mb']:
            # mem_per_row = df_size_mb / len(df)
            # sample_rows = int(config['params']['memory']['max_size_mb'] / mem_per_row)

            log("Size limit exceeded: {:0.2f} Mb. Dataset rows: {}. Subsample to {} rows." \
                    .format(df_size_mb, len(df), sample_rows), config.verbose)
            _, df_drop = train_test_split(df, train_size=sample_rows, random_state=1)
            df.drop(df_drop.index, inplace=True)

            config["nrows"] = sample_rows
        elif config["nrows_stage_nb"]==0:
            config["nrows"] = max(sample_rows, len(df))
        else:
            config["nrows"] = min(sample_rows, config["nrows"])

        config["nrows_stage_nb"] = config["stage_nb"]

@timeit
def rename_id_columns(df: pd.DataFrame, config: Config):
    if "id_columns" not in config:
        config["id_columns"] = dict([(c, 'string_'+c) for c in df if c.startswith("id_")])
        log("Id columns: " + ", ".join(config["id_columns"]), config.verbose)
    if len(config["id_columns"]) > 0:
        df.rename(columns=config["id_columns"], inplace=True)

@timeit
def non_negative_target_detect(df: pd.DataFrame, config: Config):
    if config.is_train():
        config["non_negative_target"] = df["target"].lt(0).sum() == 0


@timeit
def feature_selection(df: pd.DataFrame, config: Config):
    if config.is_train():
        df_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if df_size_mb < 2 * 1024:
            return

        selected_columns = []
        config_sample = copy.deepcopy(config)
        for i in range(10):
            df_sample = df.sample(min(1000, len(df)), random_state=i).copy(deep=True)
            # preprocess_pipeline(df_sample, config_sample)
            pipeline(df_sample, config_sample)
            y = df_sample["target"]
            X = df_sample.drop("target", axis=1)

            if len(selected_columns) > 0:
                X = X.drop(selected_columns, axis=1)

            if len(X.columns) > 0:
                selected_columns += select_features(X, y, config["mode"])
            else:
                break

        log("Selected columns: {}".format(selected_columns), config.verbose)

        drop_number_columns = [c for c in df if (c.startswith("number_") or c.startswith("id_")) and c not in selected_columns]
        if len(drop_number_columns) > 0:
            config["drop_number_columns"] = drop_number_columns

        config["date_columns"] = {}
        for c in [c for c in selected_columns if c.startswith("datetime_")]:
            d = c.split("_")
            date_col = d[0] + "_" + d[1]
            date_part = d[2]

            if date_col not in config["date_columns"]:
                config["date_columns"][date_col] = []

            config["date_columns"][date_col].append(date_part)

        drop_datetime_columns = [c for c in df if c.startswith("datetime_") and c not in config["date_columns"]]
        if len(drop_datetime_columns) > 0:
            config["drop_datetime_columns"] = drop_datetime_columns

    if "drop_number_columns" in config:
        log("Drop number columns: {}".format(config["drop_number_columns"]), config.verbose)
        df.drop(config["drop_number_columns"], axis=1, inplace=True)

    if "drop_datetime_columns" in config:
        log("Drop datetime columns: {}".format(config["drop_datetime_columns"]), config.verbose)
        df.drop(config["drop_datetime_columns"], axis=1, inplace=True)


@timeit
# https://github.com/bagxi/sdsj2018_lightgbm_baseline
# https://forum-sdsj.datasouls.com/t/topic/304/3
def leak_detect(df: pd.DataFrame, config: Config) -> bool:
    if config.is_predict():
        return "leak" in config

    id_cols = [c for c in df if c.startswith('id_')]
    dt_cols = [c for c in df if c.startswith('datetime_')]

    if id_cols and dt_cols:
        num_cols = [c for c in df if c.startswith('number_')]
        for id_col in id_cols:
            group = df.groupby(by=id_col).get_group(df[id_col].iloc[0])

            for dt_col in dt_cols:
                sorted_group = group.sort_values(dt_col)

                for lag in range(-1, -10, -1):
                    for col in num_cols:
                        corr = sorted_group['target'].corr(sorted_group[col].shift(lag))
                        if corr >= 0.99:
                            config["leak"] = {
                                "num_col": col,
                                "lag": lag,
                                "id_col": id_col,
                                "dt_col": dt_col,
                            }
                            return True

    return False


@timeit
def working_days_zero_detect(df: pd.DataFrame, config: Config) -> bool:
    if config.is_train() & ("is_working_datetime_0" in df):
        if (df.loc[df["is_working_datetime_0"]==0,"target"]==0).all():
            config["working_days_zero"]=True
            log("Working days zero detect", config.verbose)

        


@timeit
def feature_generation(df: pd.DataFrame, config: Config):
    # warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    feature_columns = [c for c in df if c.startswith("number_") 
                            # and df[c].dtype != np.int8 
                            and c not in config["categorical_columns"]]

    if len(feature_columns) > 0:
        df[feature_columns] += 1



class _Node(object):

    """
    Parameters
    ----------
    function : callable
        A function with signature function(df: pd.DataFrame, config: Config, Optional[params: dict]).

    name : str
        The name for the node.

    """

    def __init__(self, function, name):
        self.function = function
        self.name = name

    def __call__(self, *args):
        return self.function(*args)


def make_node(function: object, name: str):
    """Make a function node.

    This factory function creates a function node.

    Parameters
    ----------
    function : callable
        A function with signature `function(df: pd.DataFrame, config: Config)`.

    name : str
        The name for the node.

    """

    return _Node(function, name)



_node_map = { k : make_node(function=v, name=k) for k, v in [
                # ('read_df', read_df),
                ('drop_columns', drop_columns),
                ('check_columns_exists', check_columns_exists),
                ('fillna', fillna),
                ('to_int8', to_int8),
                ('non_negative_target_detect', non_negative_target_detect),
                ('transform_datetime', transform_datetime),
                ('transform_categorical', transform_categorical),
                ('scale', scale),
                ('feature_generation', feature_generation),
                ('subsample', subsample),
                ('columns_float64_to_32', columns_float64_to_32),
                ('split_X_y', split_X_y),
                ('model', model),
                # ('time_series_detect', time_series_detect),
                # ('working_days_zero_detect', working_days_zero_detect),
                ('rename_id_columns', rename_id_columns),
                # ('', ),
                # ('', ),
                ] 

            }
