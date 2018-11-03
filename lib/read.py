import pandas as pd
from lib.util import timeit, log, Config
import random
import numpy as np

@timeit
def read_df(csv_path: str, config: Config) -> pd.DataFrame:
    if "dtype" not in config:
        preview_df(csv_path, config)

    df = pandas_read_csv(csv_path, config)
    if config.is_train():
        config["nrows_stage_nb"] = 0
        config["nrows"] = len(df)

    return df


@timeit
def pandas_read_csv(csv_path: str, config: Config) -> pd.DataFrame:
    
    return pd.read_csv(csv_path, encoding="utf-8", 
                       low_memory=False, dtype=config["dtype"], 
                       parse_dates=config["parse_dates"],
                       skiprows=config['params']['memory']['skiprows']
                       )


@timeit
def preview_df(train_csv: str, config: Config, nrows: int=3000):
    num_rows = sum(1 for line in open(train_csv)) - 1
    log("Rows in train: {}".format(num_rows), config.verbose)

    df = pd.read_csv(train_csv, encoding="utf-8", low_memory=False, nrows=nrows)
    mem_per_row = df.memory_usage(deep=True).sum() / nrows
    log("Memory per row: {:0.2f} Kb".format(mem_per_row / 1024), config.verbose)

    df_size = num_rows * mem_per_row
    log("Approximate dataset size: {:0.2f} Mb".format(df_size / 1024 / 1024), config.verbose)

    config["parse_dates"] = []
    config["dtype"] = {
        "line_id": int,
    }

    counters = {
        "id": 0,
        "number": 0,
        "string": 0,
        "datetime": 0,
    }

    for c in df:
        if c.startswith("number_"):
            counters["number"] += 1
            config["dtype"][c] = 'float32' if str(df[c].dtype)=='float64'\
                                           else str(df[c].dtype)
        elif c.startswith("string_"):
            counters["string"] += 1
            config["dtype"][c] = str
        elif c.startswith("datetime_"):
            counters["datetime"] += 1
            config["dtype"][c] = str
            config["parse_dates"].append(c)
        elif c.startswith("id_"):
            counters["id"] += 1

    log("Number columns: {}".format(counters["number"]), config.verbose)
    log("String columns: {}".format(counters["string"]), config.verbose)
    log("Datetime columns: {}".format(counters["datetime"]), config.verbose)

    config["counters"] = counters


    df_y = pd.read_csv(train_csv, usecols=['target'])
    df_size = df_y.shape[0] * mem_per_row

    if df_size > config['params']['memory']['max_size_mb'] * 1024 * 1024\
        or ('max_size_train_samples' in config['params']['memory'] \
            and df_y.shape[0] > config['params']['memory']['max_size_train_samples']):
        
        if config["mode"] == "regression":
            y_median = df_y['target'].agg('median')
        else: # "binary"
            y_median = 0.5

        ids_low = df_y[df_y['target']<=y_median].index.tolist()
        ids_up = df_y[df_y['target']>y_median].index.tolist()

        rows_cnt_half = int(config['params']['memory']['max_size_mb'] * 1024 * 1024 / 2 / mem_per_row)
        rows_cnt_half = min(rows_cnt_half, len(ids_low), len(ids_up))

        random.shuffle(ids_low)
        ids_low = ids_low[:rows_cnt_half]
        random.shuffle(ids_up)
        ids_up = ids_up[:rows_cnt_half]

        config['params']['memory']['skiprows'] = \
            (np.array([x for x in set(df_y.index.tolist()) - set(ids_low) - set(ids_up)])+1).tolist()

    else:
        config['params']['memory']['skiprows'] = None
        
