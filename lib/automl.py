import os
import pandas as pd
import numpy as np
from lib.util import timeit, Config
from lib.read import read_df
# from lib.preprocess import preprocess
from lib.model import validate # train, predict, 
from typing import Optional

from graphviz import Digraph
# from lib.nodes import _node_map
from lib.nodes import pipeline

class AutoML:
    def __init__(self, model_dir: str, params: dict, verbose: bool=False):
        self.config = Config(model_dir, params)

        if not 'memory' in self.config['params']:
            self.config['params']['memory'] = {}
        if not 'max_size_mb' in self.config['params']['memory']:
            self.config['params']['memory']['max_size_mb'] = 2
        if not 'max_size_train_samples' in self.config['params']['memory']:
            self.config['params']['memory']['max_size_train_samples'] = 10000
        if not 'field_target_name' in self.config['params']:
            self.config['params']['field_target_name'] = 'target'

    @timeit
    def train(self, train_csv: str, mode: str):
        self.config["task"] = "train"
        self.config["mode"] = mode
        self.config.tmp_dir = os.path.join(self.config.model_dir, "tmp")
        os.makedirs(self.config.tmp_dir, exist_ok=True)

        df = read_df(train_csv, self.config)
        pipeline(df, self.config)
    
    @timeit
    def predict(self, test_csv: str, prediction_csv: str) -> (pd.DataFrame, Optional[np.float64]):

        self.config["task"] = "predict"
        self.config.tmp_dir = os.path.join(os.path.dirname(prediction_csv), "tmp")
        os.makedirs(self.config.tmp_dir, exist_ok=True)

        result = {
            "line_id": [],
            "prediction": [],
        }

        for X in pd.read_csv(
                test_csv,
                encoding="utf-8",
                low_memory=False,
                dtype=self.config["dtype"],
                parse_dates=self.config["parse_dates"],
                chunksize=self.config["nrows"]
        ):
            result["line_id"] += list(X["line_id"])
            # preprocess(X, self.config)
            # result["prediction"] += list(predict(X, self.config))
            pipeline(X, self.config)
            result["prediction"] += list(X[self.config['graph'][-1][0]])

        result = pd.DataFrame(result)
        result.to_csv(prediction_csv, index=False)

        target_csv = test_csv.replace("test", "test-target")
        if os.path.exists(target_csv):
            score = validate(result, target_csv, self.config["mode"])
        else:
            score = None

        return result, score


    def pipeline_draw(self, file_name='AutoML_pipeline.gv', view=False):
        g = Digraph('G', filename=file_name)
        for i in self.config['graph']:
            g.edge(i[0], i[1])
    
        if view:
            g.view()
    
        return g

    @timeit
    def save(self):
        self.config.save()

    @timeit
    def load(self):
        self.config.load()
