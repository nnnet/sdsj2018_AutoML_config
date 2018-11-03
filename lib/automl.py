import os
import pandas as pd
import numpy as np
from lib.util import timeit, Config
from lib.read import read_df
from lib.model import validate
from typing import Optional

from graphviz import Digraph
from lib.nodes import pipeline, stage_time_inc
import time

class AutoML:
    def __init__(self, model_dir: str, params: dict, verbose: int=0):
        self.config = Config(model_dir, params, verbose=verbose)
        self.verbose=verbose

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
        self.config['stages_time'] = {}
        self.config.tmp_dir = os.path.join(self.config.model_dir, "tmp")
        os.makedirs(self.config.tmp_dir, exist_ok=True)

        start_time = time.time()
        df = read_df(train_csv, self.config)
        stage_time_inc(self.config, start_time, 'train read_df')
        
        pipeline(df, self.config)
        
        if self.config.verbose:
            self.stages_time_print()
    
    @timeit
    def predict(self, test_csv: str, prediction_csv: str) -> (pd.DataFrame, Optional[np.float64]):

        self.config["task"] = "predict"
        self.config.tmp_dir = os.path.join(os.path.dirname(prediction_csv), "tmp")
        self.config['stages_time'] = {}
        os.makedirs(self.config.tmp_dir, exist_ok=True)

        result = {
            "line_id": [],
            "prediction": [],
        }

        start_time = time.time()
        
        for X in pd.read_csv(
                test_csv,
                encoding="utf-8",
                low_memory=False,
                dtype=self.config["dtype"],
                parse_dates=self.config["parse_dates"],
                chunksize=self.config["nrows"]
           ):
            stage_time_inc(self.config, start_time, 'test pd.read_csv')
            result["line_id"] += list(X["line_id"])
            pipeline(X, self.config)
            result["prediction"] += list(X[self.config['graph'][-1][0]])
            start_time = time.time()

        result = pd.DataFrame(result)
        result.to_csv(prediction_csv, index=False)
        stage_time_inc(self.config, start_time, 'result.to_csv')
        
        target_csv = test_csv.replace("test", "test-target")
        if os.path.exists(target_csv):
            start_time = time.time()
            score = validate(result, target_csv, self.config["mode"], self.config.verbose)
            stage_time_inc(self.config, start_time, 'validate')
        else:
            score = None

        if self.config.verbose:
            self.stages_time_print()

        return result, score

    def stages_time_print(self, sort_by_time=True):
        if 'stages_time' in self.config.data.keys():
            d = self.config['stages_time']
            print('\n','-'*3, 'Pipeline stages time, sec:','-'*3)
            l_just = max([len(x) for x in d.keys()]) + 4
            if sort_by_time:
                for k, v in [(k, d[k]) for k in sorted(d, key=d.get, reverse=True)]:
                    print(k.replace('\n', '_').ljust(l_just), '{:<10} {:.2f}'.format(' ', v))
            else:
                for k, v in self.config['stages_time'].items():
                    print(k.replace('\n', '_').ljust(l_just), '{:<10} {:.2f}'.format(' ', v))
            print('-'*34, '\n')


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
        self.config.verbose = self.verbose
