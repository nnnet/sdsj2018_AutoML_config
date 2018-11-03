import os
import time
import pickle
from typing import Any

from copy import deepcopy
import numpy as np

nesting_level = 0
is_start = None

def timeit(method):
    def timed(*args, **kw):
        global is_start
        global nesting_level

        # if not is_start:
        #     print()

        is_start = True
        verbose = args[0].config.verbose if hasattr(args[0], '__class__') and '.AutoML' in args[0].__str__() else 0

        log("Start {}.".format(method.__name__), verbose)
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        
        nesting_level -= 1
        log("End {}. Time: {:0.2f} sec.".format(method.__name__, end_time - start_time), verbose)
        is_start = False
        
        return result

    return timed


def log(entry: Any, verbose):
    global nesting_level
    if verbose!=0:
        space = "." * (4 * nesting_level)
        print("{}{}".format(space, entry))


def get_mode_from_file_name(file_name: str):
    return 'regression' if file_name.split(os.path.sep)[-2][-1]=='r' else 'classification'


class Config:
    def __init__(self, model_dir: str, params: dict, verbose:int=0):

        self.verbose = verbose
        self.model_dir = model_dir
        self.tmp_dir = self.model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.data = {
            "start_time": time.time(),
            "time_limit": params["TIME_LIMIT"] 
                            if "TIME_LIMIT" in params.keys()
                            else int(os.environ.get("TIME_LIMIT", 5 * 60)),
            }

        self.data['params'] = deepcopy(params)
        if 'pipeline' in params:
            self.data['graph'] = self.pipeline_prepare(params['pipeline'])

    def is_train(self) -> bool:
        return self["task"] == "train"

    def is_predict(self) -> bool:
        return self["task"] == "predict"

    def is_regression(self) -> bool:
        return self["mode"] == "regression"

    def is_classification(self) -> bool:
        return self["mode"] == "classification"

    def time_left(self):
        return self["time_limit"] - (time.time() - self["start_time"])

    def save(self):
        with open(os.path.join(self.model_dir, "config.pkl"), "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(os.path.join(self.model_dir, "config.pkl"), "rb") as f:
            data = pickle.load(f)
            self.data = {**self.data, **data}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)

    def pipeline_prepare(self, params):

        output=[]
        for k, v in params.items():
            if self.verbose>2: print(k)
            if not 'parents' in v \
                or v['parents'] is None:
            # add node "Start"
                output.append(['Start', k])
    
            elif isinstance(v['parents'], str):
                output.append([v['parents'], k])
    
            elif isinstance(v['parents'], (tuple, list)):
                for p in v['parents']:
                    output.append([p, k])
            else:
                raise ValueError('Unknow type of parameter "parents" for key "{0}"/n{1}' \
                                 .format(k, type(v) ) )
        
        lst_parents = [x[0] for x in output]
        dct_childs_cnt = {i:lst_parents.count(i) for i in lst_parents}
        if self.verbose>1: print(dct_childs_cnt)
        
        # add node "End"
        for p in set(np.unique([x[1] for x in output])) - set(np.unique(lst_parents)):
            output.append([str(p), 'End'])
    
        def get_childs_sequence(output, output_, parent_node, verbose=False):
            if parent_node=='End':
                return output, output_
    
            order_nb = 0
            while len(output)!=0:
                if output_[-1][1]=='End':
                    break
    
                order_nb +=1 
                parent_node = output_[-1][1]
                if verbose>2: print('parent_node :', parent_node)
                
                lst_childs = [x for x in output if x[0]==parent_node]
        
                if len(lst_childs)==0:
                    output_[-1] += [1]
                    break
                else:
                    output_[-1] += [len(lst_childs)]
    
                    for i, node in enumerate(lst_childs):
    
                        output, output0_ = get_childs_sequence(output, [node], parent_node)
                        if node in output:
                            output.remove(node)
                        output_ += output0_
                        if output_[-1][1]=='End':
                            if len(output_[-1])==2:
                                output_[-1] += [1]
                            if i == len(lst_childs)-1:
                                break
    
                if len(output)==0:
                    return output, output_
    
                if order_nb > 10000:
                    raise ValueError('Error in pipeline config')
    
            return output, output_
    
        output_=[x for x in output if x[0]=='Start']
        if len(output_) != 1:
            raise ValueError('More one nodes "Start"')
        output.remove(output_[-1])
    
        output, output_ = get_childs_sequence(output, output_, parent_node=output_[-1][1])
        
        if self.verbose>1: print(output_)
    
        return output_
    
