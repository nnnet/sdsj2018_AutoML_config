import argparse
from lib.util import timeit, get_mode_from_file_name
from lib.automl import AutoML
# import os
import numpy as np

from sklearn.externals.joblib import cpu_count


def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--train-csv', type=argparse.FileType('r')
    #     , default="../sdsj2018_automl_check_datasets/check_1_r/train.csv")
    #     # , required=True)
    # parser.add_argument('--model-dir')

    parser.add_argument('--train-csv'
        , default="../../sdsj2018_automl_check_datasets/check_8_c/train.csv"
        )
    parser.add_argument('--test-csv'
        , default='../../sdsj2018_automl_check_datasets/check_8_c/test.csv'
                        )
    parser.add_argument('--prediction-csv'
        , default='../../sdsj2018_automl_check_datasets/check_8_c/prediction.csv'
        )
# # python main.py --mode regression --train-csv '../sdsj2018_automl_check_datasets/check_1_r/train.csv' --model-dir '../model/'
# # python main.py --test-csv '../sdsj2018_automl_check_datasets/check_1_r/test.csv' --prediction-csv '../sdsj2018_automl_check_datasets/check_1_r/prediction.csv' --model-dir '../model/'
    parser.add_argument('--model-dir'
        , default="../../model/")
        # , required=True)
    parser.add_argument('--mode', choices=['classification', 'regression']
        , default="classification")
        # , required=True)
        
    # parser.add_argument('--mode', choices=['classification', 'regression'])
    # parser.add_argument('--model-dir')
    # parser.add_argument('--train-csv')
    # parser.add_argument('--test-csv')
    # parser.add_argument('--prediction-csv')

    args = parser.parse_args()

    if not args.train_csv is None:
        params_autoML = \
                { 'memory': {'max_size_mb': 2*1024, # * 1024 * 1024,
                             'max_size_train_samples': 100000
                            },
                  'field_target_name': 'target',
        
                  'pipeline' : {
                        'check\ncolumns\n_exists_start': {'node':'check_columns_exists',
                              'parents':None,
                              'args':{'key_stage':'preprocess_begin', 'drop_columns_test':True}
                              },
                        'drop_columns': {'node':'drop_columns',
                             'parents':'check\ncolumns\n_exists_start', # 'args':{}
                             },
                        'fillna': {'node':'fillna',
                             'parents':'drop_columns', # 
                             'args':{'args':{"number_": {'agg': 'median'},
                                     "datetime_": {'agg': 'median'},
                                     "string_": {'value': ''},
                                    }},
                             },
                        'to_int8': {'node':'to_int8',
                             'parents':'fillna', # 'args':{}
                             },
                        'non_negative\ntarget_detect': {'node':'non_negative_target_detect',
                             'parents':'to_int8', # 'args':{}
                             },
                        'subsample0': {'node':'subsample',
                             'parents':'non_negative\ntarget_detect', # 'args':{}
                             },
                        'transform\ndatetime': {'node':'transform_datetime',
                             'parents':'subsample0', # 'args':{}
                             },
                        'transform\ncategorical': {'node':'transform_categorical',
                             'parents':'transform\ndatetime', # 'args':{}
                             },
                        'scale': {'node':'scale',
                             'parents':'transform\ncategorical', # 'args':{}
                             },
                        # 'feature_generation': {'node':'feature_generation',
                        #      'parents':'scale', # 'args':{}
                        #      },
                        'subsample1': {'node':'subsample',
                             'parents':'scale', # 'args':{}
                             },
                        'columns_float64\nto_float32': {'node':'columns_float64_to_32',
                             'parents':'subsample1', # 'args':{}
                             },
                        'check\ncolumns\nexists_end': {'node':'check_columns_exists',
                             'parents':'columns_float64\nto_float32',
                             'args':{'key_stage':'preprocess_end', 'drop_columns_test':True}
                             },
                        # # 'split_X_y': {'node':'split_X_y',
                        # #       'parents':'check\ncolumns\nexists_end', # 'args':{}
                        # #       },
                        'bayes': {'node':'model',
                              'parents':'check\ncolumns\nexists_end',
                              'args':{'models': ['bayes']}
                              },
                        'lightgbm\nend': {'node':'model',
                              'parents':'bayes', #'bayes',
                              'args':{'models': ['lightgbm']}
                              },
                        # '': {'node':'',
                        #      'parents':'', # 'args':{}
                        #      },
                        # '': {'node':'',
                        #      'parents':'', # 'args':{}
                        #      },
                        }
                }

        # print('args.train_csv', type(args.train_csv.name), args.train_csv.name)
        # args.train_csv = args.train_csv.name

        automl = AutoML(args.model_dir, params=params_autoML)
        # automl.pipeline_draw(view=True)
        automl.train(args.train_csv, args.mode)
        automl.save()
    elif args.test_csv is not None:
        
        automl = AutoML(args.model_dir, params={})
        automl.load()
        _, score = automl.predict(args.test_csv, args.prediction_csv)
        print('score', score)
    else:
        exit(1)

# %%
if __name__ == '__main__':
    main()
# # %%    
# params_autoML = \
#         { 'model_dir' : 'args.model_dir',
#           'memory': {'max_size_mb': 11, # * 1024 * 1024,
#                      'max_size_train_samples': 100000
#                     },
#           'field_target_name': 'target',

#           'pipeline' : {
#                 'check\ncolumns\nexists_start': {'node':'check_columns_exists',
#                       'parents':None,
#                       'args':{'key_stage':'preprocess_begin', 'drop_columns_test':True}
#                       },
#                 'drop\ncolumns': {'node':'drop_columns',
#                      'parents':'check\ncolumns\nexists_start', # 'args':{}
#                      },
#                 'fillna': {'node':'fillna',
#                      'parents':'drop\ncolumns', # 'args':{}
#                      },
#                 'to_int8': {'node':'to_int8',
#                      'parents':'fillna', # 'args':{}
#                      },
#                 'transform\ndatetime': {'node':'transform_datetime',
#                      'parents':'to_int8', # 'args':{}
#                      },
#                 'transform\ncategorical': {'node':'transform_categorical',
#                      'parents':'transform\ndatetime', # 'args':{}
#                      },
#                 'scale': {'node':'scale',
#                      'parents':'transform\ncategorical', # 'args':{}
#                      },
#                 'float64\nto_float32': {'node':'columns_float64_to_32',
#                      'parents':'scale', # 'args':{}
#                      },
#                 'subsample0': {'node':'subsample',
#                      'parents':'float64\nto_float32', # 'args':{}
#                      },
#                 'non_negative\ntarget_detect': {'node':'non_negative_target_detect',
#                      'parents':'subsample0', # 'args':{}
#                      },
#                 # 'feature_generation': {'node':'feature_generation',
#                 #      'parents':'scale', # 'args':{}
#                 #      },
#                 # 'subsample1': {'node':'subsample',
#                 #      'parents':'scale', # 'args':{}
#                 #      },
#                 'check\ncolumns\nexists_end': {'node':'check_columns_exists',
#                      'parents':'non_negative\ntarget_detect',
#                      'args':{'key_stage':'preprocess_end', 'drop_columns_test':True}
#                      },
#                 # 'split_X_y': {'node':'split_X_y',
#                 #       'parents':'check\ncolumns\nexists_end', # 'args':{}
#                 #       },
#                 'bayes': {'node':'model',
#                       'parents':'check\ncolumns\nexists_end',
#                       'args':{'models': ['bayes']}
#                       },
#                 'lightgbm\nend': {'node':'model',
#                       'parents':'bayes',
#                       'args':{'models': ['lightgbm']}
#                       },
#                 # '': {'node':'',
#                 #      'parents':'', # 'args':{}
#                 #      },
#                 # '': {'node':'',
#                 #      'parents':'', # 'args':{}
#                 #      },
#                 }
#         }

# params_autoML = \
#         { 'model_dir' : 'args.model_dir',
#           'memory': {'max_size_mb': 11, # * 1024 * 1024,
#                      'max_size_train_samples': 100000
#                     },
#           'field_target_name': 'target',

#           'pipeline' : {
#                 'check\ncolumns\nexists_start': {'node':'check_columns_exists',
#                       'parents':None,
#                       'args':{'key_stage':'preprocess_begin', 'drop_columns_test':True}
#                       },
#                 'drop\ncolumns': {'node':'drop_columns',
#                      'parents':'check\ncolumns\nexists_start', # 'args':{}
#                      },
#                 'fillna': {'node':'fillna',
#                      'parents':'drop\ncolumns', # 'args':{}
#                      },
#                 'to_int8': {'node':'to_int8',
#                      'parents':'fillna', # 'args':{}
#                      },
#                 'transform\ndatetime': {'node':'transform_datetime',
#                      'parents':'fillna', # 'args':{}
#                      },
#                 'transform\ncategorical': {'node':'transform_categorical',
#                      'parents':'fillna', # 'args':{}
#                      },
#                 'stack\ndata': {'node':'transform_categorical',
#                      'parents':['to_int8', 'transform\ndatetime', 'transform\ncategorical'], # 'args':{}
#                      },
#                 'scale': {'node':'scale',
#                      'parents':'scale', # 'args':{}
#                      },
#                 'float64\nto_float32': {'node':'columns_float64_to_32',
#                      'parents':'stack\ndata', # 'args':{}
#                      },
#                 'subsample0': {'node':'subsample',
#                      'parents':'float64\nto_float32', # 'args':{}
#                      },
#                 'non_negative\ntarget_detect': {'node':'non_negative_target_detect',
#                      'parents':'subsample0', # 'args':{}
#                      },
#                 # 'feature_generation': {'node':'feature_generation',
#                 #      'parents':'scale', # 'args':{}
#                 #      },
#                 # 'subsample1': {'node':'subsample',
#                 #      'parents':'scale', # 'args':{}
#                 #      },
#                 'check\ncolumns\nexists_end': {'node':'check_columns_exists',
#                      'parents':'non_negative\ntarget_detect',
#                      'args':{'key_stage':'preprocess_end', 'drop_columns_test':True}
#                      },
#                 # 'split_X_y': {'node':'split_X_y',
#                 #       'parents':'check\ncolumns\nexists_end', # 'args':{}
#                 #       },
#                 'bayes': {'node':'model',
#                       'parents':'check\ncolumns\nexists_end',
#                       'args':{'models': ['bayes']}
#                       },
#                 'lightgbm\nend': {'node':'model',
#                       'parents':'bayes',
#                       'args':{'models': ['lightgbm']}
#                       },
#                 # '': {'node':'',
#                 #      'parents':'', # 'args':{}
#                 #      },
#                 # '': {'node':'',
#                 #      'parents':'', # 'args':{}
#                 #      },
#                 }
#         }

# params_autoML = \
#         { 'model_dir' : 'args.model_dir',
#           'memory': {'max_size_mb': 11, # * 1024 * 1024,
#                      'max_size_train_samples': 100000
#                     },
#           'field_target_name': 'target',

#           'pipeline' : {
#                 'check\ncolumns\nexists_start': {'node':'check_columns_exists',
#                       'parents':None,
#                       'args':{'key_stage':'preprocess_begin', 'drop_columns_test':True}
#                       },
#                 'drop\ncolumns': {'node':'drop_columns',
#                      'parents':'check\ncolumns\nexists_start', # 'args':{}
#                      },
#                 'fillna': {'node':'fillna',
#                      'parents':'drop\ncolumns', # 'args':{}
#                      },
#                 'to_int8': {'node':'to_int8',
#                      'parents':'fillna', # 'args':{}
#                      },
#                 'transform\ndatetime': {'node':'transform_datetime',
#                      'parents':'fillna', # 'args':{}
#                      },
#                 'transform\ncategorical': {'node':'transform_categorical',
#                      'parents':'fillna', # 'args':{}
#                      },
#                 'stack\ndata': {'node':'transform_categorical',
#                      'parents':['to_int8', 'transform\ndatetime', 'transform\ncategorical'], # 'args':{}
#                      },
#                 'scale': {'node':'scale',
#                      'parents':'scale', # 'args':{}
#                      },
#                 'float64\nto_float32': {'node':'columns_float64_to_32',
#                      'parents':'stack\ndata', # 'args':{}
#                      },
#                 'subsample0': {'node':'subsample',
#                      'parents':'float64\nto_float32', # 'args':{}
#                      },
#                 'non_negative\ntarget_detect': {'node':'non_negative_target_detect',
#                      'parents':'subsample0', # 'args':{}
#                      },
#                 # 'feature_generation': {'node':'feature_generation',
#                 #      'parents':'scale', # 'args':{}
#                 #      },
#                 # 'subsample1': {'node':'subsample',
#                 #      'parents':'scale', # 'args':{}
#                 #      },
#                 'check\ncolumns\nexists_end': {'node':'check_columns_exists',
#                      'parents':'non_negative\ntarget_detect',
#                      'args':{'key_stage':'preprocess_end', 'drop_columns_test':True}
#                      },
#                 # 'split_X_y': {'node':'split_X_y',
#                 #       'parents':'check\ncolumns\nexists_end', # 'args':{}
#                 #       },
#                 'vw': {'node':'model',
#                       'parents':'check\ncolumns\nexists_end',
#                       'args':{'models': ['vw']}
#                       },
#                 'rf': {'node':'model',
#                       'parents':'check\ncolumns\nexists_end',
#                       'args':{'models': ['rf']}
#                       },
#                 'lcv': {'node':'model',
#                       'parents':'check\ncolumns\nexists_end',
#                       'args':{'models': ['lcv']}
#                       },
#                 'stack\ndata&models': {'node':'transform_categorical',
#                      'parents':['vw', 'rf', 'lcv', 'check\ncolumns\nexists_end'], # 'args':{}
#                      },
#                 'h2o\nend': {'node':'model',
#                       'parents':'stack\ndata&models',
#                       'args':{'models': ['h2o']}
#                       },
#                 # '': {'node':'',
#                 #      'parents':'', # 'args':{}
#                 #      },
#                 # '': {'node':'',
#                 #      'parents':'', # 'args':{}
#                 #      },
#                 }
#         }

                    
# automl = AutoML("../model/", params=params_autoML)

# # # %%
# automl.pipeline_draw()
# # # %%
# # automl.train("../sdsj2018_automl_check_datasets/check_1_r/train.csv", 'regression')

# # # automl.config.data
# # # automl.config
# # # dir(automl.config.params) #.pipeline
# # os.exit(0)
# # # %%
# # _, score = automl.predict('../sdsj2018_automl_check_datasets/check_1_r/test.csv', 
# #             '../sdsj2018_automl_check_datasets/check_1_r/prediction.csv')
# # print(score)

# # # python main.py --mode regression --train-csv '../sdsj2018_automl_check_datasets/check_1_r/train.csv' --model-dir '../model/'
# # # python main.py --test-csv '../sdsj2018_automl_check_datasets/check_1_r/test.csv' --prediction-csv '../sdsj2018_automl_check_datasets/check_1_r/prediction.csv' --model-dir '../model/'


# # 