import argparse
import time
import custom_tests
import os

import numpy as np
import yaml
import json
import torch

from custom_hyper import objective_function, get_ho_algo, build_space
from custom_utils import get_available_gpu_memory, prepare_dataset, view_results
from custom_constants import *
from recbole.trainer import HyperTuning


def session_hyper():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='GRU4Rec', help='Model for session-based rec.')
    parser.add_argument('--dataset', '-d', type=str, default='diginetica-sample', help='Benchmarks for session-based rec.')
    parser.add_argument('--split_type', type=str, default='slices', help='Either single split or slices.')

    args, _ = parser.parse_known_args()
    args.__setattr__('params_file', f'./config/model/{args.model}.hyper')
    args.__setattr__('config_files', f'./config/data/{args.dataset}.yaml'.replace('-sample', ''))

    with open(args.params_file, 'r') as f:
        params_dict = yaml.safe_load(f)

    params_dict['intquniform'] = {'epochs': [5, 30, 1]}
    space = build_space(params_dict, model=args.model, dataset=args.dataset)
    space['gpu_id'] = np.argmin(get_available_gpu_memory())
    space['checkpoint_dir'] = os.path.join(PREFIX_PATH, CHECKPOINT_PATH)
    space['data_path'] = os.path.join(PREFIX_PATH, DATASET_PATH, args.split_type)
    ho_algo = get_ho_algo('exhaustive')

    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    hyper = HyperTuning(objective_function, algo=ho_algo, space=space, fixed_config_file_list=config_file_list)
    hyper.run()

    if not os.path.exists(os.path.join(PREFIX_PATH, RESULTS_PATH, args.split_type)):
        os.makedirs(os.path.join(PREFIX_PATH, RESULTS_PATH, args.split_type))
    if not os.path.exists(os.path.join(PREFIX_PATH, RESULTS_PATH, args.split_type, args.dataset)):
        os.makedirs(os.path.join(PREFIX_PATH, RESULTS_PATH, args.split_type, args.dataset))

    results_file = os.path.join(PREFIX_PATH, RESULTS_PATH, args.split_type, args.dataset,
                                f'{args.model}_{time.strftime("%d-%m-%Y", time.localtime())}.result')

    hyper.export_result(output_file=results_file)
    print('best params: ', hyper.best_params)
    print('best result: ')
    print(hyper.params2result[hyper.params2str(hyper.best_params)])

    # Retrain on the whole training data and test (average test results of all slices)
    test_results = objective_function(hyper.best_params, config_file_list, is_val=False)['test_result']
    for k in ['checkpoint_dir', 'data_path']:
        hyper.best_params.pop(k, None)
    with open(results_file, 'a') as f:
        f.write(f'Best Hyperparameters from optimization: {hyper.best_params}\n')
        f.write(f'Final test result: {test_results}\n')

    print('Test results: ')
    print(test_results)



def session_single():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='NARM', help='Model for session-based rec.')
    parser.add_argument('--dataset', '-d', type=str, default='nowplaying',
                        help='Benchmarks for session-based rec.')
    args = parser.parse_known_args()[0]
    args.__setattr__('config_files', [f'./config/data/{args.dataset}.yaml'])
    with open(f'./recbole/properties/model/{args.model}.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    config_dict['model'] = args.model
    config_dict['data'] = args.dataset
    config_dict['checkpoint_dir'] = os.path.join(PREFIX_PATH, CHECKPOINT_PATH)
    config_dict['data_path'] = os.path.join(PREFIX_PATH, DATASET_PATH)
    # config_dict['gpu_id'] = 2  # torch.cuda.device_count() - 1
    results = objective_function(config_dict=config_dict, config_file_list=args.config_files, show_progress=True)
    print(json.dumps(results, indent=4, sort_keys=False))


if __name__ == '__main__':
    # prepare_dataset('diginetica', slices=True, sample=True)
    # custom_tests.scipy_vs_torch_sparse()
    # view_results()
    # session_single()
    # prepare_sample_dataset()
    session_hyper()
    # custom_tests.test_hyperaugmentation()
