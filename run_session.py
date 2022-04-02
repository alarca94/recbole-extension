import argparse
import time
import os
import yaml
import json

from custom_hyper import objective_function, get_ho_algo, build_space, tests_function
from custom_utils import prepare_dataset, view_results
from custom_constants import *
from recbole.trainer import HyperTuning


def session_hyperopt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='GRU4Rec', help='Model for session-based rec.')
    parser.add_argument('--dataset', '-d', type=str, default='diginetica-sample', help='Benchmarks for session-based rec.')
    parser.add_argument('--test', type=bool, default=False, help='Whether it is a test or real experiment.')
    parser.add_argument('--ho_max_evals', type=int, default=60, help='Maximum number of hyperparameter evaluations')
    parser.add_argument('--ho_algo', type=str, default='tpe', help='Hyper-optimization algorithm')

    args, _ = parser.parse_known_args()
    args.__setattr__('params_file', f'./config/model/{args.model}.hyper')
    args.__setattr__('config_files', f'./config/data/{args.dataset}.yaml'.replace('-sample', ''))

    # Load global hyperparameters to be optimized on all models
    with open(f'./config/model/global.hyper', 'r') as f:
        params_dict = yaml.safe_load(f)

    with open(args.params_file, 'r') as f:
        model_ho_config = yaml.safe_load(f)
        if model_ho_config:
            for k, v in model_ho_config.items():
                if k in params_dict:
                    params_dict[k].update(v)
                else:
                    params_dict[k] = v

    with open(f'./config/data/{args.dataset}.yaml', 'r') as f:
        data_config = yaml.safe_load(f)

    space = build_space(params_dict, model=args.model, dataset=args.dataset)
    ho_algo = get_ho_algo(args.ho_algo)

    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    hyper = HyperTuning(objective_function if not args.test else tests_function,
                        algo=ho_algo, space=space, fixed_config_file_list=config_file_list, max_evals=args.ho_max_evals)
    hyper.run()

    if not os.path.exists(os.path.join(PREFIX_PATH, RESULTS_PATH, data_config['split_type'])):
        os.makedirs(os.path.join(PREFIX_PATH, RESULTS_PATH, data_config['split_type']))
    if not os.path.exists(os.path.join(PREFIX_PATH, RESULTS_PATH, data_config['split_type'], args.dataset)):
        os.makedirs(os.path.join(PREFIX_PATH, RESULTS_PATH, data_config['split_type'], args.dataset))

    results_file = os.path.join(PREFIX_PATH, RESULTS_PATH, data_config['split_type'], args.dataset,
                                f'{args.model}_{time.strftime("%d-%m-%Y", time.localtime())}.result')

    if not args.test:
        hyper.export_result(output_file=results_file)

    print('best params: ', hyper.best_params)
    print('best result: ')
    print(hyper.params2result[hyper.params2str(hyper.best_params)])

    # Retrain on the whole training data and test (average test results of all slices)
    if args.test:
        test_results = tests_function(hyper.best_params.copy(), config_file_list, is_val=False)['test_result']
    else:
        test_results = objective_function(hyper.best_params.copy(), config_file_list, is_val=False)['test_result']

    if not args.test:
        with open(results_file, 'a') as f:
            # best_params = {k: v for k, v in hyper.best_params.items() if k not in ['checkpoint_dir', 'data_path']}
            f.write(f'Best Hyperparameters from optimization: {hyper.best_params}\n')
            f.write(f'Best valid result: {hyper.params2result[hyper.params2str(hyper.best_params)]["best_valid_result"]}\n')
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
    session_hyperopt()
    # custom_tests.test_hyperaugmentation()
