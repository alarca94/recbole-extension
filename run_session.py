import argparse
import time

import numpy as np
import yaml

from custom_hyper import objective_function, get_ho_algo, build_space
from custom_utils import get_available_gpu_memory
from recbole.trainer import HyperTuning


def session_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='GRU4Rec', help='Model for session-based rec.')
    parser.add_argument('--dataset', '-d', type=str, default='mi-diginetica-session',
                        help='Benchmarks for session-based rec.')

    args, _ = parser.parse_known_args()
    args.__setattr__('params_file', f'./config/model/{args.model}.hyper')
    args.__setattr__('config_files', f'./config/dataset/{args.dataset}.yaml')

    with open(args.params_file, 'r') as f:
        params_dict = yaml.safe_load(f)

    space = build_space(params_dict, model=args.model, dataset=args.dataset)
    space['gpu_id'] = np.argmin(get_available_gpu_memory())
    ho_algo = get_ho_algo('exhaustive')

    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    hyper = HyperTuning(objective_function, algo=ho_algo, space=space, fixed_config_file_list=config_file_list)
    hyper.run()
    hyper.export_result(output_file=f'./results/{args.model}_{time.strftime("%d-%m-%Y", time.localtime())}.result')
    print('best params: ', hyper.best_params)
    print('best result: ')
    print(hyper.params2result[hyper.params2str(hyper.best_params)])


def prepare_dataset():
    import pandas as pd
    import numpy as np
    import os

    dataset = 'Tmall-session'
    trn_val_ratio = 0.9
    if os.path.isfile(os.path.join('./dataset', dataset, f'{dataset}.train_all.inter')):
        trn_data = pd.read_csv(os.path.join('./dataset', dataset, f'{dataset}.train_all.inter'), delimiter='\t')
    else:
        trn_data = pd.read_csv(os.path.join('./dataset', dataset, f'{dataset}.train.inter'), delimiter='\t')
        trn_data.to_csv(os.path.join('./dataset', dataset, f'{dataset}.train_all.inter'), index=False, sep='\t')
    # tst_data = pd.read_csv(os.path.join('./dataset', dataset, f'{dataset}.test.inter'), delimiter='\t')
    print(f'Number of training sessions: {trn_data.shape[0]}')
    # print(f'Number of test sessions: {tst_data.shape[0]}')
    # print(f'Total number of sessions: {trn_data.shape[0] + tst_data.shape[0]}')
    # print()

    n_trn_samples = int(trn_val_ratio * trn_data.shape[0])
    shuffled_idxs = np.random.permutation(np.arange(trn_data.shape[0], dtype=int))
    trn_trn_data = trn_data.iloc[shuffled_idxs[:n_trn_samples]]
    trn_trn_data.to_csv(os.path.join('./dataset', dataset, f'{dataset}.train.inter'), index=False, sep='\t')
    val_data = trn_data.iloc[shuffled_idxs[n_trn_samples:]]
    val_data.to_csv(os.path.join('./dataset', dataset, f'{dataset}.val.inter'), index=False, sep='\t')

    print('Done')


if __name__ == '__main__':
    # prepare_dataset()
    session_test()
