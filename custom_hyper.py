import logging
import os
import re
import numpy as np
from hyperopt.pyll import scope

from custom_constants import *
from custom_trainers import CustomTrainer
from custom_utils import get_available_gpu_memory
from recbole.config import Config
from recbole.data import data_preparation, create_dataset, get_dataloader, create_samplers
from recbole.utils import get_model, init_seed
from hyperopt import hp, tpe, atpe, mix, rand, anneal

HYPER_OPT_ALGS = {
    "tpe": tpe.suggest,
    "atpe": atpe.suggest,
    "mix": mix.suggest,
    "rand": rand.suggest,
    "anneal": anneal.suggest
}


def build_space(config_dict, model, dataset):
    space = {'model': model, 'dataset': dataset}
    for para_type in config_dict:
        if para_type == 'choice':
            for para_name in config_dict['choice']:
                para_value = config_dict['choice'][para_name]
                space[para_name] = hp.choice(para_name, para_value)
        elif para_type == 'list_choice':
            for para_name in config_dict['list_choice']:
                para_value = [str(v) for v in config_dict['list_choice'][para_name]]
                space[para_name] = hp.choice(para_name, para_value)
        elif para_type == 'uniform':
            for para_name in config_dict['uniform']:
                para_value = config_dict['uniform'][para_name]
                low = para_value[0]
                high = para_value[1]
                space[para_name] = hp.uniform(para_name, float(low), float(high))
        elif para_type == 'quniform':
            for para_name in config_dict['quniform']:
                para_value = config_dict['quniform'][para_name]
                low = para_value[0]
                high = para_value[1]
                q = para_value[2]
                space[para_name] = hp.quniform(para_name, float(low), float(high), float(q))
        elif para_type == 'intquniform':
            for para_name in config_dict['intquniform']:
                para_value = config_dict['intquniform'][para_name]
                low = para_value[0]
                high = para_value[1]
                q = para_value[2]
                space[para_name] = scope.int(hp.quniform(para_name, int(low), int(high), int(q)))
        elif para_type == 'loguniform':
            for para_name in config_dict['loguniform']:
                para_value = config_dict['loguniform'][para_name]
                low = para_value[0]
                high = para_value[1]
                space[para_name] = hp.loguniform(para_name, float(low), float(high))
        else:
            raise ValueError('Illegal param type [{}]'.format(para_type))
    return space


def get_ho_algo(algo='exhaustive'):
    return HYPER_OPT_ALGS.get(algo, algo)


def objective_function(config_dict=None, config_file_list=None, verbose=False, show_progress=False, is_val=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        verbose (bool, optional): Whether to print single iteration progress. Defaults to ``False``.
        show_progress (bool, optional): Whether to show trainer.fit() progress or not.
        is_val (bool, optional): Whether the objective function is used for validation or testing
    """
    config_dict['gpu_id'] = np.argmin(get_available_gpu_memory())
    config_dict['checkpoint_dir'] = os.path.join(PREFIX_PATH, CHECKPOINT_PATH)

    # Check whether the loss function is consistent with the negative sampling configuration
    if config_dict['loss_type'] == 'BPR' and config_dict.get('neg_sampling', None) is None:
        config_dict['neg_sampling'] = {'uniform': 1}

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    config['data_path'] = os.path.join(PREFIX_PATH, DATASET_PATH, config['split_type'], config['dataset'])

    if 'slices' in config['data_path']:
        n_slices = max([int(re.sub('.*\.(\d+)\..*', '\g<1>', s)) for s in os.listdir(config['data_path'])]) + 1

        max_result_cols = ['max_gpu_usage']
        avg_results = dict()
        for split in range(n_slices):
            if is_val:
                config['benchmark_filename'] = [f'train.{split}', f'val.{split}']
            else:
                config['benchmark_filename'] = [f'train_all.{split}', f'test.{split}']

            result = run_single(config, verbose, show_progress, is_val)
            for k in result:
                if k in max_result_cols:
                    avg_results[k] = max(result[k], avg_results.get(k, 0))
                else:
                    avg_results[k] = result[k] + avg_results.get(k, 0)

        for k in avg_results:
            if k not in max_result_cols:
                avg_results[k] /= n_slices

        if is_val:
            return {
                'best_valid_score': avg_results[config['valid_metric'].lower()],
                'valid_score_bigger': config['valid_metric_bigger'],
                'best_valid_result': avg_results,
                'test_result': dict()
            }

        else:
            return {
                'best_valid_score': None,
                'valid_score_bigger': None,
                'best_valid_result': dict(),
                'test_result': avg_results
            }

    elif 'single' in config['data_path']:
        raise NotImplementedError('Only "slices" mode is implemented in this objective_function')

    raise NotImplementedError('Only "single" or "slices" modes are implemented in this objective_function')


def run_single(config, verbose, show_progress, is_val):
    dataset = create_dataset(config)

    # Cut sequences with max_sequence_length
    if dataset.max_item_list_len is not None:
        if dataset.inter_feat.item_length.max() > dataset.max_item_list_len:
            cut_sequences = (lambda s: s if len(s) <= dataset.max_item_list_len else s[-dataset.max_item_list_len:])
            dataset.inter_feat.item_id_list = dataset.inter_feat.item_id_list.apply(cut_sequences)
            dataset.field2seqlen['item_id_list'] = dataset.max_item_list_len
            dataset.inter_feat.item_length = np.minimum(dataset.inter_feat.item_length, dataset.max_item_list_len)

    built_datasets = dataset.build()

    trn_sampler, val_sampler, tst_sampler = create_samplers(config, dataset, built_datasets)

    trn_data, tst_data = built_datasets
    trn_loader = get_dataloader(config, 'train')(config, trn_data, trn_sampler, shuffle=True)

    if is_val:
        tst_loader = get_dataloader(config, 'evaluation')(config, tst_data, val_sampler, shuffle=False)
    else:
        tst_loader = get_dataloader(config, 'evaluation')(config, tst_data, tst_sampler, shuffle=False)

    model = get_model(config['model'])(config, trn_loader.dataset).to(config['device'])
    trainer = CustomTrainer(config, model)

    _, _ = trainer.fit(trn_loader, None, verbose=verbose, saved=False, show_progress=show_progress)
    return trainer.evaluate(tst_loader, load_best_model=False)


def tests_function(config_dict=None, config_file_list=None, verbose=False, show_progress=False, is_val=True):
    r""" The test objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        verbose (bool, optional): Whether to print single iteration progress. Defaults to ``False``.
        show_progress (bool, optional): Whether to show trainer.fit() progress or not.
        is_val (bool, optional): Whether the objective function is used for validation or testing
    """
    # Check whether the loss function is consistent with the negative sampling configuration
    if config_dict['loss_type'] == 'BPR' and config_dict.get('neg_sampling', None) is None:
        config_dict['neg_sampling'] = {'uniform': 1}

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)

    if 'slices' in config['data_path']:
        n_slices = 1

        max_result_cols = ['max_gpu_usage']
        avg_results = dict()
        for split in range(n_slices):
            if is_val:
                config['benchmark_filename'] = [f'train.{split}', f'val.{split}']
            else:
                config['benchmark_filename'] = [f'train_all.{split}', f'test.{split}']

            result = run_single(config, verbose, show_progress, is_val)
            for k in result:
                if k in max_result_cols:
                    avg_results[k] = max(result[k], avg_results.get(k, 0))
                else:
                    avg_results[k] = result[k] + avg_results.get(k, 0)

        for k in avg_results:
            if k not in max_result_cols:
                avg_results[k] /= n_slices

        if is_val:
            return {
                'best_valid_score': avg_results[config['valid_metric'].lower()],
                'valid_score_bigger': config['valid_metric_bigger'],
                'best_valid_result': avg_results,
                'test_result': dict()
            }

        else:
            return {
                'best_valid_score': None,
                'valid_score_bigger': None,
                'best_valid_result': dict(),
                'test_result': avg_results
            }

    elif 'single' in config['data_path']:
        raise NotImplementedError('Only "slices" mode is implemented in this objective_function')

    raise NotImplementedError('Only "single" or "slices" modes are implemented in this objective_function')
