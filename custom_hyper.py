import logging

from custom_trainers import CustomTrainer
from recbole.config import Config
from recbole.data import data_preparation, create_dataset
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


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # Check whether the loss function is consistent with the negative sampling configuration
    if config_dict['loss_type'] == 'BPR' and config_dict.get('neg_sampling', None) is None:
        config_dict['neg_sampling'] = {'uniform': 1}

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = CustomTrainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }