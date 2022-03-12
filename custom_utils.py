import subprocess
import os
import time
import yaml

import pandas as pd
import numpy as np

from ast import literal_eval
from datetime import datetime, timezone, timedelta
from custom_constants import *

SESSION_COL = 'session_id:token'
ITEM_COL = 'item_id:token'
TIME_COL = 'timestamp:float'
ITEM_LIST_COL = 'item_id_list:token_seq'
TARGET_COL = 'item_id:token'


def get_available_gpu_memory():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory


def get_original_sessions(data):
    data['full_session'] = data['item_id_list:token_seq'] + ' ' + data['item_id:token'].astype(str)
    # We assume the augmented sessions appear consecutively in the dataset. In case sorting is necessary, please adapt
    # this method accordingly to account for possible duplicate sessions in the dataset.
    # data.sort_values(by='full_session', ascending=False, inplace=True)
    # Condition: no consecutive duplicates (full session record vs partial session of previous record)
    keep = [True] + (data['full_session'][1:] != data['item_id_list:token_seq'][:-1]).tolist()
    return data[keep]


def dataset_stats():
    dataset_full = 'diginetica'
    dataset = 'mi-diginetica-session'
    trn_data = pd.read_csv(os.path.join('./dataset', dataset, f'{dataset}.train.inter'), delimiter='\t')
    tst_data = pd.read_csv(os.path.join('./dataset', dataset, f'{dataset}.test.inter'), delimiter='\t')
    print(f'Number of training sessions: {trn_data.shape[0]}')
    print(f'Number of test sessions: {tst_data.shape[0]}')
    print(f'Total number of sessions: {trn_data.shape[0] + tst_data.shape[0]}')
    print()

    def get_sess_lengths(data):
        return data['item_id_list:token_seq'].apply(lambda s: len(s.split())+1)

    mean_trn_s_len = get_sess_lengths(trn_data)
    mean_tst_s_len = get_sess_lengths(tst_data)
    mean_s_len = pd.concat([mean_trn_s_len, mean_tst_s_len])
    print(f'Mean training session length: {mean_trn_s_len.mean()}')
    print(f'Mean test session length: {mean_tst_s_len.mean()}')
    print(f'Mean session length: {mean_s_len.mean()}')
    print()

    # Number of equal sessions in train and test --> 752 (3812 if we don't count target items)
    uni_trn_data = get_original_sessions(trn_data)
    uni_tst_data = get_original_sessions(tst_data)
    f_trn_sess = uni_trn_data['item_id_list:token_seq'] + ' ' + uni_trn_data['item_id:token'].astype(str)
    f_tst_sess = uni_tst_data['item_id_list:token_seq'] + ' ' + uni_tst_data['item_id:token'].astype(str)
    print(f'Number of unique training sessions: {uni_trn_data.shape[0]}')
    print(f'Number of unique test sessions: {uni_tst_data.shape[0]}')
    print(f'Total number of sessions: {uni_trn_data.shape[0] + uni_tst_data.shape[0]}')
    print()

    mean_trn_s_len = get_sess_lengths(uni_trn_data)
    mean_tst_s_len = get_sess_lengths(uni_tst_data)
    mean_s_len = pd.concat([mean_trn_s_len, mean_tst_s_len])
    print(f'Mean unique training session length: {mean_trn_s_len.mean()}')
    print(f'Mean unique test session length: {mean_tst_s_len.mean()}')
    print(f'Mean unique session length: {mean_s_len.mean()}')
    print()
    # print(np.isin(f_tst_sess, f_trn_sess).sum())

    # Number of sessions matching the original dataset
    data = pd.read_csv(os.path.join('./dataset', dataset_full, f'{dataset_full}.inter'), delimiter='\t')
    item_id_lists = data.groupby('session_id:token').apply(lambda s: ' '.join([str(item) for item in s['item_id:token']]))
    print(f'Number of sessions in full dataset: {item_id_lists.shape[0]} -- {data["session_id:token"].nunique()}')
    print('Hey')


def filter_data(data, min_session_length, min_item_support):
    n_rows = data.shape[0] + 1

    while n_rows > data.shape[0]:
        n_rows = data.shape[0]
        session_lengths = data.groupby(SESSION_COL).size()
        session_lengths = session_lengths[session_lengths >= min_session_length]
        data = data[np.in1d(data[SESSION_COL], session_lengths.index)]

        # filter item support
        data['ItemSupport'] = data.groupby(ITEM_COL)[ITEM_COL].transform('count')
        data = data[data.ItemSupport >= min_item_support]

        # filter session length
        session_lengths = data.groupby(SESSION_COL).size()
        data = data[np.in1d(data[SESSION_COL], session_lengths[session_lengths >= min_session_length].index)]

    data_start = datetime.fromtimestamp(data[TIME_COL].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_COL].max(), timezone.utc)

    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data[SESSION_COL].nunique(), data[ITEM_COL].nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    return data


def split_data(data, days_test):
    data_end = datetime.fromtimestamp(data[TIME_COL].max(), timezone.utc)
    test_from = data_end - timedelta(days=days_test)
    val_from = data_end - timedelta(days=2*days_test)

    session_max_times = data.groupby(SESSION_COL)[TIME_COL].max()
    session_train = session_max_times[session_max_times < val_from.timestamp()].index
    session_val = session_max_times[val_from.timestamp() <= session_max_times < test_from.timestamp()].index
    session_test = session_max_times[session_max_times >= test_from.timestamp()].index
    trn_data = data[np.in1d(data[SESSION_COL], session_train)]
    val_data = data[np.in1d(data[SESSION_COL], session_val)]
    val_data = val_data[np.in1d(val_data[ITEM_COL], trn_data[ITEM_COL])]
    vslength = val_data.groupby(SESSION_COL).size()
    val_data = val_data[np.in1d(val_data[SESSION_COL], vslength[vslength >= 2].index)]
    tst_data = data[np.in1d(data[SESSION_COL], session_test)]
    tst_data = tst_data[np.in1d(tst_data[ITEM_COL], trn_data[ITEM_COL])]
    tslength = tst_data.groupby(SESSION_COL).size()
    tst_data = tst_data[np.in1d(tst_data[SESSION_COL], tslength[tslength >= 2].index)]

    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(trn_data),
                                                                        trn_data[SESSION_COL].nunique(),
                                                                        trn_data[ITEM_COL].nunique()))
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(val_data),
                                                                             val_data[SESSION_COL].nunique(),
                                                                             val_data[ITEM_COL].nunique()))
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(tst_data), tst_data[SESSION_COL].nunique(),
                                                                           tst_data[ITEM_COL].nunique()))
    return trn_data, val_data, tst_data


def augment_session(s_data):
    s_ids = []
    subsessions = ['']
    for i, r in s_data.iloc[:-1].iterrows():
        s_ids.append(f'{r[SESSION_COL]}_{len(s_ids)}')
        subsessions.append(f'{subsessions[-1]} {r[ITEM_COL]}')

    return pd.DataFrame(zip(s_ids, subsessions[1:], s_data[ITEM_COL].tolist()[1:], s_data[TIME_COL].tolist()[1:]),
                        columns=[SESSION_COL, ITEM_LIST_COL, ITEM_COL, TIME_COL])


def augment_sessions(sessions):
    new = []
    for i, s in enumerate(sessions):
        new.extend([(f'{i}_{j-1}', ' '.join(map(str, s[0][:j])), s[0][j], s[1][j]) for j in range(1, len(s[0]))])
    return pd.DataFrame(new, columns=[SESSION_COL, ITEM_LIST_COL, ITEM_COL, TIME_COL])


def slice_data(data, conf, dataset):
    for slice_id in range(0, conf['n_slices']):
        train, val, test = split_data_slice(data, slice_id, conf['days_offset'] + (slice_id * conf['days_shift']),
                                            conf['days_train'], conf['days_val'], conf['days_test'])

        augment_and_save(train, val, test, dataset, slice_id=slice_id)


def split_data_slice(data, slice_id, days_offset, days_train, days_val, days_test):
    data_start = datetime.fromtimestamp(data[TIME_COL].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_COL].max(), timezone.utc)

    print('Full data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format(slice_id, len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.isoformat(),
                 data_end.isoformat()))

    start = datetime.fromtimestamp(data[TIME_COL].min(), timezone.utc) + timedelta(days_offset)
    middle_train = start + timedelta(days_train - days_val)
    middle_val = middle_train + timedelta(days_val)
    end = middle_val + timedelta(days_test)

    # prefilter the timespan
    session_max_times = data.groupby('SessionId')[TIME_COL].max()
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data_filtered = data[np.in1d(data.SessionId, greater_start.intersection(lower_end))]

    print('Slice data set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} / {} / {}'.
          format(slice_id, len(data_filtered), data_filtered.SessionId.nunique(), data_filtered.ItemId.nunique(),
                 start.date().isoformat(), middle_train.date().isoformat(), middle_val.date().isoformat(),
                 end.date().isoformat()))

    # split to train and test
    session_max_times = data_filtered.groupby('SessionId')[TIME_COL].max()
    sessions_train = session_max_times[session_max_times < middle_train.timestamp()].index
    sessions_val = session_max_times[middle_train.timestamp() < session_max_times < middle_val.timestamp()].index
    sessions_test = session_max_times[session_max_times >= middle_val.timestamp()].index

    train = data[np.in1d(data.SessionId, sessions_train)]

    print('Train set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}'.
          format(slice_id, len(train), train.SessionId.nunique(), train.ItemId.nunique(), start.date().isoformat(),
                 middle_train.date().isoformat()))

    val = data[np.in1d(data.SessionId, sessions_val)]
    val = val[np.in1d(val.ItemId, val.ItemId)]

    vslength = val.groupby('SessionId').size()
    val = val[np.in1d(val.SessionId, vslength[vslength >= 2].index)]

    print('Validation set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format(slice_id, len(val), val.SessionId.nunique(), val.ItemId.nunique(), middle_train.date().isoformat(),
                 middle_val.date().isoformat()))

    test = data[np.in1d(data.SessionId, sessions_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]

    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]

    print('Test set {}\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {} \n\n'.
          format(slice_id, len(test), test.SessionId.nunique(), test.ItemId.nunique(), middle_val.date().isoformat(),
                 end.date().isoformat()))

    return train, val, test


def single_split(data, config, dataset):
    print('Splitting the data...')
    trn_data, val_data, tst_data = split_data(data, config['days_test'])
    del data

    augment_and_save(trn_data, val_data, tst_data, dataset)


def augment_and_save(trn_data, val_data, tst_data, dataset, slice_id=None):
    if slice_id is None:
        mode = 'single'
        suffix = ''
    else:
        mode = 'slices'
        suffix = f'.{slice_id}'

    if not os.path.exists(os.path.join(DATASET_PATH, mode)):
        os.makedirs(os.path.join(DATASET_PATH, mode))

    if not os.path.exists(os.path.join(DATASET_PATH, mode, dataset)):
        os.makedirs(os.path.join(DATASET_PATH, mode, dataset))

    print('Sorting the data...')
    trn_data.sort_values(by=TIME_COL, inplace=True)
    val_data.sort_values(by=TIME_COL, inplace=True)
    tst_data.sort_values(by=TIME_COL, inplace=True)

    print('Augmenting Training data...')
    trn_sessions = trn_data.groupby(SESSION_COL).apply(lambda s: (s[ITEM_COL].tolist(), s[TIME_COL].tolist())).tolist()
    trn_data = augment_sessions(trn_sessions)

    print('Saving Training data...')
    trn_data.to_csv(os.path.join('./dataset', dataset, mode, f'{dataset}.train{suffix}.inter'), index=False, sep='\t')
    del trn_data, trn_sessions

    print('Augmenting Validation data...')
    val_sessions = val_data.groupby(SESSION_COL).apply(lambda s: (s[ITEM_COL].tolist(), s[TIME_COL].tolist())).tolist()
    val_data = augment_sessions(val_sessions)

    print('Saving Validation data...')
    val_data.to_csv(os.path.join('./dataset', dataset, mode, f'{dataset}.val{suffix}.inter'), index=False, sep='\t')
    del val_data, val_sessions

    print('Augmenting Test data...')
    tst_sessions = tst_data.groupby(SESSION_COL).apply(lambda s: (s[ITEM_COL].tolist(), s[TIME_COL].tolist())).tolist()
    tst_data = augment_sessions(tst_sessions)

    print('Saving Test data...')
    tst_data.to_csv(os.path.join('./dataset', dataset, mode, f'{dataset}.test{suffix}.inter'), index=False, sep='\t')


def prepare_dataset(dataset, config=None, slices=False):
    def prep_time(data):
        data['time'] = data.time.fillna(0).astype(np.int64)
        data['date'] = data.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        data['datestamp'] = data['date'].apply(lambda x: x.timestamp())
        data['time'] = (data['time'] / 1000)
        data[TIME_COL] = data['time'] + data['datestamp']
        data.drop(['time', 'date', 'datestamp'], axis=1, inplace=True)
        return data

    if config is None:
        with open(os.path.join('./config/data/preprocessing', f'{dataset}.yaml'), 'r') as f:
            config = yaml.safe_load(f)

    print('Reading the data...')
    data = pd.read_csv(os.path.join(PREFIX_PATH, DATASET_PATH, dataset, f'{dataset}.inter'), delimiter='\t')
    if dataset == 'diginetica':
        data = prep_time(data)

    print('Filtering the data...')
    data = filter_data(data, config['min_session_length'], config['min_item_support'])

    print('Encoding Item IDs...')
    data[ITEM_COL] = data[ITEM_COL].astype('category').cat.codes + 1

    if slices:
        slice_data(data, config)

    else:
        single_split(data, config)

    print('Done')


def view_results():
    pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.expand_frame_repr', False)

    results_path = './results'
    df_results = None
    opt_metric = 'tst_mrr@20'
    for file in os.listdir(results_path):
        with open(os.path.join(results_path, file), 'r') as f:
            content = f.read()

        for results in content.split('\n\n'):
            if results:
                l_results = results.split('\n')
                # Configuration appears in the first line
                config = literal_eval("{'" + l_results[0].replace(', ', "', '").replace(':', "': '") + "'}")
                results = {k: config[k] for k in ['dataset', 'model']}
                results['config'] = ', '.join([f'{k}: {v}' for k, v in config.items() if k not in ['dataset', 'model']])
                results['file'] = file
                # Third line contains validation results for this configuration
                val_results = literal_eval("{'" + l_results[2].rstrip().replace('    ', "', '").replace(' : ', "': '") + "'}")
                # Fifth line contains the test results for this configuration
                tst_results = literal_eval("{'" + l_results[4].rstrip().replace('    ', "', '").replace(' : ', "': '") + "'}")

                metrics = list(tst_results.keys())
                for k in metrics:
                    val_results['val_' + k] = val_results.pop(k)
                for k in metrics:
                    tst_results['tst_' + k] = tst_results.pop(k)

                results.update(val_results)
                results.update(tst_results)

                if df_results is None:
                    df_results = pd.DataFrame(data=[results.values()], columns=list(results.keys()))
                else:
                    df_results = df_results.append(results, ignore_index=True)

    df_results = df_results.sort_values(opt_metric, ascending=False).drop_duplicates(['file'])
    keep_cols = ['dataset', 'model', 'config'] + [c for c in df_results.columns if 'mrr' in c or 'hit' in c]

    print(df_results[keep_cols].head())
