import subprocess
import os

import pandas as pd


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

    def get_unique_sessions(data):
        comp_sess = data['item_id_list:token_seq'] + ' ' + data['item_id:token'].astype(str)
        keep = [True] + (comp_sess[1:] != data['item_id_list:token_seq'].values[:-1]).tolist()
        return data[keep]

    # Number of equal sessions in train and test --> 752 (3812 if we don't count target items)
    uni_trn_data = get_unique_sessions(trn_data)
    uni_tst_data = get_unique_sessions(tst_data)
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