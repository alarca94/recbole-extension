import os
import re
import pandas as pd
from ast import literal_eval


def view_results(model_name=None, dataset=None):
    log_path = './log'
    log_files = os.listdir(log_path)

    if model_name is not None:
        log_files = [file for file in log_files if model_name in file]

    results = None

    for file in log_files:
        model_name, month, day, year_hour, minute, sec = file.replace('.log', '').split('-')
        year, hour = year_hour.split('_')

        best_epoch_pattern = 'Finished training, best eval result in epoch \d+'
        dataset_pattern = "Namespace\(dataset='[^,]*"
        test_results_pattern = 'test result: \{.*\}'

        with open(os.path.join(log_path, file), 'r') as f:
            log = f.read()
            dataset_name = re.findall(dataset_pattern, log)[0][19:-1]
            best_epoch = int(re.findall(best_epoch_pattern, log)[0][45:])
            test_results = literal_eval(re.findall(test_results_pattern, log)[0][13:])
            test_results.update({'dataset': dataset_name, 'best_epoch': best_epoch, 'model': model_name,
                                 'date': f'{day}-{month}-{year} {hour}-{minute}-{sec}'})

        if results is None:
            results = pd.DataFrame(data=test_results, index=[0])
        else:
            results = results.append(test_results, ignore_index=True)

    first_cols = ['dataset', 'model', 'best_epoch', 'date']
    results = results[first_cols + [c for c in results.columns if c not in first_cols]]

    print(results.to_latex(index=False))


if __name__ == '__main__':
    view_results(model_name=None, dataset=None)