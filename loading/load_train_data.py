import pandas as pd
from os import path
from .preprocessing_data import preprocessing_data

def load_train_data(cfg):
    data = pd.read_csv(path.join(cfg.directory.input_dir, cfg.data.input_dataname + '.csv'), encoding='utf-8')
    data.dropna(how='all', axis=0, inplace=True)
    data = data[~data['text'].isnull()]

    #data['language'] = data['Description'].fillna('').apply(judge_language2)

    data = data.reset_index()
    data = preprocessing_data(data, cfg)

    return data