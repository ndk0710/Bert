import os
import sys
import gc

from utils.set_config import set_config
from loading.load_train_data import load_train_data
from funcs.create_dataset import create_dataset
from funcs.fine_tuning import fine_tuning

def train(cfg, seed, model_name):
    cfg.hyper_parameter.random_state = seed
    cfg.model.exp_name = model_name

    #データロード
    data = load_train_data(cfg)

    #ラベル取得
    list_multi_label = data[cfg.data.ans].value_counts()
    num_labels = len(list_multi_label)

    #データセット作成
    train_eval_flg = 'train'
    dataloader_train, dataloader_val, dataloader_test , tokenizer = create_dataset(data, cfg, train_eval_flg)

    #finetuning
    fine_tuning(dataloader_train, dataloader_val, dataloader_test, num_labels, cfg, tokenizer)

if __name__ == '__main__':
    seed = 32
    cfg = set_config('config_multi_3')
    name = 'multi'
    for i in range(3):
        gc.collect()
        seed = seed + 10
        model_name = 'model_seed_' + str(seed) + name
        print(name)
        print(seed)
        train(cfg, seed, model_name)

