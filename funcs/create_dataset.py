import torch
import os
from os import path
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from utils.make_save_directory_test_data import make_save_directory_test_data

def create_dataset(data, cfg, train_eval_flg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    
    '''#辞書追加
    master = pd.read_csv('./data/master/DS_master.csv')
    master = master[~master['ES_DHM5'].isnull()]
    master.drop_duplicates(subset='ES_DHM5', inplace=True)
    masters = master['ES_DHM5'].tolist()
    tokenizer.add_tokens(masters, special_tokens=True)'''

    if train_eval_flg == 'train':
        train_data, test_data = train_test_split(data, test_size=cfg.hyper_parameter.train_test_split_size,
                                                 random_state=cfg.hyper_parameter.random_state, stratify=data['label'])
        train_data, valid_data = train_test_split(train_data, test_size=cfg.hyper_parameter.train_val_split_size,
                                                 random_state=cfg.hyper_parameter.random_state, stratify=train_data['label'])
        
        test_data_save_dir = make_save_directory_test_data(cfg)

        test_data.to_csv(path.join(test_data_save_dir, 'test_data' + cfg.model.exp_name + '.csv'), index=False)
        
        dataset_for_loader = [[] for _ in range(3)]
        for i, data in enumerate([train_data, valid_data, test_data]):
            text = list(data[cfg.data.text].astype(str).values)
            label = torch.from_numpy(data[cfg.data.ans].astype(int).values)
            for j in range(len(data)):
                encoding = tokenizer(
                    text[j],
                    max_length=cfg.model.max_length,
                    padding=cfg.model.padding,
                    truncation=cfg.model.truncation,
                )
                encoding['labels'] = label[j]
                encoding = {k: torch.tensor(v, dtype=torch.int64) for k, v in encoding.items()}
                dataset_for_loader[i].append(encoding)
        
        dataset_train = dataset_for_loader[0]
        dataset_val = dataset_for_loader[1]
        dataset_test = dataset_for_loader[2]

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=cfg.model.train_batch_size,
            shuffle=cfg.model.train_shuffle,
        )

        dataloader_val = DataLoader(
            dataset_val,
            batch_size=cfg.model.val_batch_size,
        )

        dataloader_test = DataLoader(
            dataset_test,
            batch_size=cfg.model.test_batch_size,
        )

        return dataloader_train, dataloader_val, dataloader_test, tokenizer
    

    else:
        dataset_for_loader = []

        text = list(data[cfg.data.text].astype(str).values)
        label = torch.from_numpy(data[cfg.data.ans].astype(int).values)

        for j in range(len(data)):
            encoding = tokenizer(
                text[j],
                max_length=cfg.model.max_length,
                padding=cfg.model.padding,
                truncation=cfg.model.truncation,
            )
            encoding['labels'] = label[j]
            encoding = {k: torch.tensor(v, dtype=torch.int64) for k, v in encoding.items()}
            dataset_for_loader.append(encoding)
    
        dataloader_pred = DataLoader(
            dataset_for_loader,
            batch_size=cfg.model.val_batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True
        )

        return dataloader_pred