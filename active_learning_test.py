import pandas as pd
import numpy as np

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

from utils.set_config import set_config

from lib.advanced_active_learning import AdvancedActiveLearning

def remove_duplicates(lst):
    """
    リスト内の重複要素を削除し、インデックスが小さいものを残す関数
    Args:
        lst (list): 重複を含むリスト
    Returns:
        list: 重複を削除したリスト
    """
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

def remove_elements(np_array, elements_to_remove):
    """
    numpy型の1次元文字列配列から指定したリストに一致する要素を削除する関数
    Args:
        np_array (numpy.ndarray): numpy型の1次元文字列配列
        elements_to_remove (list): 削除したい文字列のリスト
    Returns:
        numpy.ndarray: 指定した要素が削除されたnumpy型の1次元文字列配列
    """
    mask = np.isin(np_array, elements_to_remove, invert=True)
    return np_array[mask]

if __name__ == '__main__':
    #config設定
    cfg = set_config('config_multi_3')
    
    #データ読込み
    #original_data = pd.read_csv('./data/test/test.csv')
    original_data = pd.read_csv('./data/input/testdata.csv')

    #ラベル情報取得
    labels = original_data.dropna(subset=['label'], axis=0)
    labels.drop_duplicates(subset='label', inplace=True)
    labels = labels['label'].to_list()

    original_data.dropna(subset=['text'], axis=0, inplace=True)
    #unlabeled_data = unlabeled_data['text'].to_numpy()
    


    """能動学習（データサンプリング）"""

    #変数
    perc_uncertain = 0.1                            #サンプリングする割合
    num_clusters = 10                               #クラスタリング数
    #num = int(len(unlabeled_data)*perc_uncertain)   #サンプリングで取得する総数
    num = 30
    num_labels = 6                                  #分類したいクラス数


    #インスタンス作成
    instance = AdvancedActiveLearning()

    device = torch.device('cuda')

    #初回のサンプリングのモデル
    config = AutoConfig.from_pretrained(cfg.model.model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_config(config).to(device)
    
    #2回目以降のサンプリングのモデル
    #model = AutoModelForSequenceClassification.from_pretrained(cfg.directory.use_model_dir).to(device)
    model.eval()

    #初期値
    original_data['1st_sammpling'] = 'False'

    for label in labels:
        unlabeled_data = original_data.loc[original_data['label']==label]['text'].to_numpy()

        index = 0

        #必要なデータ数獲得まで
        while True:
            """クラスタリング"""
            samples = instance.get_clustered_uncertainty_samples(model, unlabeled_data, device, cfg, perc_uncertain, num_clusters)
            if index==0:
                pickup_samples=samples
            else:
                pickup_samples.extend(samples)
            
            #データからサンプリング結果除外
            column_index = 1
            filtered_list = [row[column_index] for row in samples]

            unlabeled_data = remove_elements(unlabeled_data, filtered_list)

            #サンプリング数の確認
            if len(pickup_samples) >= num:
                pickup_samples = pickup_samples[:num]
                break
            
            index += 1

        #サンプリング情報をカラムに格納
        pickup_list = [row[column_index] for row in pickup_samples]
        
        original_data.loc[original_data['text'].isin(pickup_list), '1st_sampling'] = 'True'
        check=0

    #サンプリング結果を記録
    original_data.to_csv('output.csv', index=False)

    