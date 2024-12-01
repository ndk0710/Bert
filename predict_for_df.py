import warnings
warnings.simplefilter('ignore')
import os
import sys
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
sys.path.append(os.path.join(Path().resolve(), '../'))
sys.path.append(os.path.join(Path().resolve(), '../.'))
from utils.set_config import set_config
from funcs.create_dataset import create_dataset
from loading.tokenize_data import tokenize_data
from utils.translate_label_to_tag import translate_label_to_tag
from funcs.excecute_model import excecute_model
import gc

from os import path

def analysis_function_by_organization(test_df, col_correct, col_pred):
    test_df['result'] = (test_df[col_pred] == test_df[col_correct]).astype(int)
    test_df['result']

    #カテゴリデータ毎にデータ件数をカウントしてCount列を追加
    test_df['Count'] = test_df.groupby(col_correct)[col_correct].transform('count')

    #カテゴリデータ毎にB列の合計を計算してsum列を追加
    test_df['Correct'] = test_df.groupby(col_correct)['result'].transform('sum')

    #ユニークな結果だけ表示
    unique_df = test_df.drop_duplicates(subset=[col_correct])

    #正解率計算
    unique_df['Precision'] = unique_df['Correct']/unique_df['Count']

    unique_df_ = unique_df[[col_correct, 'Count', 'Correct', 'Precision']].sort_values('Precision')

    return unique_df_

def calc_cos(all_features, category_feature_vector):
    all_cos = []
    for feature in all_features:
        cos_sim = np.dot(feature, category_feature_vector)/(np.linalg.norm(feature)*np.linalg.norm(category_feature_vector))
        all_cos.append(cos_sim)

    return all_cos

def calc_cos_all(all_features):
    data = pd.read('./output_feature_vector.csv', eccoding='shift-jis')
    category_feature_vectors = data.to_numpy()
    TD_cos, EMI_cos, FMC_cos, IMC_cos, thr_cos, other_cos = [], [], [], [], [], []
    for feature in all_features:
        for index, category_feature_vector in enumerate(category_feature_vectors.T):
            cos_sim = np.dot(feature, category_feature_vector) / (np.linalg.norm(feature) * np.linalg.norm(category_feature_vector))
            if index == 0:
                TD_cos.append(cos_sim)
            elif index == 1:
                EMI_cos.append(cos_sim)
            elif index == 2:
                FMC_cos.append(cos_sim)
            elif index == 3:
                IMC_cos.append(cos_sim)
            elif index == 4:
                thr_cos.append(cos_sim)
            elif index == 5:
                other_cos.append(cos_sim)
    
    return TD_cos, EMI_cos, FMC_cos, IMC_cos, thr_cos, other_cos

def exact(all_features, all_losses):
    outputs = []
    indexs = np.array(outputs)
    for index in indexs:
        outputs.append(all_features[index])
    outputs = np.array(outputs)
    outputs = pd.DataFrame(outputs)
    outputs.to_csv('./Top30_vectors_EMI.csv', index = False)

def KNN(all_querys):
    dict = pd.read('./Top30_vectors.csv', encoding='shift-jis', header=None)
    dict = dict.to_numpy()
    dict = dict.copy(order='c')
    dict = dict.astype(np.float32)
    dim = int(all_querys.shape[1])
    all_querys = np.array(all_querys)
    all_querys = all_querys.copy(order='c')
    all_querys = all_querys.astype(np.float32)
    index = faiss.IndexFlatL2(dim)
    #faiss.normalize_L2_(dict)
    index.add(dict)
    #faiss.normalize_L2(all_query)
    distances, indices = index.search(all_querys, k=5)


def predict_for_df():
    #config設定
    cfg = set_config('config_multi_3')

    #data = pd.read_csv('./data/test/20240729_model_seed_42multi_testdata/test_datamodel_seed_42multi.csv')
    data = pd.read_csv(path.join(cfg.directory.test_dir, cfg.data.test_dataname + '.csv'), encoding='utf-8')

    #textが入っていない列を削除
    tmp_data_null = data[data['text'].isnull()]
    data.dropna(subset=['text'], axis=0, inplace=True)

    #メモリ解放
    #del data
    gc.collect()

    #dataloader生成
    data, tokenizer = tokenize_data(data, cfg)
    train_eval_flg = 'eval'
    dataloader_pred = create_dataset(data, cfg, train_eval_flg)

    #推論実行
    all_preds, all_losses, all_features, category_feature_vector, all_probs, all_indexs, all_counts = excecute_model(dataloader_pred, cfg, tokenizer)
    all_cos = calc_cos(all_features, category_feature_vector)

    #推論結果をdfに整理
    pred_data = pd.DataFrame()
    pred_data["predetected_label"] = all_preds
    pred_data["loss"] = all_losses
    pred_data["cosine"] = all_cos
    #pred_Data["attention"] = all_htmls
    pred_data["top-k-probs"] = all_probs
    pred_data["top-k-indexs"] = all_indexs
    pred_data["top-k-accuracy-counts"] = all_counts
    pred_data["features"] = all_features

    out_pred_data = pd.concat([data, pred_data], axis=1)
    out_pred_data["inferred_date"] = datetime.datetime.today().strftime('%Y/%m/%d')

    #推論結果のラベルをtagを変換
    csv_path = "./"
    out_pred_data = translate_label_to_tag(out_pred_data, csv_path)

    # inputの順番を揃える
    out_pred_data["inferred_tag"].where(~(out_pred_data["inferred_tag"].isnull()), "（削除）", inplace=True)
    
    label_to_tag = pd.read_csv("./label_to_tag.csv")
    out_pred_data["label"] = out_pred_data["label"].replace(label_to_tag["label"].to_list(), label_to_tag["tag"])
    
    print('------')
    print(analysis_function_by_organization(out_pred_data, 'label', 'inferred_tag'))
    print('------')

    out_pred_data.to_csv(f'./output.csv', columns=['text', 'label', 'inferred_tag'], index=False)
    #out_pred_data.to_csv(f'./output.csv', index=False)

    return out_pred_data

    #out_pred_data.drop("level_0", axis=1, inplace=True)
    #out_pred_data.reset_index(inplace=True)

    oo=0


    """今回のテストデータでは英語/日本語判定は不要
    #gateモデルを実行し、言語の推論結果を付与
    #data['language'] = data['text'].fillna.apply(judge_language2)

    #data.drop('level_0', axis=1, inplace=True)

    #data_jp = data[data['language']=='japanese']
    #data_en = data[data['language']=='English']
    #data_en = data
    #data_en = data[~data['ヒット'].isnull()]

    #メモリ解放
    del data
    gc.collect()

    #tokenize_data実行でインデックスをリセットしなければ、推論時のoutputの列の位置がずれる
    data_jp = data_jp.reset_index()
    data_en = data_en.reset_index()

    #言語別に推論を実行
    df_all_pred = pd.DataFrame()
    #for data, cfg in [[data_jp, cfg_jp], [data_en, cfg_en]]:
    for data, cfg in[[data_en, cfg_en]]:
        #dataloader生成
        data, tokenizer = tokenize_data(data, cfg)
        train_eval_flg = 'eval'
        dataloader_pred = create_dataset(data, cfg, train_eval_flg)

        #推論実行
        all_preds, all_losses, all_features, category_feature_vector = excecute_model(dataloader_pred, cfg, tokenizer)
        all_cos = calc_cos(all_features, category_feature_vector)

        #推論結果をdfに整理
        pred_data = pd.DataFrame()
        pred_data['predected_label'] = all_preds
        pred_data['loss'] = all_losses
        pred_data['cosine'] = all_cos
        #pred_data['attention'] = all_htmls
        out_pred_data = pd.concat([data, pred_data], axis=1)
        out_pred_data['inferred_date'] = datetime.datetime.today().strftime('%Y/%m/%d')

        #推論結果のラベルをtagと組織に変換
        csv_path = './env/conversion_table/'
        out_pred_data = translae_label_to_tag(out_pred_data, csv_path)

        #推論結果をdfに保存
        df_all_pred = pd.concat([df_all_pred, out_pred_data])

    #inputの順番を揃える
    df_all_pred = pd.concat([df_all_pred, tmp_data_null], axis=0, ignore_index=True)
    df_all_pred['inferred_date'].where(~(df_all_pred['inferred_date'].isnull()), '(削除)', inplace=True)
    df_all_pred.drop('level_0', axis=1, inplace=True)
    df_all_pred.reset_index(inplace=True)

    seed=42
    label_to_tag = pd.read_csv('./env/conversion_table/label_to_tag.csv')
    is_en = df_all_pred['language']=='English'
    df_all_pred['label'] = df_all_pred['label'].replace(label_to_tag['label'].to_list(), label_to_tag['tag'])
    is_auto = df_all_pred['ヒット'].isnull()
    is_manual = ~(df_all_pred['ヒット'].isnull())
    print('seed',seed)
    print('------')
    print(analysis_function_by_organization(df_all_pred[is_manual], 'label', 'inferred_tag'))
    print('------')

    df_all_pred.to_csv(f'./output.csv', columns=['明細No(詳細)', 'ヒット', 'Description(明細)', '形態素解析', 'label', 'inferred_tag', 'loss', 'cosine'], index=False)

    return df_all_pred"""

if __name__ == '__main__':
    predict_for_df()