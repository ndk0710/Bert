import pandas as pd
import os

def translate_label_to_tag(df, csv_path):

    path_label_to_tag = os.path.join(csv_path, 'label_to_tag.csv')
    label_to_tag = pd.read_csv(path_label_to_tag)

    df['inferred_tag'] = df['predetected_label']
    df['inferred_tag'] = df['inferred_tag'].replace(label_to_tag['label'].to_list(), label_to_tag['tag'])

    return df