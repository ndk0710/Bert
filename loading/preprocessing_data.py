from .tokenize_data import tokenize_data
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import datetime

def preprocessing_data(data, cfg):
    le = LabelEncoder()
    le = le.fit(data[cfg.data.ans])
    data[cfg.data.ans] = le.transform(data[cfg.data.ans])

    label_tag_df = pd.DataFrame(columns=['label', 'tag'])
    label_tag_df['tag'] = le.classes_
    label_tag_df['label'] = label_tag_df.index

    now = datetime.datetime.now()

    label_tag_df.to_csv(
        './label_to_tag_' + cfg.model.exp_name + '.csv', index=False
    )

    return data