import copy
import pandas as pd
import numpy as np

def cut_text_in_dataframe_to_token_size(data, encoding_end_index_list, text_encoded_list, original_col_name, new_col_name, new_col_name2):
    def identify_col_index(data, target_column):
        col_index = 0
        for i in list(data.columns.values):
            if target_column == i:
                break
            col_index += 1
        return col_index

    df = copy.deepcopy(data)
    df[new_col_name] = pd.Series()
    df[new_col_name2] = pd.Series()

    col_index_original = identify_col_index(df, original_col_name)
    col_index_new = identify_col_index(df, new_col_name)
    col_index_new2 = identify_col_index(df, new_col_name2)
    
    for (idx, row), encoding_end_index, text_encoded in zip(df.iterrows(), encoding_end_index_list, text_encoded_list):
        try:
            df.iloc[idx, col_index_new2] = str(text_encoded)
            if row[col_index_original] is np.nan:
                encoding_end_index = 0
                df.iloc[idx, col_index_new] = str('')[0: encoding_end_index]
            else:
                df.iloc[idx, col_index_new] = str(row[col_index_original])[0: encoding_end_index] + '[SEP]'
        except IndexError:
            print(str(row))
            print(col_index_original)
            print(encoding_end_index)

    return df