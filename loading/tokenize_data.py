import pandas as pd
from .match_token_end_index import match_token_end_index
from .cut_text_in_dataframe_to_token_size import cut_text_in_dataframe_to_token_size
from transformers import AutoTokenizer

def tokenize_data(data, cfg):
    #title = list(data['Description(明細)'].astype(str).values)
    #title = list(data['text'].astype(str).values)
    title = list(data)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    '''#辞書追加
    master = pd.read_csv('./data/master/DS_master.csv')
    master = master[~master['ES_DHM5'].isnull()]
    master.drop_duplicates(subset='ES_DHM5', inplace=True)
    masters = master['ES_DHM5'].tolist()
    print('vocab size', len(tokenizer))
    tokenizer.add_tokens(masters, special_tokens=True)
    print('vocab size', len(tokenizer))'''

    encoding_end_index_list_title, text_encodeed_list = match_token_end_index(title, cfg.model.text_component.title, tokenizer)

    """data = cut_text_in_dataframe_to_token_size(data, encoding_end_index_list_title, text_encodeed_list=text_encodeed_list,
                                               original_col_name='Description(明細)', new_col_name='title_new', new_col_name2='形態素解析')"""
    
    """data = cut_text_in_dataframe_to_token_size(data, encoding_end_index_list_title, text_encodeed_list=text_encodeed_list,
                                               original_col_name='text', new_col_name='title_new', new_col_name2='形態素解析')"""

    """#コメントアウト
    data[cfg.data.text] = data['title_new']"""

    return data, tokenizer