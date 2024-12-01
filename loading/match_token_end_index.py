import copy
import re

def match_token_end_index(text_list, token_length, tokenizer):
    encoding_end_index_list = []
    text_encoded_list = []

    for text in text_list:
        text_lowered = text.lower()
        text_encoded = tokenizer.tokenize(text)
        text_encoded_list.append(text_encoded)

        max_morpheme_list = text_encoded[0: token_length]
        max_morpheme_list.reverse()

        for idx, morpheme in enumerate(max_morpheme_list):
            match_length = 0
            try:
                match_length = len(re.findall(morpheme, text_lowered))
            except Exception as e:
                # WARN: エラーの原因を確認する
                pass
            # if morheme > 2を加えるか検討
            if match_length == 1:
                break
            else:
                continue
        
        max_morpheme_list_cut = copy.deepcopy(max_morpheme_list[0:idx+1])
        max_morpheme_list_cut.reverse()

        match_index_last = 0
        for morpheme in max_morpheme_list_cut:
            match_index = text_lowered.find(morpheme, match_index_last)

            if match_index == -1:
                assert match_index == -1, 'Warning: 複数テキスト結合時の前処理エラー'
                break
            match_index_last = match_index + len(morpheme)

        encoding_end_index_list.append(match_index_last)

    return encoding_end_index_list, text_encoded_list