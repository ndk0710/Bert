import pandas as pd

def remove_duplicates_preserve_order(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

data = pd.read_csv('./rensyuu.csv')
keywords = data['キーワード'].to_numpy()

dict_words = []

for keyword in keywords:
    #カンマ区切り
    comma_separated_keywords = keyword.split(',')
    for comma_separated_keyword in comma_separated_keywords:
        #スラッシュ区切り
        slash_separeted_keywords = comma_separated_keyword.split('/')
        for slash_separeted_keyword in slash_separeted_keywords:
            #特殊文字([,')を削除
            slash_separeted_keyword = slash_separeted_keyword.replace('[', '')
            slash_separeted_keyword = slash_separeted_keyword.replace("'", '')
            #特殊文字(])区切り
            special_char_separated_keywords = slash_separeted_keyword.split(']')
            for special_char_separated_keyword in special_char_separated_keywords:
                dict_words.append(special_char_separated_keyword)
        
#リスト要素
dict_words = [dict_word for dict_word in dict_words if dict_word != '']
unique_dict_words = remove_duplicates_preserve_order(dict_words)

outputs = pd.DataFrame(unique_dict_words, columns=['keywords'])

outputs.to_csv(f'./output.csv', columns=['keywords'], index=False)