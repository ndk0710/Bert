import pandas as pd

data = pd.read_csv('./rensyuu2.csv')

#重複データの確認（残らない情報）
"""disappear_data = data[['明細','番号','部署(ヘッダ)']][data['番号'].duplicated()].sort_values('番号')
#重複リスト
duplicated_list = disappear_data.drop_duplicates(subset='番号',inplace=False)"""

duplicated_list = data.drop_duplicates(subset='番号',inplace=False)


extract_x = []

for _, row in duplicated_list.iterrows():
    tmp_data = data[data['番号'] == row['番号']]
    #部署(ヘッダ)の要素数を確認
    department_number = tmp_data['部署(ヘッダ)'].nunique()
    if department_number == 1:
        extract_x.append(duplicated_list[duplicated_list['番号'] == row['番号']])

output = pd.concat(extract_x)


check=0