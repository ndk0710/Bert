from datasets import load_dataset
import pandas as pd

emotions = load_dataset("emotion")

emotions.set_format(type="pandas")
df = emotions["train"]
df.to_csv('./data/input/testdata.csv')
df = pd.read_csv('./data/input/testdata.csv')
#df.rename(columns={'text':'商品部(ヘッダ)'},inplace=True)
df.replace({'label':{0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}}, inplace=True)
df.to_csv('./data/input/testdata.csv', index=False)
print(type(emotions))