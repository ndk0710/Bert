import numpy as np
import pandas as pd

RANDOM_STATE = 42

def hist(number):
    for index in range(40):
        if (0.05*index - 1.00 < number) & (number <= 0.05*index - 0.95):
            return index 

rng = np.random.default_rng()
x = pd.DataFrame(rng.random(50),columns=['cosine'])

hist_info = [hist(row['cosine']) for _, row in x.iterrows()]
x['hist_info'] = pd.DataFrame(hist_info)

print(x['hist_info'].value_counts())

extract_x = []

for index in range(40):
    tmp = x[x['hist_info']==index]
    if not tmp.empty:
        num = len(tmp)
        extract_tmp = tmp.sample(n=int(num/2), random_state=RANDOM_STATE)
        """if index == 38:
            extract_tmp = tmp.sample(n=300, random_state=RANDOM_STATE)
        elif index == 39:
            extract_tmp = tmp.sample(n=400, random_state=RANDOM_STATE)
        else:
            num = len(tmp)
            extract_tmp = tmp.sample(n=int(num/2), random_state=RANDOM_STATE)"""
        extract_x.append(extract_tmp)

output = pd.concat(extract_x)

print(output['hist_info'].value_counts())
        



ooo = x.fillna('').apply(hist)