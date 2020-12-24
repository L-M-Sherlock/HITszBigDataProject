import pandas as pd
import numpy as np


df = pd.read_csv("d_train.csv", encoding='GBK')
cols = df.columns.values
new_cols = []

for col in cols:
    col = col.replace('%', '')
    col = col.replace('\n', '')
    new_cols.append(col)

df.columns = new_cols
pd.set_option('display.width', None)

df.drop(['id', '体检日期'], axis=1, inplace=True)
df['性别'] = df['性别'].map({"男": 0, "女": 1}).fillna(0).astype(int)
for col in df.columns:
    null_rate = df[col].isnull().sum() / df.shape[0]
    if null_rate < 0.3:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].map(lambda x: 0 if x is np.nan else 1)
    if col == '血糖':
        continue
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

df.to_csv('process_data.csv', index=False)
