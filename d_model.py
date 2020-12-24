import pandas as pd
import math
from xgboost import XGBRegressor


def mean(lst):
    # the average of a list
    return float(sum(lst)) / len(lst)


def mse(l1, l2):
    return mean([math.pow(abs(l1[i] - l2[i]), 2) for i in range(len(l1))]) / 2


df = pd.read_csv('process_data.csv')
df = df.iloc[:, 1:]
cols = list(df.columns.values)
cols.remove('血糖')
x = df[cols]
y = df[['血糖']]
splitpoint = int(0.8 * df.shape[0])
x_train = x[:splitpoint]
y_train = y[:splitpoint]
x_test = x[splitpoint:]
y_test = y[splitpoint:]

model = XGBRegressor(max_depth=4, learning_rate=0.1, n_estimators=360, objective='reg:gamma')
model.fit(x_train, y_train)
ans = model.predict(x_test)
pd_data = pd.DataFrame(ans, columns=['血糖'])
print(mse(y_test['血糖'].values, pd_data['血糖'].values))
