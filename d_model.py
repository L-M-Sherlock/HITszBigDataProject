import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.metrics import mean_squared_error


class Model:
    def __init__(self):
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
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 0.01)
        self.train_data = df

    def train(self, X, y):
        model = ensemble.GradientBoostingRegressor(n_estimators=100)
        model.fit(X, y.values.ravel())
        self.model = model
        return model

    def predict(self, X):
        pred_y = self.model.predict(X)
        return pred_y


if __name__ == '__main__':
    model = Model()
    df = model.train_data
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
    model.train(x_train, y_train)
    ans = model.predict(x_test)
    pd_data = pd.DataFrame(ans, columns=['血糖'])
    print(mean_squared_error(y_test['血糖'].values, pd_data['血糖'].values) / 2)
