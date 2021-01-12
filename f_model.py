import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


class Model:
    def __init__(self):
        df = pd.read_csv("f_train.csv", encoding='GBK')
        cols = df.columns.values
        new_cols = []
        for col in cols:
            col = col.replace('%', '')
            col = col.replace('\n', '')
            new_cols.append(col)
        df.columns = new_cols
        pd.set_option('display.width', None)
        df.drop(['id'], axis=1, inplace=True)
        for col in df.columns:
            null_rate = df[col].isnull().sum() / df.shape[0]
            if null_rate < 0.3:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].map(lambda x: 0 if x is np.nan else 1)
            if col == 'label':
                continue
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 0.01)
        self.train_data = df

    def train(self, X, y):
        model = LogisticRegression(C=100.0, random_state=1, max_iter=5000)
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
    cols.remove('label')
    x = df[cols]
    y = df[['label']]
    splitpoint = int(0.8 * df.shape[0])
    x_train = x[:splitpoint]
    y_train = y[:splitpoint]
    x_test = x[splitpoint:]
    y_test = y[splitpoint:]
    model.train(x_train, y_train)
    ans = model.predict(x_test)
    pd_data = pd.DataFrame(ans, columns=['label'])
    print(f1_score(y_test['label'].values, pd_data['label'].values) / 2)
