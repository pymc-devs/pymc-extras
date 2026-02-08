import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class StandardScalerDF(StandardScaler, TransformerMixin, BaseEstimator):
    def __init__(self, with_mean=True, with_std=True):
        super().__init__(with_mean=with_mean, with_std=with_std)

    def transform(self, X, y=None):
        z = super().transform(X)
        return pd.DataFrame(z, index=X.index, columns=X.columns)

    def fit_transform(self, X, y=None):
        z = super().fit_transform(X)
        return pd.DataFrame(z, index=X.index, columns=X.columns)
