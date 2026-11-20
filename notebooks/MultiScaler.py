import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer
)
from sklearn.utils.validation import check_is_fitted


class MultiScaler(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        method="standard",
        skew_threshold=1.0,
        apply_log=True,
        random_state=1822,
        scaler_kwargs=None
    ):
        self.method = method
        self.skew_threshold = skew_threshold
        self.apply_log = apply_log
        self.random_state = random_state
        self.scaler_kwargs = scaler_kwargs

    def _get_scaler(self):
        scalers = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
            "quantile": QuantileTransformer
        }

        if self.method not in scalers:
            raise ValueError(
                f"Método '{self.method}' inválido. "
                f"Escolha entre: {list(scalers.keys())}"
            )

        kwargs = self.scaler_kwargs or {}

        if self.method == "quantile":
            kwargs = {
                "random_state": self.random_state,
                **kwargs
            }

        return scalers[self.method](**kwargs)

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MultiScaler espera um DataFrame pandas.")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        self.num_cols_ = X.select_dtypes(
            include=np.number
        ).columns.tolist()

        if self.apply_log:
            self.cols_para_log_ = [
                col
                for col in self.num_cols_
                if (
                    X[col].skew() > self.skew_threshold
                    and (X[col].dropna() >= 0).all()
                )
            ]
        else:
            self.cols_para_log_ = []

        if self.num_cols_:
            X_fit = X.astype({col: np.float64 for col in self.num_cols_})
        else:
            X_fit = X.copy()

        if self.cols_para_log_:
            X_fit[self.cols_para_log_] = np.log1p(
                X_fit[self.cols_para_log_]
            )

        self.scaler_ = self._get_scaler()

        if self.num_cols_:
            self.scaler_.fit(X_fit[self.num_cols_])

        return self

    def transform(self, X):
        check_is_fitted(
            self,
            ["scaler_", "num_cols_", "cols_para_log_"]
        )

        if not isinstance(X, pd.DataFrame):
            raise TypeError("MultiScaler espera um DataFrame pandas.")

        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(
                "As colunas de X devem ser iguais às utilizadas no fit, "
                "na mesma ordem."
            )

        if self.num_cols_:
            X_transformed = X.astype({col: np.float64 for col in self.num_cols_})
        else:
            X_transformed = X.copy()

        if self.cols_para_log_:
            X_transformed[self.cols_para_log_] = np.log1p(
                X_transformed[self.cols_para_log_]
            )

        if self.num_cols_:
            X_transformed[self.num_cols_] = self.scaler_.transform(
                X_transformed[self.num_cols_]
            )

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")

        if input_features is not None:
            return np.asarray(input_features, dtype=object)

        return self.feature_names_in_
