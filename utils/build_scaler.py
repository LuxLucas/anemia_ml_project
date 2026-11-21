from sklearn.base import clone
from sklearn.pipeline import Pipeline
from utils.MultiScaler import MultiScaler


SCALERS = [
    "standard",
    "minmax",
    "robust",
    "quantile"
]


def build_scaler(
    model,
    apply_log=True,
    skew_threshold=1.0,
    random_state=1978
):
    build = {}

    for method in SCALERS:

        build[method] = Pipeline([
            (
                "scaler",
                MultiScaler(
                    method=method,
                    apply_log=apply_log,
                    skew_threshold=skew_threshold,
                    random_state=random_state
                )
            ),
            ("model", clone(model))
        ])

    return build