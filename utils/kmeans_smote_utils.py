from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import QhullError, Voronoi, voronoi_plot_2d

from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler


CLASS_NAMES = {
    0: "Sem anemia",
    1: "HGB-anemia",
    2: "Deficiência de ferro",
    3: "Deficiência de folato",
    4: "Deficiência de B12",
}


def make_scalers(random_state: int = 2005) -> Dict[str, object]:
    """Retorna os escalonadores usados nos experimentos."""
    return {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "quantile": QuantileTransformer(
            output_distribution="normal",
            random_state=random_state,
        ),
    }


def load_processed_data(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carrega X_train, X_test, y_train e y_test a partir de uma pasta."""
    data_dir = Path(data_dir)
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze("columns")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train e X_test precisam ter as mesmas colunas e ordem.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("X e y possuem quantidades incompatíveis de linhas.")

    return X_train, X_test, y_train, y_test


def _as_frame(values, columns: Iterable[str], index=None) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.copy()
    return pd.DataFrame(values, columns=list(columns), index=index)


def fit_kmeans_scenarios(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_clusters: int = 5,
    random_state: int = 2005,
    scalers: Optional[Mapping[str, object]] = None,
) -> Dict[str, dict]:
    """Treina K-means sem oversampling, mantendo objetos e dados transformados."""
    scalers = dict(scalers or make_scalers(random_state))
    scenarios = {}

    for name, scaler in scalers.items():
        Xtr = _as_frame(scaler.fit_transform(X_train), X_train.columns)
        Xte = _as_frame(scaler.transform(X_test), X_test.columns)
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
        model.fit(Xtr)
        scenarios[name] = {
            "scaler": scaler,
            "model": model,
            "X_train_scaled": Xtr,
            "X_test_scaled": Xte,
            "clusters_test": model.predict(Xte),
            "X_resampled": Xtr,
            "y_resampled": None,
            "summary": None,
        }
    return scenarios


def smote_mrs(
    X: pd.DataFrame,
    y: pd.Series,
    n_groups: int = 5,
    target_ratio: float = 0.20,
    k_max: int = 3,
    random_state: int = 2005,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Aplica K-means + SMOTE local + ROS no conjunto fornecido.

    O target_ratio é deliberadamente parcial: cada classe minoritária chega a
    target_ratio * tamanho_da_maior_classe do grupo, não a 100% da maior classe.
    """
    X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    group_labels = KMeans(
        n_clusters=n_groups,
        n_init=20,
        random_state=random_state,
    ).fit_predict(X)

    X_parts, y_parts, summaries = [], [], []
    binary_columns = [c for c in X.columns if X[c].nunique(dropna=False) == 2]

    for group in np.unique(group_labels):
        mask = group_labels == group
        Xg = X.loc[mask].copy()
        yg = y.loc[mask].copy()
        before = yg.value_counts().sort_index()
        target = int(target_ratio * before.max())
        requested = {cls: target for cls, count in before.items() if count < target}
        n_smote = 0
        n_ros = 0

        if len(before) >= 2 and requested:
            smote_strategy = {
                cls: target for cls, target in requested.items() if before[cls] >= 2
            }
            if smote_strategy:
                k = max(1, min(k_max, min(before[cls] for cls in smote_strategy) - 1))
                Xg, yg = SMOTE(
                    sampling_strategy=smote_strategy,
                    k_neighbors=k,
                    random_state=random_state,
                ).fit_resample(Xg, yg)
                Xg = _as_frame(Xg, X.columns)
                yg = pd.Series(yg)
                n_smote = len(yg) - int(before.sum())

            after_smote = yg.value_counts()
            ros_strategy = {
                cls: target for cls in requested if after_smote.get(cls, 0) < target
            }
            if ros_strategy:
                before_ros = len(yg)
                Xg, yg = RandomOverSampler(
                    sampling_strategy=ros_strategy,
                    random_state=random_state,
                ).fit_resample(Xg, yg)
                Xg = _as_frame(Xg, X.columns)
                yg = pd.Series(yg)
                n_ros = len(yg) - before_ros

        for col in binary_columns:
            original_values = np.sort(X[col].unique())
            Xg[col] = original_values[
                np.abs(Xg[col].to_numpy()[:, None] - original_values[None, :]).argmin(axis=1)
            ]

        X_parts.append(Xg)
        y_parts.append(pd.Series(yg).reset_index(drop=True))
        summaries.append({
            "Grupo": int(group),
            "Originais": int(before.sum()),
            "Sintéticos (SMOTE)": int(n_smote),
            "Duplicados (ROS)": int(n_ros),
            "Total final": int(len(yg)),
        })

    X_resampled = pd.concat(X_parts, ignore_index=True)
    y_resampled = pd.concat(y_parts, ignore_index=True)
    return X_resampled, y_resampled, pd.DataFrame(summaries)


def fit_smote_mrs_scenarios(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    n_clusters: int = 5,
    target_ratio: float = 0.20,
    random_state: int = 2005,
    scalers: Optional[Mapping[str, object]] = None,
) -> Dict[str, dict]:
    """Treina K-means após SMOTE-MRS, sem tocar no conjunto de teste."""
    scalers = dict(scalers or make_scalers(random_state))
    scenarios = {}

    for name, scaler in scalers.items():
        Xtr_scaled = _as_frame(scaler.fit_transform(X_train), X_train.columns)
        Xte_scaled = _as_frame(scaler.transform(X_test), X_test.columns)
        Xres, yres, summary = smote_mrs(
            Xtr_scaled,
            y_train,
            n_groups=n_clusters,
            target_ratio=target_ratio,
            random_state=random_state,
        )
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
        model.fit(Xres)
        scenarios[name] = {
            "scaler": scaler,
            "model": model,
            "X_train_scaled": Xtr_scaled,
            "X_test_scaled": Xte_scaled,
            "clusters_test": model.predict(Xte_scaled),
            "X_resampled": Xres,
            "y_resampled": yres,
            "summary": summary,
        }
    return scenarios


def clustering_metrics(
    scenarios: Mapping[str, dict],
    y_test: pd.Series,
    scenario_name: str,
) -> pd.DataFrame:
    rows = []
    y_test = pd.Series(y_test).to_numpy()
    for scaler_name, item in scenarios.items():
        labels = item["clusters_test"]
        Xtest = item["X_test_scaled"]
        rows.append({
            "Cenário": scenario_name,
            "Scaler": scaler_name,
            "ARI": adjusted_rand_score(y_test, labels),
            "NMI": normalized_mutual_info_score(y_test, labels),
            "Silhouette": silhouette_score(Xtest, labels),
            "Davies-Bouldin": davies_bouldin_score(Xtest, labels),
            "Calinski-Harabasz": calinski_harabasz_score(Xtest, labels),
        })
    return pd.DataFrame(rows)


def compare_scenarios(
    baseline: Mapping[str, dict],
    smote: Mapping[str, dict],
    y_test: pd.Series,
) -> pd.DataFrame:
    return pd.concat([
        clustering_metrics(baseline, y_test, "Sem SMOTE"),
        clustering_metrics(smote, y_test, "SMOTE-MRS"),
    ], ignore_index=True)


def plot_metric_comparison(metrics: pd.DataFrame, figsize=(14, 9)):
    cols = ["ARI", "NMI", "Silhouette", "Davies-Bouldin"]
    long = metrics.melt(
        id_vars=["Cenário", "Scaler"],
        value_vars=cols,
        var_name="Métrica",
        value_name="Valor",
    )
    g = sns.catplot(
        data=long,
        x="Scaler",
        y="Valor",
        hue="Cenário",
        col="Métrica",
        col_wrap=2,
        kind="bar",
        height=4,
        aspect=1.25,
        sharey=False,
    )
    g.set_axis_labels("Scaler", "Valor")
    g.set_titles("{col_name}")
    g.figure.suptitle("Comparação entre K-means sem SMOTE e SMOTE-MRS", y=1.03)
    return g


def plot_resampling_counts(smote_scenarios: Mapping[str, dict], figsize=(12, 6)):
    rows = []
    for scaler_name, item in smote_scenarios.items():
        summary = item["summary"].copy()
        rows.extend([
            {"Scaler": scaler_name, "Tipo": "Originais", "Quantidade": summary["Originais"].sum()},
            {"Scaler": scaler_name, "Tipo": "Sintéticos (SMOTE)", "Quantidade": summary["Sintéticos (SMOTE)"].sum()},
            {"Scaler": scaler_name, "Tipo": "Duplicados (ROS)", "Quantidade": summary["Duplicados (ROS)"].sum()},
        ])
    counts = pd.DataFrame(rows)
    ax = sns.barplot(data=counts, x="Scaler", y="Quantidade", hue="Tipo", errorbar=None)
    ax.set_title("Composição do conjunto de treino após SMOTE-MRS")
    ax.set_xlabel("Scaler")
    ax.set_ylabel("Quantidade de amostras")
    return ax


def plot_contingency_heatmap(item: dict, y_test: pd.Series, normalize: bool = True, ax=None):
    table = pd.crosstab(
        pd.Series(y_test, name="Classe real"),
        pd.Series(item["clusters_test"], name="Cluster"),
    )
    if normalize:
        table = table.div(table.sum(axis=1), axis=0) * 100
    table.index = [CLASS_NAMES.get(i, str(i)) for i in table.index]
    ax = ax or plt.gca()
    sns.heatmap(table, annot=True, fmt=".1f" if normalize else "d", cmap="Blues", ax=ax)
    ax.set_title("Classe real × cluster" + (" (%)" if normalize else ""))
    return ax


def plot_pca_comparison(
    baseline_item: dict,
    smote_item: dict,
    y_test: pd.Series,
    title_suffix: str = "",
    figsize=(14, 6),
):
    pca = PCA(n_components=2, random_state=2005)
    pca.fit(baseline_item["X_train_scaled"])
    Xbase = pca.transform(baseline_item["X_test_scaled"])
    Xsmote = pca.transform(smote_item["X_test_scaled"])
    y_test = pd.Series(y_test).to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panels = [
        (Xbase, baseline_item["clusters_test"], "Clusters sem SMOTE"),
        (Xsmote, smote_item["clusters_test"], "Clusters com SMOTE-MRS"),
        (Xbase, y_test, "Classes reais"),
    ]
    for ax, (coords, labels, title) in zip(axes, panels):
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=12, alpha=0.55)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
    fig.suptitle(f"PCA: comparação dos agrupamentos {title_suffix}", y=1.02)
    plt.tight_layout()
    return fig, axes


def plot_feature_boxplots(
    baseline_item: dict,
    smote_item: dict,
    X_test: pd.DataFrame,
    feature: str,
    figsize=(14, 5),
):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, item, title in [
        (axes[0], baseline_item, "Sem SMOTE"),
        (axes[1], smote_item, "SMOTE-MRS"),
    ]:
        plot_data = X_test[[feature]].copy()
        plot_data["Cluster"] = item["clusters_test"]
        sns.boxplot(data=plot_data, x="Cluster", y=feature, hue="Cluster", legend=False, ax=ax, palette="viridis")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Distribuição de {feature} por cluster", y=1.02)
    plt.tight_layout()
    return fig, axes


def plot_voronoi_comparison(
    baseline_item: dict,
    smote_item: dict,
    title_suffix: str = "",
    figsize=(16, 7),
):
    """Compara Voronoi e centroides do K-means nos dois cenários.

    A PCA é ajustada no treino escalonado do cenário base. Como os cenários
    usam o mesmo scaler, os centroides podem ser projetados no mesmo plano.
    """
    pca = PCA(n_components=2, random_state=2005)
    pca.fit(baseline_item["X_train_scaled"])

    panels = [
        (baseline_item, "K-means sem SMOTE"),
        (smote_item, "K-means com SMOTE-MRS"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for ax, (item, title) in zip(axes, panels):
        coords = pca.transform(item["X_test_scaled"])
        centroids_df = pd.DataFrame(
            item["model"].cluster_centers_,
            columns=baseline_item["X_train_scaled"].columns,
        )
        centroids = pca.transform(centroids_df)
        labels = item["clusters_test"]

        try:
            vor = Voronoi(centroids)
            voronoi_plot_2d(
                vor,
                ax=ax,
                show_vertices=False,
                show_points=False,
                line_colors="black",
                line_width=1.2,
                line_alpha=0.65,
            )
        except QhullError:
            ax.text(
                0.5,
                0.96,
                "Voronoi indisponível para estes centroides",
                transform=ax.transAxes,
                ha="center",
                va="top",
            )

        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=labels, cmap="tab10", s=14, alpha=0.45,
            label="Amostras de teste",
        )
        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            marker="X", s=220,
            c=np.arange(len(centroids)), cmap="tab10",
            edgecolors="black", linewidths=1.2,
            label="Centroides",
        )
        for idx, (x, y) in enumerate(centroids):
            ax.annotate(
                f"C{idx}", (x, y), xytext=(6, 6),
                textcoords="offset points", fontsize=10, fontweight="bold",
            )

        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")

    fig.suptitle(f"Diagramas de Voronoi com centroides {title_suffix}", y=1.02)
    plt.tight_layout()
    return fig, axes

def plot_voronoi_grid(
    scenarios: Mapping[str, dict],
    random_state: int = 2005,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Plota os diagramas de Voronoi (projetados em PCA 2D) para todos os scalers de um cenário."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for ax, (name, item) in zip(axes.flat, scenarios.items()):
        pca = PCA(n_components=2, random_state=random_state)
        pca.fit(item["X_train_scaled"])
        coords = pca.transform(item["X_test_scaled"])

        centers_df = pd.DataFrame(
            item["model"].cluster_centers_,
            columns=item["X_train_scaled"].columns,
        )
        centroides = pca.transform(centers_df)

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=item["clusters_test"],
            cmap="tab10",
            s=12,
            alpha=0.55,
        )

        try:
            vor = Voronoi(centroides)
            voronoi_plot_2d(
                vor,
                ax=ax,
                show_vertices=False,
                show_points=False,
                line_colors="#333333",
                line_width=1.4,
                line_alpha=0.8,
            )
        except QhullError:
            pass

        ax.scatter(
            centroides[:, 0],
            centroides[:, 1],
            c="red",
            marker="X",
            s=120,
            edgecolors="black",
            linewidths=1.2,
            label="Centroides",
            zorder=5,
        )

        ax.set_title(f"Voronoi (PCA) — {name}", fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    return fig
