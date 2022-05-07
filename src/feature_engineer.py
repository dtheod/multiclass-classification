# import warnings
from typing import Tuple

import hydra
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from omegaconf import DictConfig
from prefect import Flow, task
from prefect.engine.results import LocalResult
from prefect.engine.serializers import PandasSerializer
# from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

# from sklearn.pipeline import Pipeline

INTERMEDIATE_OUTPUT = LocalResult(
    "data/features_data/",
    location="{task_name}.csv",
    serializer=PandasSerializer("csv", serialize_kwargs={"index": False}),
)


@task
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, delimiter=",")


@task
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.assign(
            n_components=lambda df_: df_.groupby(
                ["product_name", "component_name"]
            )["component_name"].transform("count")
        )
        .assign(
            n_assignees=lambda df_: df_.groupby(
                ["product_name", "assignee_name"]
            )["assignee_name"].transform("count")
        )
        .assign(
            n_un_components=lambda df_: df_.groupby(["product_name"])[
                "component_name"
            ].transform("nunique")
        )
        .assign(
            n_product_rows=lambda df_: df_.groupby(["product_name"])[
                "product_name"
            ].transform("count")
        )
    )
    return df


@task
def one_hot_encoding(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = df.drop("product_name", axis=1)
    y = df["product_name"]
    X = pd.get_dummies(X)
    return X, y


@task
def feature_variance(df: pd.DataFrame, thres: float) -> pd.DataFrame:
    selector = VarianceThreshold(threshold=thres)
    res = selector.fit(df)
    df = df[df.columns[res.get_support(indices=True)]]
    return df


@task
def principal_components(df: pd.DataFrame) -> PCA:
    pca = PCA(n_components=5)
    pca.fit(df)
    return pca


@task
def apply_principal_components(df: pd.DataFrame, pca: PCA) -> pd.DataFrame:
    return pca.transform(df)


@task
def oversampling(
    df_x: pd.DataFrame, df_y: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ros = SMOTE(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(df_x, df_y)
    return X_resampled, y_resampled


@task(result=INTERMEDIATE_OUTPUT)
def model_input(df_x: np.ndarray, df_y: pd.DataFrame) -> pd.DataFrame:
    df_x = pd.DataFrame(
        df_x, columns=["pca" + str(c) for c in range(df_x.shape[1])]
    )
    return pd.concat([df_x, df_y], axis=1)


@hydra.main(config_path="../config", config_name="main")
def feature_data(config: DictConfig):

    with Flow("model_input") as flow:

        df = load_data(config.processed.path)
        df = create_features(df)
        X, y = one_hot_encoding(df)
        X = feature_variance(X, config.parameters.var_thres)
        X, y = oversampling(X, y)
        pca = principal_components(X)
        X = apply_principal_components(X, pca)
        df = model_input(X, y)

    flow.run()


if __name__ == "__main__":
    feature_data()
