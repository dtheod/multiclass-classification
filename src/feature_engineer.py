# import warnings
from typing import Tuple

import hydra
import joblib
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from imblearn.over_sampling import SMOTE
from omegaconf import DictConfig
from prefect import Flow, task
from prefect.engine.results import LocalResult
from prefect.engine.serializers import PandasSerializer
# from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder

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
def create_features(df: pd.DataFrame, feat_path: str) -> pd.DataFrame:
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
    joblib.dump(
        df.filter(
            [
                "component_name",
                "product_name",
                "assignee_name",
                "n_components",
                "n_assignees",
                "n_un_components",
                "n_product_rows",
            ]
        ).drop_duplicates(),
        feat_path,
    )
    return df


@task
def one_hot_encoding(
    df: pd.DataFrame, enc_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = df.drop("product_name", axis=1)
    y = df["product_name"]
    print(X.dtypes)

    df_object = X.select_dtypes("object")
    df_nobject = X.select_dtypes(exclude="object")
    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    enc.fit(df_object)
    joblib.dump(enc, abspath(enc_path))

    codes = enc.transform(df_object)
    feature_names = enc.get_feature_names(df_object.columns)
    df_one_hot = pd.DataFrame(codes, columns=feature_names).astype(int)
    df = pd.concat([df_nobject, df_one_hot], axis=1)
    return df, y


@task
def feature_variance(
    df: pd.DataFrame, thres: float, var_path: str
) -> pd.DataFrame:
    selector = VarianceThreshold(threshold=thres)
    res = selector.fit(df)
    joblib.dump(res, abspath(var_path))
    df = df[df.columns[res.get_support(indices=True)]]
    print(df.shape)
    return df


@task
def principal_components(df: pd.DataFrame, pca_path: str) -> PCA:
    pca = PCA(n_components=5)
    pca.fit(df)
    joblib.dump(pca, abspath(pca_path))
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
        df = create_features(df, config.feats.path)
        X, y = one_hot_encoding(df, config.encoder.path)
        X = feature_variance(
            X, config.parameters.var_thres, config.variance.path
        )
        X, y = oversampling(X, y)
        pca = principal_components(X, config.pca.path)
        X = apply_principal_components(X, pca)
        df = model_input(X, y)

    flow.run()


if __name__ == "__main__":
    feature_data()
