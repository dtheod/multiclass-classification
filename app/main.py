import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

app = FastAPI()


class CustomFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("Initialising education feature")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        feat_init = joblib.load("/models/feat")
        feat_init = feat_init[
            [
                "component_name",
                "assignee_name",
                "n_components",
                "n_assignees",
                "n_un_components",
                "n_product_rows",
            ]
        ].drop_duplicates()

        feat_init = (
            feat_init.groupby(["component_name", "assignee_name"])
            .agg(
                {
                    "n_components": "mean",
                    "n_assignees": "mean",
                    "n_un_components": "mean",
                    "n_product_rows": "mean",
                }
            )
            .reset_index()
        )

        X = pd.merge(
            X, feat_init, on=["assignee_name", "component_name"], how="left"
        )
        print("feature transform")
        return X


class DataSelection(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("Initialising education feature")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.filter(
            [
                "component_name",
                "resolution_code",
                "status_code",
                "quantity_of_votes",
                "quantity_of_comments",
                "bug_fix_time",
                "severity_code",
                "assignee_name",
            ]
        ).pipe(
            lambda df_: df_.assign(
                component_name=df_["component_name"].str.lower().str.strip(),
                assignee_name=df_["assignee_name"].str.lower().str.strip(),
            )
        )
        return X


class OneHot(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("Initialising education feature")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        onehot = joblib.load("/models/one_hot")
        df_object = X.select_dtypes("object")
        df_nobject = X.select_dtypes(exclude="object").values
        encoded = onehot.transform(df_object)
        X = np.concatenate((df_nobject, encoded), axis=1)
        return X


class FeatureVariance(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("Initialising education feature")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        var = joblib.load("/models/vars")
        X = var.transform(X)
        return X


class PCA_Components(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("Initialising education feature")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pca = joblib.load("/models/pca")
        return pca.transform(X)


class XGBoosting(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("Initialising education feature")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        xgboost_model = joblib.load("/models/XGBoost")
        return xgboost_model.predict(X)


class ReverseEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("Initialising education feature")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        label_encoder = joblib.load("/models/label_enc")
        return label_encoder.inverse_transform(X)


class Project(BaseModel):
    creation_date: str = "2015-05-22"
    component_name: str = "engine"
    short_description: str = "LogTraceException in ProposalUtils.toMethodNam"
    long_description: str = "The following incident was reported via the au"
    assignee_name: str = "serg.boyko2011"
    reporter_name: str = "error-reports-inbox"
    resolution_category: str = "fixed"
    resolution_code: int = 1
    status_category: str = "closed"
    status_code: int = 4
    update_date: str = "2015-05-27"
    quantity_of_votes: int = 0
    quantity_of_comments: int = 8
    resolution_date: str = "2015-05-27"
    bug_fix_time: int = 2
    severity_category: str = "normal"
    severity_code: int = 2


@app.post("/predict")
def predict(project: Project):

    data_dict = pd.DataFrame(
        {
            "creation_date": [project.creation_date],
            "component_name": [project.component_name],
            "short_description": [project.short_description],
            "long_description": [project.long_description],
            "assignee_name": [project.assignee_name],
            "reporter_name": [project.reporter_name],
            "resolution_category": [project.resolution_category],
            "resolution_code": [project.resolution_code],
            "status_category": [project.status_category],
            "status_code": [project.status_code],
            "update_date": [project.update_date],
            "quantity_of_votes": [project.quantity_of_votes],
            "quantity_of_comments": [project.quantity_of_comments],
            "resolution_date": [project.resolution_date],
            "bug_fix_time": [project.bug_fix_time],
            "severity_category": [project.severity_category],
            "severity_code": [project.severity_code],
        }
    )

    pipeline = Pipeline(
        steps=[
            ("selection", DataSelection()),
            ("custom", CustomFeature()),
            ("one_hot", OneHot()),
            ("feature_variance", FeatureVariance()),
            ("pca", PCA_Components()),
            ("modelling", XGBoosting()),
            ("reverse_encoder", ReverseEncoder()),
        ]
    )

    result = pipeline.transform(data_dict)
    return {"product": result[0]}
