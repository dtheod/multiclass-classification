import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from fastapi import FastAPI


app = FastAPI()


class CustomFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("Initialising education feature")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        
        feat_init = joblib.load("../models/feat")
        feat_init = feat_init[['component_name', 'assignee_name', 'n_components',
                'n_assignees','n_un_components','n_product_rows']].drop_duplicates()

        feat_init = feat_init.groupby(['component_name', 'assignee_name']).agg({'n_components':'mean',
                                                     'n_assignees':'mean',
                                                     'n_un_components':'mean',
                                                     'n_product_rows':'mean'}).reset_index()

        X = pd.merge(X, feat_init, on = ['assignee_name', 'component_name'], how = 'left')
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
                "product_name",
                "resolution_code",
                "status_code",
                "quantity_of_votes",
                "quantity_of_comments",
                "bug_fix_time",
                "severity_code",
                "assignee_name",
            ]
        )
        return X



class OneHot(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("Initialising education feature")

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        onehot = joblib.load("../models/one_hot")
        df_object = X.select_dtypes("object")
        df_nobject = X.select_dtypes(exclude="object").values
        encoded = onehot.transform(df_object)
        X = np.concatenate((df_nobject,encoded), axis = 1)
        return X



class FeatureVariance(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("Initialising education feature")

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        var = joblib.load("../models/vars")
        X = var.transform(X)
        return X


class PCA_Components(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("Initialising education feature")

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        pca = joblib.load("../models/pca")
        return pca.transform(X)


class XGBoosting(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("Initialising education feature")

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        xgboost_model = joblib.load("../models/XGBoost")
        return xgboost_model.predict(X) 


class ReverseEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("Initialising education feature")

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        label_encoder = joblib.load("../models/label_enc")
        return label_encoder.inverse_transform(X)


class Project(BaseModel):
    creation_date: str
    component_name: str
    short_description: str
    long_description: str
    assignee_name: str
    reporter_name: str
    resolution_category: str
    resolution_code: int
    status_category: str
    status_code: int
    update_date: str
    quantity_of_votes: int
    quantity_of_comments: int
    resolution_date: str
    bug_fix_time: int
    severity_category: str
    severity_code: int



@app.post("/predict")
def predict(project: Project):

    data_dict = pd.DataFrame(
        {
            'creation_date': [project.creation_date],
            'component_name': [project.component_name],
            'short_description': [project.short_description],
            'long_description': [project.long_description],
            'assignee_name': [project.assignee_name],
            'reporter_name': [project.reporter_name],
            'resolution_category': [project.resolution_category],
            'resolution_code': [project.resolution_code],
            'status_category': [project.status_category],
            'status_code': [project.status_code],
            'update_date': [project.update_date],
            'quantity_of_votes': [project.quantity_of_votes],
            'quantity_of_comments': [project.quantity_of_comments],
            'resolution_date': [project.resolution_date],
            'bug_fix_time': [project.bug_fix_time],
            'severity_category': [project.severity_category],
            'severity_code': [project.severity_code]

        }
    )


    pipeline = Pipeline(steps = [
        ('selection', DataSelection()),
        ('custom',CustomFeature()),
        ('one_hot', OneHot()),
        ('feature_variance', FeatureVariance()),
        ('pca', PCA_Components()),
        ('modelling', XGBoosting()),
        ('reverse_encoder', ReverseEncoder())
    ])

    result = pipeline.transform(data_dict)
    return { 'product' :result[0]}
