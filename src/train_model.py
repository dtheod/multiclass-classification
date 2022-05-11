# from utils import BaseLogger
import warnings
from functools import partial
from typing import Callable, Tuple
from urllib.parse import urlparse

import hydra
import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from omegaconf import DictConfig
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def load_features(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, delimiter=",")
    return data


def stratified_split(
    df_x: pd.DataFrame, df_y: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    ss.get_n_splits(df_x, df_y)
    for train_index, test_index in ss.split(df_x, df_y):
        X_train, X_test = df_x.iloc[train_index], df_x.iloc[test_index]
        y_train, y_test = df_y[train_index], df_y[test_index]
    return X_train, X_test, y_train, y_test


def label_encoding(
    df: pd.DataFrame, l_enc_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lb = LabelEncoder()
    X = df.drop("product_name", axis=1)
    y = df["product_name"]
    lb.fit(y)
    joblib.dump(lb, abspath(l_enc_path))
    labels = lb.transform(y)
    return X, labels


def get_objective(
    df_x_train: pd.DataFrame,
    df_x_test: pd.DataFrame,
    df_y_train: pd.DataFrame,
    df_y_test: pd.DataFrame,
    config: DictConfig,
    space: dict,
):

    model = XGBClassifier(
        objective=config.model.objective,
        n_estimators=space["n_estimators"],
        max_depth=int(space["max_depth"]),
        gamma=space["gamma"],
        reg_alpha=int(space["reg_alpha"]),
        min_child_weight=int(space["min_child_weight"]),
        colsample_bytree=int(space["colsample_bytree"]),
    )

    evaluation = [
        (df_x_train.values, df_y_train),
        (df_x_test.values, df_y_test),
    ]

    model.fit(
        df_x_train.values,
        df_y_train,
        eval_set=evaluation,
        eval_metric="mlogloss",
        verbose=False,
        early_stopping_rounds=50,
    )

    pred = model.predict(df_x_test)
    f1 = f1_score(df_y_test, pred, average="micro")
    print("SCORE:", f1)
    return {"loss": f1, "status": STATUS_OK}


def optimize(objective: Callable, space: dict):
    trials = Trials()
    best_hyper = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=5,
        trials=trials,
    )
    print("The best hyperparameters are : ", "\n")
    print(best_hyper)

    mlflow.log_param("max_depth", best_hyper["max_depth"])
    mlflow.log_param("colsample_bytree", best_hyper["colsample_bytree"])
    mlflow.log_param("gamma", best_hyper["gamma"])
    mlflow.log_param("max_depth", best_hyper["max_depth"])

    best_model = XGBClassifier(
        n_estimators=space["n_estimators"],
        colsample_bytree=best_hyper["colsample_bytree"],
        gamma=best_hyper["gamma"],
        max_depth=int(best_hyper["max_depth"]),
        min_child_weight=best_hyper["min_child_weight"],
        reg_alpha=best_hyper["reg_alpha"],
        reg_lambda=best_hyper["reg_lambda"],
    )
    return best_model


def predict(model: XGBClassifier, X_test: pd.DataFrame):
    return model.predict(X_test)


@hydra.main(config_path="../config", config_name="main")
def train_model(config: DictConfig):

    with mlflow.start_run():

        # Loading the model input data
        model_input = load_features(abspath(config.model_input.path))

        # Encode the label/target column from object to int
        df_x, df_y = label_encoding(model_input, config.label_encoder.path)

        # Stratified split
        X_train, X_test, y_train, y_test = stratified_split(df_x, df_y)

        # Modelling
        space = {
            "max_depth": hp.quniform("max_depth", **config.model.max_depth),
            "gamma": hp.uniform("gamma", **config.model.gamma),
            "reg_alpha": hp.quniform("reg_alpha", **config.model.reg_alpha),
            "reg_lambda": hp.uniform("reg_lambda", **config.model.reg_lambda),
            "colsample_bytree": hp.uniform(
                "colsample_bytree", **config.model.colsample_bytree
            ),
            "min_child_weight": hp.quniform(
                "min_child_weight", **config.model.min_child_weight
            ),
            "n_estimators": config.model.n_estimators,
            "seed": config.model.seed,
        }
        objective = partial(
            get_objective, X_train, X_test, y_train, y_test, config
        )

        # Find best model
        best_model = optimize(objective, space)

        # Fit the best model
        best_model.fit(X_train.values, y_train)

        # Predict
        prediction = predict(best_model, X_test)
        train_prediction = predict(best_model, X_train)

        accuracy = balanced_accuracy_score(y_test, prediction)
        f1 = f1_score(y_test, prediction, average="micro")
        train_accuracy = balanced_accuracy_score(y_train, train_prediction)
        train_f1 = f1_score(y_train, train_prediction, average="micro")
        print("Accuracy Score of this model is P{}:".format(accuracy))
        print("F1 Score of this model is P{}:".format(f1))

        # Log parameters and metrics
        # mlflow.log_param(best_model, config.process.features)
        # log_params(best_model, features = X_train.columns)
        mlflow.log_metric("test accuracy_score", accuracy)
        mlflow.log_metric("test f1 score", f1)
        mlflow.log_metric("train accuracy_score", train_accuracy)
        mlflow.log_metric("train f1 score", train_f1)

        mlflow.log_metric("variance threshold", config.parameters.var_thres)
        mlflow.log_metric("principal_components", config.parameters.var_thres)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            mlflow.sklearn.log_model(
                best_model, "model", registered_model_name="XGBoostModel"
            )
        else:
            mlflow.sklearn.log_model(best_model, "model")

        joblib.dump(best_model, abspath(config.model.path))


if __name__ == "__main__":
    train_model()
