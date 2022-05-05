# from utils import BaseLogger
import warnings
from typing import Tuple

import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from prefect import Flow, task
from prefect.engine.results import LocalResult
from prefect.engine.serializers import PandasSerializer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

FINAL_OUTPUT = LocalResult(
    "data/final/",
    location="{task_name}.csv",
    serializer=PandasSerializer("csv", serialize_kwargs={"index": False}),
)


@task
def load_features(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, delimiter=",")
    return data


@task
def train_val_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    X = df.drop("product_name", axis=1)
    y = df["product_name"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    # wandb.log({"table": X_val})
    return X_train, X_val, y_train, y_val


@hydra.main(config_path="../config", config_name="main")
def train_model(config: DictConfig):
    with Flow("train") as flow:
        """Function to train the model"""

        input_path = abspath(config.processed.path)
        output_path = abspath(config.final.path)

        print(f"Train modeling using {input_path}")
        print(f"Model used: {config.model.name}")
        print(f"Save the output to {output_path}")
    flow.run()


if __name__ == "__main__":
    train_model()
