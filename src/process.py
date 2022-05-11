import hydra
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from omegaconf import DictConfig
from prefect import Flow, task
from prefect.engine.results import LocalResult
from prefect.engine.serializers import PandasSerializer

INTERMEDIATE_RESULT = LocalResult(
    "data/processed/",
    location="{task_name}.csv",
    serializer=PandasSerializer("csv", serialize_kwargs={"index": False}),
)


@task
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@task
def data_selection(df: pd.DataFrame) -> pd.DataFrame:
    df = df.filter(
        [
            "creation_date",
            "component_name",
            "product_name",
            "resolution_category",
            "resolution_code",
            "status_code",
            "update_date",
            "quantity_of_votes",
            "quantity_of_comments",
            "bug_fix_time",
            "severity_code",
            "assignee_name",
        ]
    )
    return df


@task
def component_name_fix(df: pd.DataFrame) -> pd.DataFrame:

    def component_func(row):
        if row.startswith('tptp'):
            return 'tptp'
        elif row.startswith('update'):
            return "update"
        elif row.startswith('git'):
            return "git"
        elif row.startswith('ci'):
            return "ci"
        elif row.startswith('user'):
            return "user"
        elif row.startswith('sql'):
            return "sql"
        elif row.startswith('ruby'):
            return "ruby"
        elif row.startswith('gef'):
            return "gef"
        elif row.startswith('cdt'):
            return "cdt"
        elif row.startswith('wst'):
            return "wst"
        elif row.startswith('ecf'):
            return "ecf"
        elif row.startswith('jst'):
            return "jst"
        elif row.startswith('xtest'):
            return "xtest"
        elif row.startswith('tptp'):
            return "tptp"
        elif row.startswith('cdo'):
            return "cdo"
        elif 'package' in row:
            return "package"
        elif 'connector' in row:
            return "connector"
        elif 'report' in row:
            return "report"
        else:
            return row

    df = (
        df.pipe(
            lambda df_: df_.assign(
                component_name=df_["component_name"].str.lower().str.strip()
            )
        )
        .assign(
            component_name=lambda df_: df_["component_name"].apply(
                component_func
            )
        )
        .assign(
            counts=lambda df_: df_.groupby("component_name")[
                "component_name"
            ].transform("count")
        )
        .assign(
            component_name=lambda df_: np.where(
                df_["counts"] >= 3, df_["component_name"], "other"
            )
        )
        .drop("counts", axis=1)
    )
    print(df.columns)
    return df


@task
def product_name_fix(df: pd.DataFrame) -> pd.DataFrame:
    percent = 0.95
    print("using percentage: {}".format(percent))
    df1 = (
        df.product_name.value_counts()
        .reset_index()
        .rename(columns={"index": "product_name", "product_name": "counts"})
        .assign(cumsum=lambda df_: df_["counts"].cumsum())
        .assign(perce=lambda df_: df_["cumsum"] / df_["counts"].sum())
        .query("perce <= @percent")
        .filter(["product_name"])
    )
    df["product_name"] = np.where(
        df["product_name"].isin(df1["product_name"].tolist()),
        df["product_name"],
        "product_other",
    )

    return df


@task
def assignee_fix(df: pd.DataFrame) -> pd.DataFrame:
    def fuzzyness(row):
        return fuzz.ratio(row[0], row[1])

    def mapping(row):
        try:
            return map_dict[row]
        except KeyError:
            return row

    df_cnt = (
        df.groupby(["product_name", "assignee_name"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )
    idx = (
        df_cnt.groupby(["product_name"])["count"].transform(max)
        == df_cnt["count"]
    )

    max_assignee = (
        df_cnt[idx]
        .rename(columns={"assignee_name": "max_assignee_name"})
        .merge(df_cnt, how="inner", on="product_name")
        .assign(
            maxim=lambda df_: df_[
                ["max_assignee_name", "assignee_name"]
            ].apply(fuzzyness, axis=1)
        )
        .query("maxim > 60 and maxim != 100 and count_x > 1")
    )
    map_dict = dict(
        zip(max_assignee.assignee_name, max_assignee.max_assignee_name)
    )
    df["assignee_name"] = df["assignee_name"].apply(mapping)

    return df


@task(result=INTERMEDIATE_RESULT)
def final_selection(df: pd.DataFrame) -> pd.DataFrame:
    df = df.filter(
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
    return df


@hydra.main(config_path="../config", config_name="main")
def process_data(config: DictConfig):

    with Flow("process_data") as flow:
        df = (
            load_data(config.raw.path)
            .pipe(data_selection)
            .pipe(component_name_fix)
            .pipe(assignee_fix)
            .pipe(product_name_fix)
        )
        df = df.pipe(final_selection)

    flow.run()


if __name__ == "__main__":
    process_data()
