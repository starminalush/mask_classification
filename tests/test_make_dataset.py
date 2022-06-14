import os

import great_expectations as ge
import pandas as pd
from click.testing import CliRunner

from src.data.make_dataset import make_dataset

runner = CliRunner()


def test_cli_command():
    result = runner.invoke(
        make_dataset,
        [
            "data/external/face-mask-detection/annotations",
            "data/interim/annotations.csv",
        ],
    )
    assert result.exit_code == 0


def test_make_dataset():
    df = pd.read_csv("data/interim/annotations.csv")
    df_ge = ge.from_pandas(df)

    expected_columns = [
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "name",
        "file",
        "width",
        "height",
    ]
    list_filenames = os.listdir("data/external/face-mask-detection/images")
    list_filenames = [filename.split(".")[0] for filename in list_filenames]
    assert (
        df_ge.expect_table_columns_to_match_ordered_list(
            column_list=expected_columns
        ).success
        is True
    )
    assert df_ge.expect_column_values_to_not_be_null(column="xmin").success is True
    assert df_ge.expect_column_values_to_not_be_null(column="ymin").success is True
    assert df_ge.expect_column_values_to_not_be_null(column="xmax").success is True
    assert df_ge.expect_column_values_to_not_be_null(column="ymax").success is True
    assert df_ge.expect_column_values_to_not_be_null(column="name").success is True
    assert df_ge.expect_column_values_to_not_be_null(column="file").success is True
    assert df_ge.expect_column_values_to_not_be_null(column="width").success is True
    assert df_ge.expect_column_values_to_not_be_null(column="height").success is True
    assert df["file"].tolist().sort() is list_filenames.sort()
