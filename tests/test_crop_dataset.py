import os

import pandas as pd
from click.testing import CliRunner

from src.data.create_crop_dataset import create_crop_dataset

runner = CliRunner()


def test_cli_command():
    result = runner.invoke(
        create_crop_dataset,
        [
            "data/interim/annotations.csv",
            "data/external/face-mask-detection/images/",
            "data/processed/cropped_images/",
        ],
    )
    assert result.exit_code == 0


def test_cropped_image_amount():
    df = pd.read_csv("data/interim/annotations.csv")
    # test amount of cropped image with mask with annotations
    amount_bbox_with_mask = len(df[df["name"] != "without_mask"])
    cropped_filenames_amount_with_mask = len(
        os.listdir("data/processed/cropped_images/with_mask")
    )
    assert amount_bbox_with_mask == cropped_filenames_amount_with_mask
    # test amount of cropped image without mask with annotations
    amount_bbox_without_mask = len(df[df["name"] == "without_mask"])
    cropped_filenames_amount_without_mask = len(
        os.listdir("data/processed/cropped_images/without_mask")
    )
    assert amount_bbox_without_mask == cropped_filenames_amount_without_mask
    # # test amount of all cropped image with annotations
    amount_bbox = len(df)
    cropped_filenames_amount = len(
        os.listdir("data/processed/cropped_images/without_mask")
    ) + len(os.listdir("data/processed/cropped_images/with_mask"))
    assert amount_bbox == cropped_filenames_amount
