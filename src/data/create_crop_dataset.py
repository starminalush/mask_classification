import os.path

import click
import cv2
import numpy as np
import pandas as pd
from loguru import logger
from pandas import DataFrame
from pandas.core.series import Series
from tqdm import tqdm


@click.command()
@click.argument("clean_annotation_filepath", type=click.Path(exists=True))
@click.argument("inputdir_filepath", type=click.Path())
@click.argument("outputdir_filepath", type=click.Path())
def main(
    clean_annotation_filepath: str, inputdir_filepath: str, outputdir_filepath: str
):
    """
    make binary classification dataset for data from download_external_dataset stage
    @param clean_annotation_filepath: preprocessed file anootation.csv from make_dataset stage
    @type clean_annotation_filepath: str
    @param inputdir_filepath: datadir with raw external images
    @type inputdir_filepath: str
    @param outputdir_filepath: datadir with cropped external images
    @type outputdir_filepath: str
    """

    df: DataFrame = pd.read_csv(clean_annotation_filepath)
    if not os.path.exists(outputdir_filepath):
        os.mkdir(outputdir_filepath)
    for i, row in enumerate(tqdm(df.iterrows(), total=len(df))):
        current_row: Series = row[1]
        image: np.ndarray = cv2.imread(
            os.path.join(inputdir_filepath, f"{current_row['file']}.png")
        )
        label: str = current_row["name"]
        if label == "mask_weared_incorrect":
            label = "with_mask"
        if not os.path.exists(f"{outputdir_filepath}/{label}"):
            os.mkdir(f"{outputdir_filepath}/{label}")
        x_min: int = current_row["xmin"]
        y_min: int = current_row["ymin"]
        x_max: int = current_row["xmax"]
        y_max: int = current_row["ymax"]
        crop_img: np.ndarray = image[y_min:y_max, x_min:x_max]

        cv2.imwrite(
            os.path.join(
                outputdir_filepath, label, f"{current_row['file']}_{i}_cropped.png"
            ),
            crop_img,
        )
    logger.info("end cropping image")


if __name__ == "__main__":
    main()
