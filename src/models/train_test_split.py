import os
import shutil
from typing import List
import shutil
import click
import numpy as np
from loguru import logger


@click.command()
@click.argument("input_datadir", type=click.Path(exists=True))
@click.argument("dataset_type", type=click.STRING)
@click.argument("output_datadir", type=click.Path())
def train_test_split(input_datadir: str, dataset_type: str, output_datadir: str):
    """
    split data on train, test and val
    @param input_datadir: input dataset(internal, external, both)
    @type input_datadir: str
    @param dataset_type: dataset type(internal, external, both)
    @type dataset_type: str
    @param output_datadir: output dataset(internal, external, both)
    @type output_datadir: str
    """
    if os.path.exists(output_datadir):
        shutil.rmtree(output_datadir)
    classes_dir: List = ["with_mask", "without_mask"]

    test_ratio: float = 0.20
    val_ratio: float = 0.10

    for cls in classes_dir:
        os.makedirs(os.path.join(output_datadir, "train", cls))
        os.makedirs(os.path.join(output_datadir, "test", cls))
        os.makedirs(os.path.join(output_datadir, "val", cls))

        src: str = os.path.join(input_datadir, cls)

        all_filenames: List = os.listdir(src)
        np.random.shuffle(all_filenames)
        train_filenames, test_filenames, val_filenames = np.split(
            np.array(all_filenames),
            [
                int(len(all_filenames) * (1 - test_ratio)),
                int(len(all_filenames) * (1 - val_ratio)),
            ],
        )

        train_filenames = [os.path.join(src, name) for name in train_filenames.tolist()]
        test_filenames = [os.path.join(src, name) for name in test_filenames.tolist()]
        val_filenames = [os.path.join(src, name) for name in val_filenames.tolist()]
        logger.debug(f"Start splitting dataset for type {dataset_type}")
        logger.debug("*****************************")
        logger.debug(f"Class {cls}")
        logger.debug(f"Total images: {len(all_filenames)}")
        logger.debug(f"Training: {len(train_filenames)}")
        logger.debug(f"Testing: {len(test_filenames)}")
        logger.debug(f"Validating: {len(val_filenames)}")
        logger.debug("*****************************")

        for name in train_filenames:
            shutil.copy(name, os.path.join(output_datadir, "train", cls))
        for name in test_filenames:
            shutil.copy(name, os.path.join(output_datadir, "test", cls))
        for name in val_filenames:
            shutil.copy(name, os.path.join(output_datadir, "val", cls))
    logger.info("Copying Done!")


if __name__ == "__main__":
    train_test_split()
