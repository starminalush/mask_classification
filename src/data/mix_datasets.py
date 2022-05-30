import os.path
import shutil

import click
from loguru import logger


@click.command()
@click.argument('raw_dataset_filepath', type=click.Path(exists=True))
@click.argument('external_dataset_filepath', type=click.Path(exists=True))
@click.argument('train_dataset_filepath', type=click.Path())
def main(raw_dataset_filepath: str, external_dataset_filepath: str, train_dataset_filepath: str):
    if not os.path.exists(train_dataset_filepath):
        os.mkdir(train_dataset_filepath)
        os.mkdir(f"{train_dataset_filepath}/with_mask")
        os.mkdir(f"{train_dataset_filepath}/without_mask")

    for data_dir in [raw_dataset_filepath, external_dataset_filepath]:
        for class_dir in os.listdir(data_dir):
            for filename in os.listdir(f"{data_dir}/{class_dir}"):
                shutil.copy(f"{data_dir}/{class_dir}/{filename}", f"{train_dataset_filepath}/{class_dir}/{filename}")
    logger.info('finish merge raw and external data')


if __name__ == '__main__':
    main()
