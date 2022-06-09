import os

import click
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USER")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

from kaggle import api

api.authenticate()


@click.command()
@click.argument("external_dataset_path", type=click.Path())
def main(external_dataset_path: str):
    ds_name = "andrewmvd/face-mask-detection"
    api.dataset_download_files(ds_name, path=external_dataset_path, unzip=True)
    logger.info("finish download dataset")


if __name__ == "__main__":
    main()
