import os

import click
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USER')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

import kaggle

kaggle.api.authenticate()


@click.command()
@click.argument('external_dataset_path', type=click.Path())
def main(external_dataset_path: str):
    kaggle.api.dataset_download_files('andrewmvd/face-mask-detection', path=external_dataset_path,
                                      unzip=True)
    logger.info('finish download dataset')


if __name__ == '__main__':
    main()
