import os
import shutil

import click
import numpy as np
from loguru import logger


@click.command()
@click.argument('input_datadir', type=click.Path(exists=True))
@click.argument('output_datadir', type=click.Path())
def main(input_datadir: str, output_datadir: str):
    classes_dir = ['with_mask', 'without_mask']

    test_ratio = 0.20
    val_ratio = 0.10

    for cls in classes_dir:
        os.makedirs(output_datadir + 'train/' + cls)
        os.makedirs(output_datadir + 'test/' + cls)
        os.makedirs(output_datadir + 'val/' + cls)

        src = input_datadir + cls

        all_filenames = os.listdir(src)
        np.random.shuffle(all_filenames)
        train_filenames, test_filenames, val_filenames = \
            np.split(np.array(all_filenames),
                     [int(len(all_filenames) * (1 - test_ratio)), int(len(all_filenames) * (1 - val_ratio))])

        train_filenames = [src + '/' + name for name in train_filenames.tolist()]
        test_filenames = [src + '/' + name for name in test_filenames.tolist()]
        val_filenames = [src + '/' + name for name in val_filenames.tolist()]

        logger.debug("*****************************")
        logger.debug(f'Class {cls}')
        logger.debug(f'Total images: {len(all_filenames)}')
        logger.debug(f'Training: {len(train_filenames)}')
        logger.debug(f'Testing: {len(test_filenames)}')
        logger.debug(f'Validating: {len(val_filenames)}')
        logger.debug("*****************************")

        for name in train_filenames:
            shutil.copy(name, output_datadir + 'train/' + cls)
        for name in test_filenames:
            shutil.copy(name, output_datadir + 'test/' + cls)
        for name in test_filenames:
            shutil.copy(name, output_datadir + 'val/' + cls)
    logger.info("Copying Done!")


if __name__ == '__main__':
    main()
