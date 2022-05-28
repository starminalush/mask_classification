import os
import numpy as np
import shutil
from loguru import logger
import click


@click.command()
@click.argument('input_datadir', type=click.Path(exists=True))
@click.argument('output_datadir', type=click.Path())
def main(input_datadir: str, output_datadir: str):
    classes_dir = ['mask_weared_incorrect', 'with_mask', 'without_mask']

    test_ratio = 0.20

    for cls in classes_dir:
        os.makedirs(output_datadir + 'train/' + cls)
        os.makedirs(output_datadir + 'test/' + cls)

        src = input_datadir + cls

        all_filenames = os.listdir(src)
        np.random.shuffle(all_filenames)
        train_filenames, test_filenames = np.split(np.array(all_filenames),
                                                   [int(len(all_filenames) * (1 - test_ratio))])

        train_filenames = [src + '/' + name for name in train_filenames.tolist()]
        test_filenames = [src + '/' + name for name in test_filenames.tolist()]

        print("*****************************")
        print('Total images: ', len(all_filenames))
        print('Training: ', len(train_filenames))
        print('Testing: ', len(test_filenames))
        print("*****************************")

        for name in train_filenames:
            shutil.copy(name, output_datadir + 'train/' + cls)

        for name in test_filenames:
            shutil.copy(name, output_datadir + 'test/' + cls)
    logger.info("Copying Done!")


if __name__ == '__main__':
    main()
