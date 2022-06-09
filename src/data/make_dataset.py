import glob
import xml.etree.ElementTree as ET

import click
import pandas as pd
from loguru import logger


@click.command()
@click.argument("annotation_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(annotation_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    dataset = {
        "xmin": [],
        "ymin": [],
        "xmax": [],
        "ymax": [],
        "name": [],
        "file": [],
        "width": [],
        "height": [],
    }

    for anno in glob.glob(annotation_filepath + "/*.xml"):
        tree = ET.parse(anno)

        for elem in tree.iter():
            if "size" in elem.tag:
                for attr in list(elem):
                    if "width" in attr.tag:
                        width = int(round(float(attr.text)))
                    if "height" in attr.tag:
                        height = int(round(float(attr.text)))

            if "object" in elem.tag:
                for attr in list(elem):

                    if "name" in attr.tag:
                        name = attr.text
                        dataset["name"] += [name]
                        dataset["width"] += [width]
                        dataset["height"] += [height]
                        dataset["file"] += [anno.split("/")[-1][0:-4]]

                    if "bndbox" in attr.tag:
                        for dim in list(attr):
                            if "xmin" in dim.tag:
                                xmin = int(round(float(dim.text)))
                                xmin = xmin if xmin > 0 else 0
                                dataset["xmin"] += [xmin]
                            if "ymin" in dim.tag:
                                ymin = int(round(float(dim.text)))
                                ymin = ymin if ymin > 0 else 0
                                dataset["ymin"] += [ymin]
                            if "xmax" in dim.tag:
                                xmax = int(round(float(dim.text)))
                                xmax = xmax if xmax > 0 else 0
                                dataset["xmax"] += [xmax]
                            if "ymax" in dim.tag:
                                ymax = int(round(float(dim.text)))
                                ymax = ymax if ymax > 0 else 0
                                dataset["ymax"] += [ymax]
    data = pd.DataFrame(dataset)
    data.to_csv(output_filepath, index=False)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    main()
