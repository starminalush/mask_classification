import os
from typing import Dict, List

import click
import mlflow
import torch
import yaml
from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch import nn
from torchvision import datasets
from torchvision import transforms

load_dotenv()

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("validation_dataset", type=click.Path(exists=True))
@click.argument("data_type", type=click.STRING)
@click.argument("config_path", type=click.Path(exists=True))
def main(model_path: str, validation_dataset: str, data_type: str, config_path: str):
    with open(config_path, "r", encoding="utf-8") as stream:
        config: Dict = yaml.safe_load(stream)
    model: nn.Module = torch.load(
        os.path.join(model_path, f"{config['model_name']}_{data_type}/data/model.pth")
    )
    model = model.to("cuda")

    data_transforms: transforms.Compose = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_dataset: datasets.ImageFolder = datasets.ImageFolder(
        validation_dataset, data_transforms
    )

    logger.info(image_dataset.classes)

    image_dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        image_dataset, batch_size=1, num_workers=1
    )

    y_predicted: List = []
    y_gt: List = []
    with torch.no_grad():
        for data in image_dataloader:
            images, labels = data
            images: torch.Tensor = images.to("cuda")
            labels: torch.Tensor = labels.to("cuda")
            outputs: torch.Tensor = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_gt.append(labels.item())
            y_predicted.append(predicted.item())

    accuracy: float = accuracy_score(y_gt, y_predicted)
    f1: float = f1_score(y_gt, y_predicted, average="weighted")
    precision: float = precision_score(y_gt, y_predicted, average="weighted")
    recall: float = recall_score(
        y_gt, y_predicted, average="weighted", zero_division=True
    )
    logger.info(f"data type {data_type}")
    logger.debug(f"accuracy: {accuracy}")
    logger.debug(f"f1: {f1}")
    logger.debug(f"precision: {precision}")
    logger.debug(f"recall: {recall}")

    experiment_id: str = mlflow.set_experiment(config["model_name"]).experiment_id
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("dataset_type", data_type)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)


if __name__ == "__main__":
    main()
