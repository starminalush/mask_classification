import os
import typing

import click
import mlflow
import mlflow.pytorch
import torch
import yaml
from dotenv import load_dotenv
from loguru import logger
from torchvision import transforms, datasets
from typing import List, Dict

from trainer import Trainer

load_dotenv()
# os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
# os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MINIO_URI")
# os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ROOT_USER")
# os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_ROOT_PASSWORD")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.argument("dataset_type", type=click.STRING)
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path())
def main(dataset_path: str, config_path: str, dataset_type: str, model_path: str):
    """
    train model
    @param dataset_path: dataset path(external, internal or both)
    @type dataset_path: str
    @param config_path: config for neural network in .yaml format
    @type config_path: str
    @param dataset_type: dataset type(external, internal or both)
    @type dataset_type: str
    @param model_path: path for model saving
    @type model_path: str
    """
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    # create dataset and dataloaders
    data_transforms:Dict = {
        "train": transforms.Compose(
            [
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(
                    224, scale=(0.96, 1.0), ratio=(0.95, 1.05)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets: Dict = {
        x: datasets.ImageFolder(root=os.path.join(dataset_path, x), target_transform=data_transforms[x])
        for x in ["train", "test"]
    }

    dataloaders: Dict = {
        x: torch.utils.data.DataLoader(
            dataset=image_datasets[x], batch_size=4, shuffle=True, num_workers=4
        )
        for x in ["train", "test"]
    }

    class_names: List = image_datasets["train"].classes
    logger.info("---DATASET INFO---")
    logger.info(f"classes {class_names}")
    logger.info(f'train image size: {len(dataloaders["train"].dataset)}')
    logger.info(f'test image size: {len(dataloaders["test"].dataset)}')
    logger.info(f"dataset type {dataset_type}")

    trainer: Trainer = Trainer(
        model_name=config["model_name"],
        num_epochs=config["num_epochs"],
        dataloaders=dataloaders,
    )
    experiment_id: str = mlflow.set_experiment(config["model_name"]).experiment_id
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("dataset", dataset_path)
        mlflow.log_param("model name", config["model_name"])
        mlflow.log_param("number of classes", config["num_classes"])
        mlflow.log_param("batch size", config["batch_size"])
        mlflow.log_param("epochs", config["num_epochs"])
        (
            model,
            best_loss,
            best_acc,
            training_time,
            test_acc_history,
            test_loss_history,
        ) = trainer.train()
        mlflow.log_param("Training time", training_time)
        mlflow.log_param("Best val Acc", float(best_acc))
        mlflow.log_param("Best loss", best_loss)
        mlflow.pytorch.log_model(
            model, config["model_name"], registered_model_name=config["model_name"]
        )
        mlflow.pytorch.save_model(model, model_path)
        for i in range(len(test_loss_history)):
            mlflow.log_metric("loss", test_loss_history[i])
            mlflow.log_metric("accuracy", test_acc_history[i])


if __name__ == "__main__":
    main()
