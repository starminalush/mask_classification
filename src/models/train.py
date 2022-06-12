import os
import shutil
from typing import Dict, List

import click
import mlflow
import mlflow.pytorch
import onnx
import torch
import yaml
from dotenv import load_dotenv
from loguru import logger
from torchvision import transforms, datasets

from trainer import Trainer

load_dotenv()
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


def torch2onnx(model: torch.nn.Module, models_output: str, model_name: str):
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    model_save_path = os.path.join(models_output, f"{model_name}.onnx")
    torch.onnx.export(
        model.to("cpu"),
        x,
        model_save_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    onnx_model = onnx.load(model_save_path)
    return onnx_model


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.argument("dataset_type", type=click.STRING)
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path())
def main(dataset_path: str, config_path: str, dataset_type: str, model_path: str):
    with open(config_path, "r", encoding="utf-8") as stream:
        config: Dict = yaml.safe_load(stream)
    data_transforms: Dict = {
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
        x: datasets.ImageFolder(os.path.join(dataset_path, x), data_transforms[x])
        for x in ["train", "test"]
    }

    dataloaders: Dict = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=4, shuffle=True, num_workers=4
        )
        for x in ["train", "test"]
    }

    class_names: List = image_datasets["train"].classes
    logger.info("---DATASET INFO---")
    logger.info(f"classes {class_names}")
    logger.info(f'train image size: {len(dataloaders["train"].dataset)}')
    logger.info(f'test image size: {len(dataloaders["test"].dataset)}')
    logger.info(f"data type {dataset_type}")

    trainer: Trainer = Trainer(config=config, dataloaders=dataloaders)
    experiment_id: str = mlflow.set_experiment(config["model_name"]).experiment_id
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("dataset", dataset_path)
        mlflow.log_param("dataset type", dataset_type)
        mlflow.log_param("model name", config["model_name"])
        mlflow.log_param("pretrained", config["pretrained"])
        mlflow.log_param("number of classes", config["num_classes"])
        mlflow.log_param("batch size", config["batch_size"])
        mlflow.log_param("epochs", config["num_epochs"])
        mlflow.log_param("criterion", config["criterion"])
        mlflow.log_param("optimizer", config["optimizer"])
        mlflow.log_param("scheduler", config["scheduler"])
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
            model,
            f"{config['model_name']}_{dataset_type}",
            registered_model_name=f"{config['model_name']}_{dataset_type}",
        )
        if os.path.exists(
            os.path.join(model_path, f"{config['model_name']}_{dataset_type}")
        ):
            shutil.rmtree(
                os.path.join(model_path, f"{config['model_name']}_{dataset_type}")
            )
        mlflow.pytorch.save_model(
            model, os.path.join(model_path, f"{config['model_name']}_{dataset_type}")
        )
        onnx_model = torch2onnx(
            model=model,
            models_output=model_path,
            model_name=f"{config['model_name']}_{dataset_type}",
        )
        mlflow.onnx.log_model(
            onnx_model,
            f"{config['model_name']}_{dataset_type}.onnx",
            registered_model_name=f"{config['model_name']}_{dataset_type}.onnx",
        )
        for i in range(len(test_loss_history)):
            mlflow.log_metric("loss", test_loss_history[i])
            mlflow.log_metric("accuracy", test_acc_history[i])


if __name__ == "__main__":
    main()
