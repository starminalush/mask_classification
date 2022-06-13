import os
import shutil
from typing import Dict

import click
import mlflow
import mlflow.pytorch
import yaml
from dotenv import load_dotenv
from loguru import logger

from trainer import Trainer

load_dotenv()
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.argument("dataset_type", type=click.STRING)
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path())
def train(dataset_path: str, config_path: str, dataset_type: str, model_path: str):
    """
    train model on train data and test in time of training on training data
    @param dataset_path: dataset for train and test model
    @type dataset_path:str
    @param config_path: neural network config
    @type config_path: str
    @param dataset_type: dataset type(internal, external, both)
    @type dataset_type: str
    @param model_path: local model path
    @type model_path:str
    """
    with open(config_path, "r", encoding="utf-8") as stream:
        config: Dict = yaml.safe_load(stream)
    trainer: Trainer = Trainer(config=config, dataset_path=dataset_path)
    logger.info("---DATASET INFO---")
    logger.info(f'train image size: {len(trainer.dataloaders["train"].dataset)}')
    logger.info(f'test image size: {len(trainer.dataloaders["test"].dataset)}')
    logger.info(f"data type {dataset_type}")

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

        for i in range(len(test_loss_history)):
            mlflow.log_metric("loss", test_loss_history[i])
            mlflow.log_metric("accuracy", test_acc_history[i])


if __name__ == "__main__":
    train()
