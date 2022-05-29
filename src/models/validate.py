import os

import click
import mlflow
import torch
import yaml
from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torchvision import datasets
from torchvision import transforms

load_dotenv()

os.environ['MLFLOW_TRACKING_URI'] = "http://0.0.0.0:5000"
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://0.0.0.0:9000"
os.environ['AWS_ACCESS_KEY_ID'] = "s3keys3key"
os.environ['AWS_SECRET_ACCESS_KEY'] = "s3keys3key"
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('validation_dataset', type=click.Path(exists=True))
@click.argument('config_path', type=click.Path(exists=True))
def main(model_path: str, validation_dataset: str, config_path: str):
    with open(config_path, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    model = torch.load(model_path)
    model = model.to('cuda')

    data_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_dataset = datasets.ImageFolder(validation_dataset,
                                         data_transforms)

    image_dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1,
                                                   shuffle=True, num_workers=1)

    y_predicted = []
    y_gt = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in image_dataloader:
            images, labels = data
            images = images.to('cuda')
            labels = labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_gt.append(labels.size(0))
            y_predicted.append(predicted.item())

    accuracy = accuracy_score(y_gt, y_predicted)
    f1 = f1_score(y_gt, y_predicted, average='weighted')
    precision = precision_score(y_gt, y_predicted, average='weighted')
    recall = recall_score(y_gt, y_predicted, average='weighted', zero_division=True)

    logger.debug(f'accuracy: {accuracy}')
    logger.debug(f'f1: {f1}')
    logger.debug(f'precision: {precision}')
    logger.debug(f'recall: {recall}')

    experiment_id = mlflow.set_experiment(config['model_name']).experiment_id
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('f1', f1)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)


if __name__ == "__main__":
    main()
