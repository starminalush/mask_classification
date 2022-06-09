import copy
import math
import time
import typing

import timm
import torch
import torchvision.models
from loguru import logger
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm


class Trainer:
    def __init__(self, model_name: str, num_epochs: int, dataloaders, device="cuda"):
        """
        create trainer class object
        @param model_name: model name. Available model names: mobilenet, resnet, vit_timm
        @type model_name: str
        @param num_epochs: amount of epochs
        @type num_epochs: int
        @param dataloaders: train and test dataloaders
        @type dataloaders: dict
        @param device: nvidia gpu device
        @type device: str
        """
        self.model: nn.Module = Trainer.load_model(model_name=model_name)
        self.num_epochs: int = num_epochs
        self.dataloaders: typing.Dict = dataloaders
        self.device = torch.device(device)
        # default criterion
        self.criterion = nn.CrossEntropyLoss()
        # default optimizer
        self.optimizer:optim.Optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # default lr
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=7, gamma=0.1)
        logger.error(type(self.scheduler))
        logger.error(type(self.criterion))
        logger.error(self.device)

    @staticmethod
    def load_model(model_name: str, pretrained=True, device="cuda") -> nn.Module:
        model = None
        if model_name == "resnet":
            model: nn.Module = torchvision.models.resnet50(pretrained=pretrained)
            num_ftrs: int = model.fc.in_features
            model.fc: nn.Linear = nn.Linear(num_ftrs, 2)
        if model_name == "mobilenet":
            model: nn.Module = torchvision.models.mobilenet_v2(pretrained=pretrained)
            for params in list(model.parameters())[0:-5]:
                params.requires_grad = False
            num_ftrs: int = model.classifier[-1].in_features
            model.classifier: nn.Sequential = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=num_ftrs, out_features=2, bias=True),
            )
        if model_name == "vit_timm":
            model: nn.Module = timm.create_model(" ", pretrained=True)
            model.head: nn.Linear = nn.Linear(model.head.in_features, 2)
        model = model.to(device)
        return model

    def train(self):
        since:time = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = math.inf
        test_acc_history = []
        test_loss_history = []
        best_acc = 0.0

        for epoch in tqdm(range(self.num_epochs)):
            logger.debug(f"Epoch {epoch}/{self.num_epochs - 1}")
            logger.debug("-" * 10)

            for phase in ["train", "test"]:
                if phase == "train":
                    self.scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()

                current_loss = 0.0
                current_accuracy = 0.0

                for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    if i % 100 == 99:
                        logger.debug(
                            f"epoch {epoch}, loss {current_loss / (i * inputs.size(0))}"
                        )

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                    current_loss += loss.item() * inputs.size(0)
                    current_accuracy += torch.sum(preds == labels.data)

                epoch_loss = current_loss / len(self.dataloaders[phase].dataset)
                epoch_accuracy = current_accuracy.double() / len(
                    self.dataloaders[phase].dataset
                )

                logger.info(
                    f"phase {phase}, loss {epoch_loss}, acuuracy {epoch_accuracy}"
                )

                if phase == "test":
                    test_acc_history.append(epoch_accuracy)
                    test_loss_history.append(epoch_loss)
                    if epoch_loss < best_loss:
                        logger.debug(f"found best model")
                        logger.debug(
                            f"best model record loss: {epoch_loss}, previous record loss: {best_loss}"
                        )
                        best_loss = epoch_loss
                        best_acc = epoch_accuracy
                        best_model_wts = copy.deepcopy(self.model.state_dict())
        time_elapsed = time.time() - since
        logger.info(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        logger.info(f"Best val Acc: {best_acc:.4f} Best val loss: {best_loss:.4f}")

        self.model.load_state_dict(best_model_wts)
        return (
            self.model,
            best_loss,
            best_acc,
            time_elapsed,
            test_acc_history,
            test_loss_history,
        )
