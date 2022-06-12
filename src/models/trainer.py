import copy
import math
import time
from typing import Dict, List
from typing import OrderedDict

import timm
import torch
import torchvision.models
from loguru import logger
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

"""
available model name: resnet, mobilenet, vit_timm
"""


class Trainer:
    def __init__(self, config: dict, dataloaders, device="cuda"):
        self.model: nn.Module = self.load_model(
            model_name=config["model_name"],
            model_classes=config["num_classes"],
            pretrained=config["pretrained"],
        )
        self.num_epochs: int = config["num_epochs"]
        self.dataloaders: Dict = dataloaders
        self.device: torch.device = torch.device(device)
        self.criterion: nn.modules.loss = self.load_criterion(
            criterion_name=config["criterion"]
        )
        self.optimizer: optim.optimizer = self.load_optimizer(
            optimizer_name=config["optimizer"]
        )
        self.scheduler: optim.lr_scheduler = self.load_scheduler(
            scheduler_name=config["scheduler"]
        )

    def load_model(
        self, model_name: str, model_classes: int, pretrained=True, device="cuda"
    ):
        model = None
        if model_name == "resnet":
            model: nn.Module = torchvision.models.resnet50(pretrained=pretrained)
            num_ftrs: int = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, model_classes)
        if model_name == "mobilenet":
            model: nn.Module = torchvision.models.mobilenet_v2(pretrained=pretrained)
            for params in list(model.parameters())[0:-5]:
                params.requires_grad = False
            num_ftrs: int = model.classifier[-1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=num_ftrs, out_features=model_classes, bias=True),
            )
        if model_name == "vit_timm":
            model: nn.Module = timm.create_model(
                "vit_base_patch16_224", pretrained=True
            )
            model.head = nn.Linear(model.head.in_features, model_classes)
        model = model.to(device)
        return model

    def load_criterion(self, criterion_name: str):
        if criterion_name == "cross_entropy_loss":
            criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        return criterion

    def load_optimizer(self, optimizer_name):
        if optimizer_name == "sgd":
            optimizer: optim.SGD = optim.SGD(
                params=self.model.parameters(), lr=0.001, momentum=0.9
            )
        if optimizer_name == "adam":
            optimizer: optim.Adam = optim.Adam(params=self.model.parameters(), lr=0.01)
        return optimizer

    def load_scheduler(self, scheduler_name):
        if scheduler_name == "step_lr":
            scheduler: lr_scheduler.StepLR = lr_scheduler.StepLR(
                self.optimizer, step_size=7, gamma=0.1
            )
        if scheduler_name == "multistep_lr":
            scheduler: lr_scheduler.MultiStepLR = lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[6, 8, 9], gamma=0.1
            )
        return scheduler

    def train(self):
        since: float = time.time()
        best_model_wts: OrderedDict = copy.deepcopy(self.model.state_dict())
        best_loss: float = math.inf
        test_acc_history: List = []
        test_loss_history: List = []
        best_acc: float = 0.0

        for epoch in tqdm(range(self.num_epochs)):
            logger.debug(f"Epoch {epoch}/{self.num_epochs - 1}")
            logger.debug("-" * 10)

            for phase in ["train", "test"]:
                if phase == "train":
                    self.scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()

                current_loss: float = 0.0
                current_accuracy: float = 0.0

                for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs: torch.Tensor = inputs.to(self.device)
                    labels: torch.Tensor = labels.to(self.device)

                    self.optimizer.zero_grad()

                    if i % 100 == 99:
                        logger.debug(
                            f"epoch {epoch}, loss {current_loss / (i * inputs.size(0))}"
                        )

                    with torch.set_grad_enabled(phase == "train"):
                        outputs: torch.Tensor = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss: torch.Tensor = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                    current_loss += loss.item() * inputs.size(0)
                    current_accuracy += torch.sum(preds == labels.data)

                epoch_loss: float = current_loss / len(self.dataloaders[phase].dataset)
                epoch_accuracy: float = current_accuracy / len(
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
        time_elapsed: float = time.time() - since
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
