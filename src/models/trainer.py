import copy
import math
import time

import timm
import torch
import torchvision.models
from loguru import logger
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

'''
available model name: resnet, mobilenet, shufflenet
'''


class Trainer:
    def __init__(self, model_name: str, num_epochs: int, dataloaders, device='cuda'):
        self.model = Trainer.load_model(model_name=model_name)
        self.num_epochs = num_epochs
        self.dataloaders = dataloaders
        self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    @staticmethod
    def load_model(model_name: str, pretrained=True, device='cuda'):
        model = None
        if model_name == 'resnet':
            model = torchvision.models.resnet50(pretrained=pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
        if model_name == 'mobilenet':
            model = torchvision.models.mobilenet_v2(pretrained=pretrained)
            for params in list(model.parameters())[0:-5]:
                params.requires_grad = False
            num_ftrs = model.classifier[-1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=num_ftrs, out_features=2, bias=True)
            )
        if model_name == 'vit_timm':
            model = timm.create_model(" ", pretrained=True)
            model.head = nn.Linear(model.head.in_features, 2)
        model = model.to(device)
        return model

    def train(self):
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = math.inf
        test_acc_history = []
        test_loss_history = []
        best_acc = 0.

        for epoch in tqdm(range(self.num_epochs)):
            logger.debug(f'Epoch {epoch}/{self.num_epochs - 1}')
            logger.debug('-' * 10)

            for phase in ['train', 'test']:
                if phase == 'train':
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
                        logger.debug(f"epoch {epoch}, loss {current_loss / (i * inputs.size(0))}")

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    current_loss += loss.item() * inputs.size(0)
                    current_accuracy += torch.sum(preds == labels.data)

                epoch_loss = current_loss / len(self.dataloaders[phase].dataset)
                epoch_accuracy = current_accuracy.double() / len(self.dataloaders[phase].dataset)

                logger.info(f"phase {phase}, loss {epoch_loss}, acuuracy {epoch_accuracy}")

                if phase == 'test':
                    test_acc_history.append(epoch_accuracy)
                    test_loss_history.append(epoch_loss)
                    if epoch_loss < best_loss:
                        logger.debug(f'found best model')
                        logger.debug(f'best model record loss: {epoch_loss}, previous record loss: {best_loss}')
                        best_loss = epoch_loss
                        best_acc = epoch_accuracy
                        best_model_wts = copy.deepcopy(self.model.state_dict())
        time_elapsed = time.time() - since
        logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        logger.info(f'Best val Acc: {best_acc:.4f} Best val loss: {best_loss:.4f}')

        self.model.load_state_dict(best_model_wts)
        return self.model, best_loss, best_acc, time_elapsed, test_acc_history, test_loss_history
