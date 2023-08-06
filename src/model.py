"""Создание модели."""

from collections.abc import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights

import wandb
from src.utils import accuracy


class ImageClassificationBase(nn.Module):
    """
    Общий класс для модели.
     
    С методами обучения, валидации, вычисления средней accuracy
    """
    def training_step(self, batch: list, device: Callable) -> float:
        """
        Обучение на одном баче.

        Args:
            batch: один батч
            device: 'cpu' или 'cuda'
        Returns:
            значение функции потерь
        """
        images, labels = batch 
        images, labels = images.to(device), labels.to(device)
        out = self(images)                   # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss
    
    def validation_step(self, batch: list, device: Callable) -> dict:
        """
        Валидация на одном баче.

        Args:
            batch: один батч
            device: 'cpu' или 'cuda'
        Returns:
            val_loss и val_acc
        """
        images, labels = batch 
        images, labels = images.to(device), labels.to(device)
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs: list) -> dict:
        """
        Вычисление средних значений функции потерь и accurancy
        на валидационной выборке.

        Args:
            outputs: результат выданный моделью

        Returns: 
            словарь с loss и accuracy
        """
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        wandb.log({'val_loss_epoch': epoch_loss, 'val_acc_epoch': epoch_acc})
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

class VggModel(ImageClassificationBase):
    """Класс для модели на основе VGG16."""
    def __init__(self, number_output_classes: int):
        """
        Конструктор модели.

        Args:
            number_output_classes: количество классов, которые должны определяться
        """
        super().__init__()
        self.network = self.build_network(number_output_classes)

    def forward(self, xb):
        return self.network(xb)

    def build_network(self, number_output_classes: int) -> nn.Module:
        """
        Создание модели.

        За основу взята VGG16, в ней веса не будут меняться.
        Добавляются полносвязные слои, в ней веса при обучении меняются. 

        Args:
            number_output_classes: количество классов, которые должны определяться

        Returns:
            модель
        """
        model_extractor = models.vgg16(weights=VGG16_Weights.DEFAULT)

        # замораживаем параметры (веса)
        for param in model_extractor.parameters():
            param.requires_grad = False

        # num_features -- это размерность вектора фич, поступающего на вход FC-слою
        num_features = 25088
        # Заменяем Fully-Connected слой на наш линейный классификатор
        model_extractor.classifier =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 10000), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(10000, 3000), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(3000, 1000), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 500), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(500, number_output_classes)
        )
        return model_extractor
