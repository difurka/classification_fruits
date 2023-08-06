"""Functions for training and validation."""

import os
import random
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import wandb
from src.prepire_data import get_trainset, get_valset

MAIN_DIR = '.'

def seed_everything(seed: int):
    """
    Make default settings for random values.

    Args:
        seed (int): seed for random
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    # будет работать - если граф вычислений не будет меняться во время обучения
    torch.backends.cudnn.benchmark = True  # оптимизации


def accuracy(outputs: list, labels: list) -> float:
    """
    Calculate accuracy. 

    Args:
        outputs: result of model
        labels: true result
    Returns:
        number
    """
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, device: Callable) -> dict:
    """
    Evaluate result that model gives.
    
    Args:
        model (nn.Module): model that is used
        val_loader (DataLoader): dataloader for validation
        device (Callable): 'cpu' or 'cuda'
    Returns:
        val_loss and val_accuracy
    """
    model.eval()
    outputs = [model.validation_step(batch, device) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def model_learning(epochs: int,  lr: float, batch_size: int, model: nn.Module, opt_func: Callable, device: Callable):
    """
    Make learning of model for epochs.

    Args:
        epochs: number of epochs
        lr: learning rate
        batch_size: size of batch    
        model: current model
        opt_func: optimize function
        device: set 'cpu' or 'cuda'
    """
    trainset = get_trainset()
    valset = get_valset()
    optimizer = opt_func(model.parameters(), lr)
    train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size, num_workers=2)
    best_val_loss = 100
    best_val_acc = 0
    for epoch in range(epochs):
        if epoch%10 == 9:
            lr/=10
            optimizer = opt_func(model.parameters(), lr)
        # Training Phase 
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch, device)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader, device)

        if best_val_loss >= result['val_loss'] and best_val_acc <= result['val_acc']:
            if (os.path.exists(MAIN_DIR + '/outs') == False):
                os.mkdir(MAIN_DIR + '/outs')
            best_val_loss = result['val_loss'] 
            best_val_acc = result['val_acc'] 
            torch.save(model.state_dict(), MAIN_DIR + '/outs/best_model.pth')

        result['train_loss'] = torch.stack(train_losses).mean().item()
        wandb.log({"train_loss_epoch": result['train_loss']})
 