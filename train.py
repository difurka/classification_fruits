""" Обучение модели. """

import argparse
import os
from collections.abc import Callable

import torch

import configs.config as cfg
import wandb
from src.model import VggModel
from src.utils import *

MAIN_DIR = '.'

def  train_model( 
        config: dict,
        device: Callable = torch.device('cpu'),
    ):
    """
    Build all together: initialize the model,
    optimizer and loss function.

    Args:
        config(dict): set of hyperparameters
        device : set "cpu" or "cuda"
    """

    wandb.login()
    with wandb.init(project="metric_learning",config=config):
        config = wandb.config
        seed_everything(config.seed)
        train_dir = MAIN_DIR + '/data/test'
        device = get_default_device()
        count_classes = len(os.listdir(train_dir))
        model = VggModel(count_classes).to(device)
        opt_func = torch.optim.Adam
        model_learning(config.epochs, config.lr, config.batch_size, model,  opt_func, device)

def train():
    """ Определение гиперпараметров, запуск обучения. """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='total training epochs')
    parser.add_argument('--batch_size', type=int, help='total batch size')
    parser.add_argument('--seed', type=int, help='global training seed')
    parser.add_argument('--lr', type=float, help='global learning rate')
    args = parser.parse_args()

    config_for_training = dict(
        architecture = cfg.architecture,  
        dataset = cfg.dataset,
        epochs = args.epochs if (args.epochs) else cfg.epochs,
        batch_size = args.batch_size if (args.batch_size) else cfg.batch_size,
        lr= args.lr if (args.lr) else cfg.lr,
        seed = args.seed if (args.seed) else cfg.seed
    )
    train_model(config_for_training, device=device)

if __name__ == '__main__':
    train()
