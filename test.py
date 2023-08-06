"""Вычисление accuracy на тестовой выборке. Загружаются веса модели."""

import os

import torch
from torch.utils.data.dataloader import DataLoader

import wandb
from src.model import VggModel
from src.prepire_data import get_testset
from src.utils import evaluate

MAIN_DIR = '.'
def test():
    """Вычисление accuracy на тестовой выборке. Загружаются веса модели."""
    wandb.login()
    with wandb.init(project="metric_learning"):
        testset = get_testset()
        test_loader = DataLoader(testset, 4, num_workers=2)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_dir = MAIN_DIR + '/data/test'
        count_classes = len(os.listdir(train_dir))
        model = VggModel(count_classes).to(device)
        model.load_state_dict(torch.load(MAIN_DIR + '/src/weights/best_model.pth'))
        model.eval()
        evaluate(model, test_loader, device)

if __name__ == '__main__':
    test()