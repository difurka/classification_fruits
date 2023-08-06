"""Load datasets with transforms."""

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

MAIN_DIR = '.'

transform=transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(40),              # resize shortest side
        transforms.CenterCrop(40),          # crop longest side
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
])

def get_trainset() -> ImageFolder:
    """
    Creating dataset for test for training.

    Returns:
        dataset for test
    """
    train_dir = MAIN_DIR + '/data/train'
    return ImageFolder(train_dir, transform=transform)

def get_valset() -> ImageFolder:
    """
    Creating dataset for validation while training.

    Returns:
        dataset for validation
    """
    val_dir = MAIN_DIR + '/data/validation'
    return ImageFolder(val_dir, transform=transform)

def get_testset() -> ImageFolder:
    """
    Creating dataset for test.

    Returns:
        dataset for test
    """
    test_dir = MAIN_DIR + '/data/test'
    return ImageFolder(test_dir, transform=transform)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = get_valset()
