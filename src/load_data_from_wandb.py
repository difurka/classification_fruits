"""Загрузка датасета с сайта W&B в папку artifacts и весов обученной модели."""

import os
import shutil
import zipfile

import wandb

MAIN_DIR = '.'
def load_from_WB():
    """ Загрузка датасета с сайта W&B. """
    
    DIR = MAIN_DIR + '/data/'
    WEIGHTS_DIR = MAIN_DIR + '/src/weights/'
    if (os.path.exists(DIR) == False):
                os.mkdir(DIR)
    if (os.path.exists(WEIGHTS_DIR) == False):
        os.mkdir(WEIGHTS_DIR)
    run = wandb.init(project="metric_learning")
    artifact = run.use_artifact('balakinakate2022/metric_learning/fruits-dataset:v0', type='dataset')
    artifact.download()

    artifact = run.use_artifact('balakinakate2022/metric_learning/fruits-weights:v0', type='weights')
    artifact.download()

    if (os.path.exists(MAIN_DIR + '/artifacts/fruits-dataset:v0/fruits.zip')):
        with zipfile.ZipFile(MAIN_DIR + '/artifacts/fruits-dataset:v0/fruits.zip', 'r') as zip_ref:
            zip_ref.extractall(DIR)
    elif (os.path.exists(MAIN_DIR + '/artifacts/fruits-dataset-v0/fruits.zip')):
        with zipfile.ZipFile(MAIN_DIR + '/artifacts/fruits-dataset-v0/fruits.zip', 'r') as zip_ref:
            zip_ref.extractall(DIR)
    
    if (os.path.exists(MAIN_DIR + '/artifacts/fruits-weights-v0/best_model.pth')):
        shutil.copyfile(MAIN_DIR + '/artifacts/fruits-weights-v0/best_model.pth', WEIGHTS_DIR + 'best_model.pth')
    if (os.path.exists(MAIN_DIR + '/artifacts/fruits-weights:v0/best_model.pth')):
        shutil.copyfile(MAIN_DIR + '/artifacts/fruits-weights:v0/best_model.pth', WEIGHTS_DIR + "best_model.pth")


if __name__ == '__main__':
    load_from_WB()