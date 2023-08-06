"""Загрузка датасета и весов на W&B."""

import os

import wandb

MAIN_DIR = '.'
def load_to_WB():
    """ 
    Загрузить датасет на W&B.
     
    Загружаем датасет на сайт, используем эту функцию только один раз.
    """
    DIR = './data/'

    run = wandb.init(project="metric_learning", job_type="dataset")
    if (os.path.exists(DIR + 'metric_learning.zip')):
        dataset = wandb.Artifact('fruits-dataset', type='dataset')
        dataset.add_file(DIR + 'metric_learning.zip')
        run.log_artifact(dataset)
    

    if (os.path.exists(MAIN_DIR + '/outs/best_model.pth')):
        wandb.save(MAIN_DIR + '/outs/best_model.pth')
    wandb.finish()


if __name__ == '__main__':
    load_to_WB()