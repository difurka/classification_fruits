"""Подбор гиперпараметров с помощью sweep W&B."""

import wandb

def sweep_func():
    """Подбор гиперпараметров с помощью sweep W&B."""
    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'maximize', 
            'name': 'val_acc_epoch'
            },
        'parameters': {
            'batch_size': {'values': [16, 32, 64]},
            'epochs': {'values': [5, 10, 15]},
            'lr': {'max': 0.1, 'min': 0.0001}
        }
    }
    wandb.init(project="metric_learning")
    sweep_id = wandb.sweep(sweep=sweep_configuration)

    wandb.agent(sweep_id, count=5)


if __name__ == '__main__':
    sweep_func()