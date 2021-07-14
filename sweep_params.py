import wandb
import train_avse_frames

sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'conv_layer_size': {
            'values': [16, 32, 64]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4, 3e-4, 3e-5, 1e-5]
        },
        'optimizer': {
            'values': ['adam', 'nadam', 'sgd', 'rmsprop']
        },
        'activation': {
            'values': ['relu', 'tanh']
        },
        'objective_zeros': {
            'values': [True, False]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, entity="carl_m", project="AV-Fusion-AVSE")
wandb.agent(sweep_id, train_avse_frames.train)