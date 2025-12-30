import os
import torch
import numpy as np
import random
import argparse

import warnings
warnings.filterwarnings("ignore")

from trainer.trainer import Trainer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset, DataLoader
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from utils.io_utils import load_yaml_config, instantiate_from_config
from model.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from data.build_dataloader import build_dataloader, build_dataloader_cond

seq_len = 96

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Parameters:
    - seed (int): The seed value.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Additional steps for CuDNN backend
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process configuration and directories.")
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the configuration file.')
    parser.add_argument('--save_dir', type=str, default='./forecasting_exp',
                        help='Directory to save experiment results.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Specify which GPU to use.')
    
    ## I added these here for wandb logging
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='forecasting with moving diffusion')
    parser.add_argument('--wandb_name', type=str, default=None, help='Run name.')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[], help='Run tags.')
    
    args = parser.parse_args()
    return args

def run(args):
    set_seed(2023)
    configs = load_yaml_config(args.config_path)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    model = instantiate_from_config(configs['model']).to(device)
    model.fast_sampling = True

    train_info = build_dataloader(configs, args)
    trainer = Trainer(config=configs, args=args, model=model, dataloader={'dataloader': train_info['dataloader']})
    trainer.train()

    args.mode = 'predict'
    args.pred_len = seq_len
    test_info = build_dataloader_cond(configs, args)

    sample, real_ = trainer.sample_forecast(test_info['dataloader'], shape=[args.pred_len, test_info['dataset'].samples.shape[-1]])

    mse = mean_squared_error(sample.reshape(-1), real_.reshape(-1))
    mae = mean_absolute_error(sample.reshape(-1), real_.reshape(-1))
    print(mse, mae)


if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(args.save_dir, exist_ok=True)
    run(args)
