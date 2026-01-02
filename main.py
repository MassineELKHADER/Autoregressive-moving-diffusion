import os
import random
import argparse
import warnings

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

import wandb

from trainer.trainer import Trainer
from utils.io_utils import load_yaml_config, instantiate_from_config
from data.build_dataloader import build_dataloader, build_val_dataloader, build_dataloader_cond

warnings.filterwarnings("ignore")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./forecasting_exp")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2023)

    # wandb
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="forecasting-with-moving-diffusion")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, nargs="*", default=[])

    return p.parse_args()


def run(args):
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    configs = load_yaml_config(args.config_path)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = instantiate_from_config(configs["model"]).to(device)
    model.fast_sampling = True

    # init wandb AFTER configs exist
    wandb_run = None
    if args.use_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            tags=args.wandb_tags,
            config=configs,
        )

    # train
    train_info = build_dataloader(configs, args)

    # val 
    val_info = build_val_dataloader(configs, args)
    assert len(val_info["dataset"]) > 0, "Validation set is empty."

    trainer = Trainer(
        config=configs,
        args=args,
        model=model,
        dataloader={"dataloader": train_info["dataloader"]},
        val_dataloader=val_info["dataloader"],
        wandb_run=wandb_run,
    )
    trainer.train()

    # eval / predict
    SEQ_LEN = configs["model"]["params"]["seq_length"]
    args.mode = "predict"
    args.pred_len = SEQ_LEN

    test_info = build_dataloader_cond(configs, args)
    feat_dim = 1
    shape = [args.pred_len, feat_dim]

    # use trainer eval (logs to wandb + returns metrics)
    metrics = trainer.evaluate_forecast(test_info["dataloader"], shape=shape)
    print(metrics)
    print("model.loss_type : ", model.loss_type)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
