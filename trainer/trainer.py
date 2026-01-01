import os
import sys
import time
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from utils.io_utils import instantiate_from_config, get_model_parameters_info

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, val_dataloader=None, logger=None, wandb_run=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device

        # NOTE: these are actually "steps" in your loop
        self.train_num_steps = config["trainer"]["max_epochs"]
        self.gradient_accumulate_every = config["trainer"]["gradient_accumulate_every"]
        self.save_cycle = config["trainer"]["save_cycle"]

        self.dl = cycle(dataloader["dataloader"])
        self.val_dataloader = val_dataloader

        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger
        self.wandb_run = wandb_run

        self.history = {"train_loss": []}

        # validate every N steps (only if val loader exists)
        self.val_interval = getattr(self.args, "val_interval", 100)

        self.results_folder = Path(config["trainer"]["results_folder"] + f"_{model.seq_length}")
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config["trainer"].get("base_lr", 1.0e-4)
        ema_decay = config["trainer"]["ema"]["decay"]
        ema_update_every = config["trainer"]["ema"]["update_interval"]

        self.opt = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=start_lr,
            betas=(0.9, 0.96),
        )
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config["trainer"]["scheduler"]
        sc_cfg["params"]["optimizer"] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))

        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info(f"Save current model to {self.results_folder / f'checkpoint-{milestone}.pt'}")

        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f"checkpoint-{milestone}.pt"))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info(f"Resume from {self.results_folder / f'checkpoint-{milestone}.pt'}")

        data = torch.load(str(self.results_folder / f"checkpoint-{milestone}.pt"), map_location=self.device)
        self.model.load_state_dict(data["model"])
        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])
        self.milestone = milestone

    def train(self):
        device = self.device
        local_step = 0

        tic = time.time()
        if self.logger is not None:
            name = getattr(self.args, "wandb_name", None) or "run"
            self.logger.log_info(f"{name}: start training...", check_primary=False)

        with tqdm(initial=local_step, total=self.train_num_steps) as pbar:
            while local_step < self.train_num_steps:
                total_loss = 0.0

                # ---- gradient accumulation ----
                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.dl)

                    if isinstance(batch, (tuple, list)):
                        x = batch[0].to(device)
                    else:
                        x = batch.to(device)

                    loss = self.model(x, target=x)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += float(loss.item())

                # ---- optimizer step ----
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()

                # ---- increment steps FIRST (so step=1 is first step) ----
                self.step += 1
                local_step += 1

                # ---- EMA update ----
                self.ema.update()

                # ---- validation + scheduler (only if val exists) ----
                val_loss = None
                if self.val_dataloader is not None and (self.step % self.val_interval == 0):
                    val_loss = self.compute_val_loss(self.val_dataloader)

                    # ReduceLROnPlateau-like expects a metric
                    try:
                        self.sch.step(val_loss)
                    except TypeError:
                        # schedulers like StepLR, CosineAnnealingLR
                        self.sch.step()
                else:
                    # if you use a scheduler that needs stepping every step, do it here.
                    # For ReduceLROnPlateauWithWarmup, you usually want metric-driven stepping.
                    pass

                # ---- tqdm + history ----
                pbar.set_description(f"loss: {total_loss:.6f}")
                self.history["train_loss"].append(total_loss)

                # ---- wandb logging ----
                if self.wandb_run is not None:
                    lr = self.opt.param_groups[0]["lr"]
                    log_dict = {"train/loss": total_loss, "train/lr": lr}
                    if val_loss is not None:
                        log_dict["val/loss"] = val_loss
                    self.wandb_run.log(log_dict, step=self.step)

                # ---- save checkpoints ----
                if self.step % self.save_cycle == 0:
                    self.milestone += 1
                    self.save(self.milestone)

                # ---- optional logger ----
                if self.logger is not None and self.step % self.log_frequency == 0:
                    self.logger.add_scalar(tag="train/loss", scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print("training complete")
        if self.logger is not None:
            self.logger.log_info(f"Training done, time: {time.time() - tic:.2f}")

    @torch.no_grad()
    def compute_val_loss(self, dataloader):
        self.model.eval()
        losses = []

        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)

            loss = self.model(x, target=x)
            losses.append(float(loss.item()))

        self.model.train()
        return float(np.mean(losses))

    @torch.no_grad()
    def sample_forecast(self, raw_dataloader, shape=None):
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])

        for batch in raw_dataloader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                x, _t_m = batch
                x = x.to(self.device)
            else:
                x = batch.to(self.device)

            sample = self.ema.ema_model.generate_mts(x)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x[:, shape[0] :, :].detach().cpu().numpy()])
            torch.cuda.empty_cache()

        return samples, reals

    @torch.no_grad()
    def evaluate_forecast(self, dataloader, shape):
        samples, reals = self.sample_forecast(dataloader, shape=shape)
        mse = float(((samples - reals) ** 2).mean())
        mae = float(np.abs(samples - reals).mean())

        if self.wandb_run is not None:
            logs = {"eval/mse": mse, "eval/mae": mae}

            try:
                import matplotlib.pyplot as plt
                import wandb as _wandb

                i, d = 0, 0
                plt.figure(figsize=(8, 3))
                plt.plot(reals[i, :, d], label="GT")
                plt.plot(samples[i, :, d], label="Pred")
                plt.legend()
                plt.tight_layout()
                logs["forecast/example"] = _wandb.Image(plt)
                plt.close()
            except Exception:
                pass

            self.wandb_run.log(logs, step=self.step)

        return {"mse": mse, "mae": mae}
