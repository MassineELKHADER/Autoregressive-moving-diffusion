import torch
import copy
from utils.io_utils import instantiate_from_config


def _build_loader(dataset_cfg, batch_size, shuffle, drop_last):
    dataset = instantiate_from_config(dataset_cfg)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=drop_last,
    )
    return loader, dataset

def build_dataloader(config, args):
    cfg = copy.deepcopy(config['dataloader'])

    cfg['train_dataset']['params']['output_dir'] = args.save_dir

    loader, dataset = _build_loader(
        cfg['train_dataset'],
        batch_size=cfg['batch_size'],
        shuffle=cfg['shuffle'],
        drop_last=cfg['shuffle'],
    )

    return {'dataloader': loader, 'dataset': dataset}

def build_val_dataloader(config, args):
    cfg = copy.deepcopy(config)

    cfg["dataloader"]["train_dataset"] = cfg["dataloader"]["val_dataset"]

    dataset = instantiate_from_config(cfg["dataloader"]["train_dataset"])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    return {"dataloader": loader, "dataset": dataset}

def build_dataloader_cond(config, args):
    cfg = copy.deepcopy(config['dataloader'])

    cfg['test_dataset']['params']['output_dir'] = args.save_dir

    if args.mode == 'infill':
        cfg['test_dataset']['params']['missing_ratio'] = args.missing_ratio
    elif args.mode == 'predict':
        cfg['test_dataset']['params']['predict_length'] = args.pred_len
    else:
        raise ValueError(f"Unknown mode for args: {args.mode}")

    loader, dataset = _build_loader(
        cfg['test_dataset'],
        batch_size=cfg['sample_size'],
        shuffle=False,
        drop_last=False,
    )

    return {'dataloader': loader, 'dataset': dataset}


if __name__ == '__main__':
    pass

