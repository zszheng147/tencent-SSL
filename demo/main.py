import random

import logging
import hydra 
from omegaconf import DictConfig

import numpy as np

import torch
import torch.distributed as dist
from torch import nn, optim

from data import ReverbDataset
from model import SSLModel
from utils import train_one_epoch, evaluate

logger = logging.getLogger(__name__)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@hydra.main(config_path='config', config_name='default.yaml', version_base="1.2")
def main(cfg: DictConfig):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    setup_seed(0 + rank)

    train_dataset = ReverbDataset(
        cfg.data.train_sound_path,
        rir_type=cfg.data.rir_type,
        reverb_json=cfg.data.reverb_train_json,
        mode='train'
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.common.batch_size,
        shuffle=True,
        num_workers=cfg.common.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn
    )

    eval_dataset = ReverbDataset(
        cfg.data.eval_sound_path,
        rir_type=cfg.data.rir_type,
        reverb_json=cfg.data.reverb_eval_json,
        mode='eval'
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg.common.batch_size,
        shuffle=False,
        num_workers=cfg.common.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=eval_dataset.collate_fn
    )

    model = SSLModel()
    model = model.to(rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=cfg.optim.lr, 
                                 betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0.05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
    
    criterion = nn.CrossEntropyLoss()

    if cfg.common.mode == 'train':
        for epoch in range(cfg.common.max_epoch):
            total_loss = train_one_epoch(
                dataloader=train_loader, model=ddp_model, 
                optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                device=rank, epoch=epoch, config=cfg
            )
            if torch.cuda.device_count() > 1:
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            train_epoch_loss = total_loss.item() / len(train_loader)
            if rank == 0:
                logger.info(f'Epoch {epoch + 1}: loss={train_epoch_loss:.4f}')
                if (epoch + 1) % 10 == 0:
                    mae = evaluate(ddp_model, eval_loader, rank, epoch, config=cfg)
                    logger.info(f'MAE: {mae}')

        torch.save(ddp_model.module.state_dict(), 'model.pth')
    elif cfg.common.mode == 'eval':
        acc = evaluate(ddp_model, eval_loader, rank, epoch, config=cfg)
        if rank == 0:
            logger.info(f'Eval accuracy: {acc:.4f}')

    dist.destroy_process_group()

if __name__ == '__main__':
    main()