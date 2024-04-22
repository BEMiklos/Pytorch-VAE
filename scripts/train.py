import os
import sys
from pathlib import Path

import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import lightning as L
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger

FILE = Path(__file__).resolve()
SCRIPTS = FILE.parents[0]
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from vae import VAE
from utils import Parser as P

def main():
    config = P.read_config(config_file= SCRIPTS / "config/base.yml")

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize VAE model
    vae_config_file = config['model']['config']
    vae = VAE(config_file= ROOT / vae_config_file).to(device)

    # Initialize optimizer
    optimizer_type = config['trainer']['optimizer']['type']
    optimizer_args = config['trainer']['optimizer']['args']
    vae.optimizer = getattr(torch.optim, optimizer_type)(vae.parameters(), **optimizer_args)

    # Initialize TensorBoard logger
    logger_type = config['trainer']['logger']['type']
    logger_args = config['trainer']['logger']['args']
    logger = getattr(L.pytorch.loggers, logger_type)(**logger_args)
    logger = EarlyStopping(monitor="val_loss", mode="min", patience=12, verbose=True)

    # Define dataset and dataloader
    train_transform = P.transforms(config['data']['train_transform'])
    # Define dataset and dataloader
    test_transform = P.transforms(config['data']['test_transform'])

    # Load dataset and apply transforms
    dataset_name = config['data']['dataset']
    val_split = config['data']['val_split']
    train_batch_size = config['data']['train_batch_size']
    val_batch_size = config['data']['val_batch_size']
    test_batch_size = config['data']['test_batch_size']
    num_workers = config['data']['num_workers']

    # Load CIFAR10 dataset
    train_dataset = datasets.__dict__[dataset_name](root= ROOT / 'data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.__dict__[dataset_name](root= ROOT / 'data', train=False, download=True, transform=test_transform)
    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    callbacks = P.callbacks(config['trainer']['callbacks'])

    # Initialize Lightning Trainer
    trainer = L.Trainer(max_epochs=config['trainer']['max_epochs'],
                        logger=logger,
                        callbacks=callbacks,
                        default_root_dir="./checkpoints")

    # Train the model
    trainer.fit(vae, train_loader, val_loader)

    # Save the model into the logdir by version
    vae.save_model(logger.log_dir)

    # Test the model
    trainer.test(vae, dataloaders=test_loader)

if __name__ == "__main__":
    main()