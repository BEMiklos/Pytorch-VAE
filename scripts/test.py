import os
import sys
from pathlib import Path

import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import lightning as L
import numpy as np
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from vae import VAE


def main():
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize VAE model
    vae = VAE(config_file=ROOT / "vae/model/vae_128_1.0.yml").to(device)

    # Initialize TensorBoard logger
    logger = TensorBoardLogger(ROOT / "logs", name="vae_experiment")

    # Define dataset and dataloader
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR10 dataset
    test_dataset = datasets.CIFAR10(root=ROOT / 'data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize Lightning Trainer
    trainer = L.Trainer(max_epochs=250,
                        logger=logger,
                        callbacks=[EarlyStopping(monitor="val_loss",
                                                 mode="min",
                                                 patience=12,
                                                 verbose=True)],
                        default_root_dir="./checkpoints")

    # Test the model
    trainer.test(vae, dataloaders=test_loader)


if __name__ == "__main__":
    main()