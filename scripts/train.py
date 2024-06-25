import os
import sys
from pathlib import Path

import torch
import lightning as L
import numpy as np

FILE = Path(__file__).resolve()
SCRIPTS = FILE.parents[0]
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from vae import VAE
from utils import Parser
from argparse import ArgumentParser

def main():
    # Argument parser
    parser = ArgumentParser(description="Train a VAE model")
    parser.add_argument('--config', type=str, default=SCRIPTS / "config/base.yml", help='Path to the configuration file')
    parser.add_argument('--model', type=str, default=None, help='Path to the model configuration file')
    parser.add_argument('--detached', type=bool, default=None, help='Wether to detach the classifier from the encoder')
    parser.add_argument('--kld', type=float, default=None, help='The weight of the KLD loss')
    parser.add_argument('--mse', type=float, default=None, help='The weight of the MSE loss')
    parser.add_argument('--class_weight', type=float, default=None, help='The weight of the classification loss')
    parser.add_argument('--output', type=str, default=None, help='Path to the model configuration file')
    parser.add_argument('--dataset', type=str, default=None, help='Pytorch dataset name')
    parser.add_argument('--val_split', type=float, default=None, help='Train-validation dataset split ratio')
    args = parser.parse_args()

    # Initialize parser
    p = Parser(config_file= args.config, args=args)

    config = p.config

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize VAE model
    vae = VAE(config_file= ROOT / p.model_config()).to(device)

    # Initialize optimizer
    vae.optimizer = p.optimizer(vae)

    # Initialize TensorBoard logger
    logger = p.logger()

    # Inintialize data loaders
    train_loader, val_loader, test_loader = p.dataloaders(ROOT)

    callbacks = p.callbacks()

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