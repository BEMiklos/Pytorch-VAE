import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import lightning as L
import torch
import yaml

class Parser:
    def __init__(self,config_file,args=None):
        self.args= args
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def config(self):
        return self.config

    def optimizer(self, model):
        optimizer_type = self.config['trainer']['optimizer']['type']
        optimizer_args = self.config['trainer']['optimizer']['args']
        return getattr(torch.optim, optimizer_type)(model.parameters(), **optimizer_args)

    def logger(self):
        logger_type = self.config['trainer']['logger']['type']
        logger_args = self.config['trainer']['logger']['args']
        return getattr(L.pytorch.loggers, logger_type)(**logger_args)

    def model_config(self):
        return self.config['model']['config'] if self.args.model is None else self.args.model

    def network(self,layers):
        modules = []
        for layer in layers:
            module = getattr(nn, layer[2])(*layer[3])
            for times in range(layer[1]):
                modules.append(module)
        return nn.Sequential(*modules)

    def model(self, config_file):
        if self.args.model is not None: config_file = self.args.model

        vae = VAE(config_file= config_file)
        if self.args.detached is not None: vae.detached = self.args.detached
        if self.args.kld is not None: vae.kld_weight = self.args.kld
        if self.args.mse is not None: vae.mse_weight = self.args.mse
        if self.args.class_weight is not None: vae.class_weight = self.args.class_weight
        return vae

    def train_transforms(self):
        return self.transforms(self.config['data']['train_transform'])

    def test_transforms(self):
        return self.transforms(self.config['data']['test_transform'])

    def transforms(self,transform_list):
        transforms_list = []
        for transform in transform_list:
            transform_name = transform[0]
            transform_args = transform[1]
            transform = getattr(transforms, transform_name)(*transform_args)
            transforms_list.append(transform)
        return transforms.Compose(transforms_list)

    def dataloaders(self, dir):
        # Define dataset and dataloader
        train_transform = self.train_transforms()
        # Define dataset and dataloader
        test_transform = self.test_transforms()

        dataset_name = self.config['data']['dataset']
        val_split = self.config['data']['val_split']
        train_batch_size = self.config['data']['train_batch_size']
        val_batch_size = self.config['data']['val_batch_size']
        test_batch_size = self.config['data']['test_batch_size']
        num_workers = self.config['data']['num_workers']

        train_dataset = datasets.__dict__[dataset_name](root=dir / 'data', train=True, download=True,
                                                        transform=train_transform)
        test_dataset = datasets.__dict__[dataset_name](root=dir / 'data', train=False, download=True,
                                                       transform=test_transform)
        val_size = int(val_split * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        return (DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers),
                DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers),
                DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers))


    def callbacks(self):
        callbacks_list = self.config['trainer']['callbacks']
        callbacks = []
        for callback in callbacks_list:
            callback_name = callback[0]
            callback_args = callback[1]
            callback = getattr(L.pytorch.callbacks, callback_name)(*callback_args)
            callbacks.append(callback)
        return callbacks
