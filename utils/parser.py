import torch.nn as nn
from torchvision import transforms
import lightning as L
import yaml

class Parser:
    def read_config(config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def network(layers):
        modules = []
        for layer in layers:
            if layer[2] == 'Conv2d':
                modules.append(nn.Conv2d(*layer[3]))
            elif layer[2] == 'ConvTranspose2d':
                modules.append(nn.ConvTranspose2d(*layer[3]))
            elif layer[2] == 'Linear':
                modules.append(nn.Linear(*layer[3]))
            elif layer[2] == 'Flatten':
                modules.append(nn.Flatten())
            elif layer[2] == 'Unflatten':
                modules.append(nn.Unflatten(*layer[3]))
            elif layer[2] == 'ReLU':
                modules.append(nn.ReLU(*layer[3]))
            elif layer[2] == 'LeakyReLU':
                modules.append(nn.LeakyReLU(*layer[3]))
            elif layer[2] == 'Softmax':
                modules.append(nn.Softmax(*layer[3]))
            elif layer[2] == 'BatchNorm1d':
                modules.append(nn.BatchNorm1d(*layer[3]))
            elif layer[2] == 'BatchNorm2d':
                modules.append(nn.BatchNorm2d(*layer[3]))
            elif layer[2] == 'MaxPool2d':
                modules.append(nn.MaxPool2d(*layer[3]))
            elif layer[2] == 'Upsample':
                modules.append(nn.Upsample(*layer[3]))
            else:
                raise NotImplementedError(f"Layer type {layer[2]} not implemented.")
        return nn.Sequential(*modules)

    def transforms(transform_list):
        transforms_list = []
        for transform in transform_list:
            transform_name = transform[0]
            transform_args = transform[1]
            transform = getattr(transforms, transform_name)(*transform_args)
            transforms_list.append(transform)
        return transforms.Compose(transforms_list)

    def callbacks(callbacks_list):
        callbacks = []
        for callback in callbacks_list:
            callback_name = callback[0]
            callback_args = callback[1:]
            callback = getattr(L.pytorch.callbacks, callback_name)(callback_args)
            callbacks.append(callback)
        return callbacks
