import numpy as np
import torch
import json

class log(object):
    def __init__(self) -> None:
        self.data = {
            'train_loss': [], 'val_loss': [], 'test_loss': [],
            'train_acc': [], 'val_acc': [], 'test_acc': []
        }
        
    def push(self, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc):
        self.data['train_loss'] += [train_loss]
        self.data['val_loss'] += [val_loss]
        self.data['test_loss'] += [test_loss]
        self.data['train_acc'] += [train_acc]
        self.data['val_acc'] += [val_acc]
        self.data['test_acc'] += [test_acc]
        
    def save(self, tar_path):
        with open(tar_path, 'w') as f:
            json.dump(self.data, f)