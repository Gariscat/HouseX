import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import json


def plot_item(key='train_loss'):
    sub_dirs = glob("./logs/*/", recursive=False)
    plt.figure(figsize=(6, 4))
    plt.xticks(range(21))
    for sub_dir in sub_dirs:
        _ = sub_dir[sub_dir.index('_')+1:]
        network_name = _[:_.index('_')]
        with open(sub_dir+'training_log.json', 'r') as f:
            training_log = json.load(f)
        plt.plot(training_log[key], label=network_name)
    plt.grid(linestyle='--')
    plt.title(key.replace('acc', 'accuracy').replace('_', ' '))
    plt.legend()
    plt.savefig('./logs/'+key+'.jpg', bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f'{key} plotted...')


if __name__ == '__main__':
    plot_item('train_loss')
    plot_item('val_loss')
    plot_item('test_loss')
    plot_item('train_acc')
    plot_item('val_acc')
    plot_item('test_acc')
        