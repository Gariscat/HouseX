import csv
import torch
from torch import nn
from torchvision import models
from train import train_loader
from tqdm import tqdm
import numpy as np
from tensorboard import summary

song_types = ['future house', 'bass house', 'progressive house', 'melodic house']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ft_model = torch.load('./param/finetuned_ResNet.pth').to(device)
"""
sd = ft_model.state_dict()
sd.pop('2.weight')
sd.pop('2.bias')
backbone = models.resnet18(pretrained=False)
backbone.load_state_dict(sd, False)
"""
backbone = ft_model
backbone.eval()
backbone.to(device)


def extract_latent(loader, mode, model=backbone):
    output_embeddings = []
    for batch in tqdm(loader, desc=f'calculating embeddings for {mode} stage'):
        images, labels = batch[0], batch[1]
        embeds = model(images.to(device)).detach().cpu()
        block = torch.cat((labels.reshape(-1, 1), embeds), dim=-1)
        output_embeddings.append(block)

    table = torch.cat(output_embeddings, dim=0).numpy()
    """
    with open(target_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for i in range(table.shape[0]):
            writer.writerow(table[i])
    """
    with open(f'./logs/{mode}_embeddings.tsv', 'w') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        for i in range(table.shape[0]):
            csv_writer.writerow(table[i, 1:])

    with open(f'./logs/{mode}_metadata.tsv', 'w') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        for i in range(table.shape[0]):
            csv_writer.writerow(table[i, :1])


extract_latent(train_loader, 'train')
