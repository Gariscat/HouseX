import csv
import torch
from torch import nn
from torchvision import models
from train import train_loader, val_loader, test_loader
from tqdm import tqdm
import numpy as np
from tensorboard import summary
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

song_types = ['future house', 'bass house', 'progressive house', 'melodic house']
colors = ['red', 'orange', 'blue', 'purple']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ft_model = torch.load('./param/finetuned_ResNet.pth').to(device)


"""
sd = ft_model.state_dict()
sd.pop('2.weight')
sd.pop('2.bias')
backbone = models.resnet18(pretrained=False)
backbone.load_state_dict(sd, False)
backbone.to(device)
backbone.eval()
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

    return table


if __name__ == '__main__':
    train_table = extract_latent(train_loader, 'train')
    val_table = extract_latent(val_loader, 'val')
    test_table = extract_latent(test_loader, 'test')

    pca = PCA(n_components=2)
    for mode, table in zip(('train', 'val', 'test'), (train_table, val_table, test_table)):
        if mode == 'train':
            Xt = pca.fit_transform(table[:, 1:])
        else:
            Xt = pca.transform(table[:, 1:])
        data = {}
        for i in range(len(song_types)):
            data[i] = {'PC1': [], 'PC2': []}
        for j in range(table.shape[0]):
            i = int(table[j, 0])
            data[i]['PC1'].append(Xt[j][0])
            data[i]['PC2'].append(Xt[j][1])
        for i in range(len(song_types)):
            plt.scatter(data[i]['PC1'], data[i]['PC2'], s=2, marker='o', c=colors[i], label=song_types[i])
            # sns.scatterplot(data[i]['PC1'], data[i]['PC2'], s=1.5, marker='h', c=colors[i], label=song_types[i])
        plt.title(f'PCA - {mode} set')
        # plt.xlabel('PC1')
        # plt.ylabel('PC2')
        plt.legend()
        # plt.show()
        plt.savefig(f'2-PCA - {mode} set', bbox_inches='tight')
        plt.close()