import os
import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from math import floor, ceil
from datetime import timedelta
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import argparse

BPM = 128
# N_FFT = 256
SR = 22050
# WIN_LEN = 60*4*4/BPM
WIN_LEN = 60 * 4 / BPM
song_types = ['future house', 'bass house', 'progressive house', 'melodic house']
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def inference(track_name, delta=WIN_LEN, window_length=WIN_LEN):
    sound, sr = librosa.load(track_name)
    print('Sample rate: {}'.format(sr))
    spec_dir = './melspecgrams/'
    save_dir = track_name[:track_name.index('.')] + '.npy'
    sample_labels = []
    window_labels = []
    delta_frame_cnt, window_frame_cnt = int(delta * sr), int(window_length * sr)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load('./param/finetuned_ResNet.pth').to(device)

    for st in tqdm(range(0, sound.shape[0], delta_frame_cnt), desc=f'inference on track {track_name}'):
        ed = np.minimum(st+window_frame_cnt, sound.shape[0])
        clip_sound = sound[st:ed]
        temp = clip_sound.shape[0]
        if clip_sound.shape[0] < window_frame_cnt:
            clip_sound = np.concatenate((clip_sound, np.zeros(window_frame_cnt-clip_sound.shape[0])))

        plt.axis('off')
        mel_spec = librosa.feature.melspectrogram(y=clip_sound, sr=sr)
        display.specshow(librosa.power_to_db(mel_spec, ref=np.max))
        plt.savefig(spec_dir + 'real_time.jpg', bbox_inches='tight')
        plt.close()

        # img = [ele for ele in os.listdir(spec_dir) if '.jpg' in ele and 'real_time' in ele][0]
        img = 'real_time.jpg'
        img_path = spec_dir + img
        img_tensor = transform(Image.open(img_path).convert('RGB')).to(device)

        output = model(img_tensor.unsqueeze(0)).flatten()
        probs = torch.softmax(output, dim=0)
        # print(probs)
        pred = probs.argmax().item()
        # print(pred)
        sample_labels += [pred for _ in range(temp)]
        window_labels += [pred]
        # print('[{}, {}) among {}'.format(st, ed, sound.shape[0]), clip_sound.shape[0], len(sample_labels))
        assert len(sample_labels) == ed

    with open(save_dir, 'wb') as f:
        np.save(f, np.array(sample_labels))
    """
    with open(save_dir, 'rb') as f:
        a = np.load(f)
    """
    print('inference done')
    plt.figure(figsize=(12, 8))
    plt.clf()
    plt.xlabel('window index')
    plt.yticks([0, 1, 2, 3], song_types)
    plt.plot(window_labels, lw=0.5)
    plt.title('predicted sub-genre (unsmoothed)')

    plt.show()
    plt.close()

    """
    tmp = np.zeros_like(a)
    st, ed, mu = -1, -1, -1
    cover_len = WIN_LEN*SR*2
    for i in trange(a.shape[0]):
        pre_st = st
        pre_ed = ed
        st = max(0, int(i-cover_len//2))
        ed = min(a.shape[0], int(i+cover_len//2))
        tmp[i] = mode(a[st:ed])[0].item()

        if tmp[i] != a[i]:
            print('{}-index is smoothed...'.format(i))
    plt.clf()
    plt.plot(tmp)
    plt.title('smoothed')
    plt.show()
    plt.close()

    with open(save_dir, 'wb') as f:
        np.save(f, tmp)
    """

"""
def play_and_inference(track_name, delta=1.875, window_length=1.875):
    import pygame
    sound, sr = librosa.load(track_name)
    spec_dir = './melspecgrams/'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load('./param/type_classifier_DenseNet.pth').to(device)

    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(track_name)
    pygame.mixer.music.play(loops=1)
    pygame.mixer.music.set_volume(0.01)
    pre, cur = -1, 0
    visited_seconds = [-1]
    while pygame.mixer.music.get_busy():
        pre = cur
        cur = pygame.mixer.music.get_pos() / 1000
        # if (int(floor(cur)) % delta == 0 or int(ceil(cur)) % delta == 0) and cur != pre:
        if int(cur) > visited_seconds[-1]:
            visited_seconds.append(int(cur))
            st = int(cur * sr)
            ed = int(st + window_length * sr)
            clip_sound = sound[st:ed]

            plt.axis('off')
            mel_spec = librosa.feature.melspectrogram(y=clip_sound, sr=sr)
            display.specshow(librosa.power_to_db(mel_spec, ref=np.max))
            plt.savefig(spec_dir + 'real_time.jpg', bbox_inches='tight')

            img = [ele for ele in os.listdir(spec_dir) if '.jpg' in ele and 'real_time' in ele][0]
            img_path = spec_dir + img
            img_tensor = transform(Image.open(img_path).convert('RGB')).to(device)

            output = model(img_tensor.unsqueeze(0)).flatten()
            probs = torch.softmax(output, dim=0).cpu()
            pred = probs.argmax()
            print('Currently timestamp is {}, next {}s-clip is "{}" with confidence {} %'
                  .format(str(timedelta(seconds=cur)), window_length, song_types[pred], probs[pred] * 100))

    pygame.mixer.music.stop()
    pygame.mixer.quit()
"""


if __name__ == '__main__':
    '''
    for i in [1, 2, 3, 4, 5]:
        track_name = f'CA7AX Set #{i}'
        inference(track_name)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--track_name', type=str, default='CA7AX Set #3.ogg')
    args = parser.parse_args()
    inference(args.track_name)
