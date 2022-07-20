import pygame
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
from nnAudio.features.mel import MelSpectrogram

os.chdir('D:/Project 2022/MIR')
BPM = 126
N_FFT = 256
SR = 22050
WIN_LEN = 60*4*4/BPM
song_types = ['cool', 'dark', 'emotional', 'happy']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def inference(track_name, delta=WIN_LEN, window_length=WIN_LEN):
    mel_converter = MelSpectrogram().cuda()
    sound, sr = librosa.load('{}.ogg'.format(track_name))
    print('Sample rate: {}'.format(sr))
    spec_dir = './melspecgrams/'
    save_dir = track_name[:-4] + '.npy'
    ret = []
    delta_frame_cnt, window_frame_cnt = int(delta * sr), int(window_length * sr)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load('./param/type_classifier.pth').to(device)

    for st in tqdm(range(0, sound.shape[0], delta_frame_cnt)):
        ed = np.minimum(st+window_frame_cnt, sound.shape[0])
        clip_sound = sound[st:ed]
        temp = clip_sound.shape[0]
        if clip_sound.shape[0] < window_frame_cnt:
            clip_sound = np.concatenate((clip_sound, np.zeros(window_frame_cnt-clip_sound.shape[0])))

        plt.axis('off')
        ''''''
        clip_sound = torch.from_numpy(clip_sound).float().cuda()
        mel_spec = mel_converter(clip_sound)[0].cpu().numpy()
        clip_sound = clip_sound.cpu()

        # mel_spec = librosa.feature.melspectrogram(y=clip_sound, sr=sr, n_fft=N_FFT)
        display.specshow(librosa.power_to_db(mel_spec, ref=np.max))
        plt.savefig(spec_dir + 'real_time.jpg', bbox_inches='tight')
        plt.close()

        img = [ele for ele in os.listdir(spec_dir) if '.jpg' in ele and 'real_time' in ele][0]
        img_path = spec_dir + img
        img_tensor = transform(Image.open(img_path).convert('RGB')).to(device)

        output = model(img_tensor.unsqueeze(0)).flatten()
        probs = torch.softmax(output, dim=0)
        # print(probs)
        pred = probs.argmax().item()
        # print(pred)
        ret += [pred for _ in range(temp)]
        # print('[{}, {}) among {}'.format(st, ed, sound.shape[0]), clip_sound.shape[0], len(ret))
        assert len(ret) == ed

    mel_converter = mel_converter.cpu()
    torch.cuda.empty_cache()

    with open(save_dir, 'wb') as f:
        np.save(f, np.array(ret))

    with open('./{}.npy'.format(track_name), 'rb') as f:
        a = np.load(f).tolist()
    a = np.array(a)
    plt.clf()
    plt.plot(a)
    plt.title('unsmoothed')
    plt.show()
    plt.close()
    tmp = np.zeros_like(a)
    st, ed, mu = -1, -1, -1
    cover_len = WIN_LEN*SR*2
    for i in trange(a.shape[0]):
        pre_st = st
        pre_ed = ed
        st = max(0, int(i-cover_len))
        ed = min(a.shape[0], int(i+cover_len))
        if pre_st < st and pre_ed < ed:  # simply speeds up the calc of mean values
            mu += (a[pre_ed] - a[pre_st]) / (ed - st)
        else:
            mu = np.mean(a[st:ed]).item()
        tmp[i] = round(mu)

        if tmp[i] != a[i]:
            print('{}-index is smoothed...'.format(i))
    plt.clf()
    plt.plot(tmp)
    plt.title('smoothed')
    plt.show()
    plt.close()

    with open(save_dir, 'wb') as f:
        np.save(f, np.array(tmp))


def play_and_inference(track_name, delta=7.5, window_length=7.5):
    sound, sr = librosa.load(track_name)
    spec_dir = './melspecgrams/'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load('./param/type_classifier.pth').to(device)

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
            mel_spec = librosa.feature.melspectrogram(y=clip_sound, sr=sr, n_fft=N_FFT)
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


'''
def get_energy(sound, sr, smooth_length=1):
    num_paddings = smooth_length * sr
    absolute_values = np.abs(sound)
    ret = np.zeros(absolute_values.shape[0]+smooth_length-1)
    for i in range(num_paddings):
        ret[i:i+absolute_values.shape[0]] += absolute_values
    ret /= num_paddings
    return ret
'''

if __name__ == '__main__':
    track_name = 'CA7AX Set #4'
    inference(track_name)
    # play_and_inference('./CA7AX Set #3.ogg')

    '''
    s, sr = librosa.load('./cool/Brooks - Lynx.wav')
    e = get_energy(s)
    plt.plot(s[:sr*5], c='blue')
    plt.plot(e[:sr*5] / 5, c='red')
    plt.show()
    plt.close()
    '''