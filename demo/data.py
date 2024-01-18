import os
import math
import json
import random

import numpy as np
import soundfile as sf
from scipy import signal

import torch
from torch.utils.data import Dataset

from FRAM_RIR import FRAM_RIR

def sample_src_pos(room_dim, num_src, array_pos,
                   min_mic_dis=0.5, max_mic_dis=5, min_dis_wall=None):
    if min_dis_wall is None:
        min_dis_wall = [0.5, 0.5, 0.5]

    # random sample the source positon
    src_pos = []
    while len(src_pos) < num_src:
        pos = np.random.uniform(np.array(min_dis_wall), np.array(
            room_dim) - np.array(min_dis_wall))
        dis = np.linalg.norm(pos - np.array(array_pos))
        
        if dis >= min_mic_dis and dis <= max_mic_dis:
            src_pos.append(pos)

    return np.stack(src_pos, 0)


def sample_mic_array_pos(mic_arch, room_dim, min_dis_wall=None):
    """
    Generate the microphone array position according to the given microphone architecture (geometry)
    :param mic_arch: np.array with shape [n_mic, 3]
                    the relative 3D coordinate to the array_pos in (m)
                    e.g., 2-mic linear array [[-0.1, 0, 0], [0.1, 0, 0]];
                    e.g., 4-mic circular array [[0, 0.035, 0], [0.035, 0, 0], [0, -0.035, 0], [-0.035, 0, 0]]
    :param min_dis_wall: minimum distance from the wall in (m)
    :return
        mic_pos: microphone array position in (m) with shape [n_mic, 3]
        array_pos: array CENTER / REFERENCE position in (m) with shape [1, 3]
    """
    def rotate(angle, valuex, valuey):
        rotate_x = valuex * np.cos(angle) + valuey * np.sin(angle)  # [nmic]
        rotate_y = valuey * np.cos(angle) - valuex * np.sin(angle)
        return np.stack([rotate_x, rotate_y, np.zeros_like(rotate_x)], -1)  # [nmic, 3]
    
    if min_dis_wall is None:
        min_dis_wall = [0.5, 0.5, 0.5]

    mic_arch = np.array(mic_arch)

    mic_array_center = np.mean(mic_arch, 0, keepdims=True)  # [1, 3]
    max_radius = max(np.linalg.norm(mic_arch - mic_array_center, axis=-1))
    array_pos = np.random.uniform(np.array(min_dis_wall) + max_radius,
                                  np.array(room_dim) - np.array(min_dis_wall) - max_radius).reshape(1, 3)
    mic_pos = array_pos + mic_arch
    # assume the array is always horizontal
    # rotate_azm = np.random.uniform(-np.pi, np.pi)
    # mic_pos = array_pos + rotate(rotate_azm, mic_arch[:, 0], mic_arch[:, 1])  # [n_mic, 3]

    return mic_pos, array_pos


def sample_a_config(simu_config):
    room_config = simu_config["min_max_room"]
    rt60_config = simu_config["rt60"]
    mic_dist_config = simu_config["mic_dist"]
    num_src = simu_config["num_src"]
    room_dim = np.random.uniform(np.array(room_config[0]), np.array(room_config[1]))
    rt60 = np.random.uniform(rt60_config[0], rt60_config[1])
    sr = simu_config["sr"]

    if "array_pos" not in simu_config.keys():   # mic_arch must be given in this case
        mic_arch = simu_config["mic_arch"]
        mic_pos, array_pos = sample_mic_array_pos(mic_arch, room_dim)
    else:
        array_pos = simu_config["array_pos"]

    if "src_pos" not in simu_config.keys():
        src_pos = sample_src_pos(room_dim, num_src, array_pos, min_mic_dis=mic_dist_config[0], max_mic_dis=mic_dist_config[1])
    else:
        src_pos = np.array(simu_config["src_pos"])

    return mic_pos, sr, rt60, room_dim, src_pos, array_pos

class ReverbDataset(Dataset):
    def __init__(
            self, 
            source_path, 
            reverb_json,
            reverb_root='/hpc_stor03/sjtu_home/zhisheng.zheng/tencent/reverbs',
            source_length=48000,
            rir_type='reverb',
            mode='train'
        ):
        self.sources = []
        with open(source_path) as f:
            tsv = f.readlines()
            root = tsv[0].strip()
            self.sources = [os.path.join(root, line.split('\t')[0]) for line in tsv[1:]]

        self.reverbs = json.load(open(reverb_json))

        self.reverb_root = reverb_root
        self.rir_type = rir_type
        self.mode = mode

    def __len__(self):
        return len(self.sources)

    def truncate(self, source, max_len=48000):
        if len(source) > max_len:
            source = source[:max_len]
        else:
            source = np.pad(source, (0, max_len - len(source)), 'constant')
        return source

    def __getitem__(self, idx):
        source1 = self.sources[idx]
        source2 = random.choice(self.sources)
        source1, sr = sf.read(source1)
        source2, _ = sf.read(source2)
        source = np.stack([self.truncate(source1), self.truncate(source2)], 0)

        rir_info = random.choice(self.reverbs)
        src_pos = np.array(rir_info['src_pos'])
        array_pos = np.array(rir_info['array_pos'])[0]

        # with shape [n_mic, n_src, rir_len]
        rir = np.load(os.path.join(self.reverb_root, self.mode, rir_info['fname']))
        received_signal = []
        azimuths = []
        for i in range(source.shape[0]):
            received_signal.append(signal.fftconvolve(source[[i]], rir[:, i], mode='full'))
            dx = src_pos[i][0] - array_pos[0]
            dy = src_pos[i][1] - array_pos[1]
            azimuths.append(int(abs(math.degrees(math.atan2(dy, dx))) // 5))
        received_signal = received_signal[0] + received_signal[1]
        return received_signal, sorted(azimuths)

    def collate_fn(self, batch):
        min_samples = min([item[0].shape[1] for item in batch])
        sources = torch.zeros(len(batch), 2, min_samples)
        for i, item in enumerate(batch):
            sources[i, :, :] = torch.from_numpy(item[0])[:, :min_samples]

        targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return sources, targets