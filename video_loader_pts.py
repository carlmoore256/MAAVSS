import torch
import torchvision
import torchaudio
import glob
import random
from torchvision.io import read_video_timestamps


all_vids = glob.glob("E:/MUSICES/*/**.mp4", recursive=True)
# all_vids = glob.glob("data/raw/*/**.mp4", recursive=True)
random.shuffle(all_vids)


for a in all_vids[:2]:
    ts = read_video_timestamps(a, pts_unit='sec')
    print(len(ts))
    print(f'ts 0 {len(ts[0])} ts 1 {ts[1]}')

    for t in ts[0][:100]:
        print(t)

    # video, audio, info = torchvision.io.read_video(a)
    # print(video.shape)