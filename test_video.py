from torch.utils import data
import torchvision
from torchvision.io import video
import torchvision.transforms as pt_transforms
from torchvision.datasets.video_utils import VideoClips
import torchaudio
import torch
import glob
import numpy as np
# import cv2

files = glob.glob("data/raw/flute/*.mp4")
print(f'num files {len(files)}')

vid_len = 50

video_clips = VideoClips(files[0:5], clip_length_in_frames=vid_len, frames_between_clips=vid_len//2)

# idx = np.random.randint(0,high=frame_count-vid_len-1)
print(f'NUM CLIPS {video_clips.num_clips()}')

clip = video_clips.get_clip(0)
print(clip[0].shape)
print(clip[1].shape)

dataloader = torch.utils.data.DataLoader(video_clips, batch_size=4, shuffle=True)

dataloader = iter(dataloader)
# print(next(iter(dataloader)))

for i in range(10):
    clip = video_clips.get_clip(i)
    print(clip[0].shape)
    print(clip[1].shape)
    # print(next(dataloader))

# for f in files:
    # cap= cv2.VideoCapture(fs)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f'TOT FRAMES {frame_count}')

    # vid = torchvision.io.read_video(f, start_pts=idx, end_pts=idx+vid_len)
    # print(vid[0].shape)



# print(vid[0].shape)