from os import read
import torch
import torchvision
import torchaudio
import glob
import random
import math
from torchvision.datasets.video_utils import VideoClips
import torch.utils.data
from torchvision.io import read_video_timestamps
from typing import List


class _VideoTimestampsDataset(object):
    """
    Dataset used to parallelize the reading of the timestamps
    of a list of videos, given their paths in the filesystem.
    Used in VideoClips and defined at top level so it can be
    pickled when forking.
    """

    def __init__(self, video_paths: List[str]):
        self.video_paths = video_paths

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        return read_video_timestamps(self.video_paths[idx])

def _collate_fn(x):
    """
    Dummy collate function to be used with _VideoTimestampsDataset
    """
    return x

all_vids = glob.glob("E:/MUSICES/*/**.mp4", recursive=True)
# all_vids = glob.glob("data/raw/*/**.mp4", recursive=True)
random.shuffle(all_vids)

dl = torch.utils.data.DataLoader(
    _VideoTimestampsDataset(all_vids),
    batch_size=16,
    num_workers=0,
    collate_fn=_collate_fn,
)


for batch in dl:
    print(f'next batch')
    clips, fps = list(zip(*batch))
    # we need to specify dtype=torch.long because for empty list,
    # torch.as_tensor will use torch.float as default dtype. This
    # happens when decoding fails and no pts is returned in the list.
    # print(f'clips {clips} fps {fps}')
    # CLIPS RETURNS VIDEO TIMESTAMPS
    clips = [torch.as_tensor(c, dtype=torch.long) for c in clips]
    # self.video_pts.extend(clips)
    # self.video_fps.extend(fps)


# for a in all_vids[:20]:
#     video, audio, info = torchvision.io.read_video(a, 1, 19)
#     print(video.shape, audio.shape)
# video_clips = VideoClips(
#     all_vids[:10],
#     clip_length_in_frames=6,
#     frames_between_clips=2,
#     frame_rate=30
#     # num_workers=num_workers
# )

# i = 0
# while i < 10:
#     video, audio, info, video_idx = video_clips.get_clip(i)
#     sr = info["audio_fps"]
#     fps = info["video_fps"]
#     if audio.shape[1] > 0:
#         print(f'video {video.shape} audio {audio.shape} fps {fps} sr {sr} \n')
#         i += 1
# for a in all_vids:
#     v = torchvision.io.read_video(a)
#     sr = v[2]["audio_fps"]
#     fps = v[2]["video_fps"]

#     print(v[1].shape[1] // sr)
#     print(v[0].shape[0] // fps)
#     print(f'file {a} video {v[0].shape} audio {v[1].shape} fps {fps} sr {sr} \n')

#     video, audio, info = torchvision.io._video_opt._read_video_from_file()

    def compute_clips_for_video(video_pts, num_frames, step, fps, frame_rate):
        if fps is None:
            # if for some reason the video doesn't have fps (because doesn't have a video stream)
            # set the fps to 1. The value doesn't matter, because video_pts is empty anyway
            fps = 1
        if frame_rate is None:
            frame_rate = fps
        total_frames = len(video_pts) * (float(frame_rate) / fps)
        idxs = VideoClips._resample_video_idx(
            int(math.floor(total_frames)), fps, frame_rate
        )
        video_pts = video_pts[idxs]
        clips = unfold(video_pts, num_frames, step)
        if not clips.numel():
            print("There aren't enough frames in the current video to get a clip for the given clip length")
            # warnings.warn("There aren't enough frames in the current video to get a clip for the given clip length and "
            #               "frames between clips. The video (and potentially others) will be skipped.")
        if isinstance(idxs, slice):
            idxs = [idxs] * len(clips)
        else:
            idxs = unfold(idxs, num_frames, step)
        return clips, idxs


def unfold(tensor, size, step, dilation=1):
    """
    similar to tensor.unfold, but with the dilation
    and specialized for 1d tensors
    Returns all consecutive windows of `size` elements, with
    `step` between windows. The distance between each element
    in a window is given by `dilation`.
    """
    assert tensor.dim() == 1
    o_stride = tensor.stride(0)
    numel = tensor.numel()
    new_stride = (step * o_stride, dilation * o_stride)
    new_size = ((numel - (dilation * (size - 1) + 1)) // step + 1, size)
    if new_size[0] < 1:
        new_size = (0, size)
    return torch.as_strided(tensor, new_size, new_stride)


