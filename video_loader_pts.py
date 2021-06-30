import enum
from numpy import double
import torch
import torchvision
import torchaudio
import glob
import random
from torchvision.io import read_video_timestamps
import utils
import numpy as np
# from moviepy.editor import *
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from torchvision.datasets.video_utils import VideoClips
import os

# all_vids = utils.get_all_files("E:/MUSICES/", "mp4")
# all_vids = utils.filter_valid_fps(all_vids, lower_lim=30., upper_lim=30.)
all_vids = utils.load_cache_obj("clipcache/valid_clips.obj")

print(f"found vids {len(all_vids)}")
random.shuffle(all_vids)


clip_len = 90
buff_secs = 0.001 # extra buffer time
frame_hop = 90

video_clips = utils.extract_clips(all_vids[:5], clip_len, frame_hop, None)
expected_samps = int((44100/30) * clip_len )


for i in range(10):

    video, orig_audio, info, video_idx = video_clips.get_clip(i)
    video_path = video_clips.video_paths[video_idx]

    print(f'ORIG AUDIO SH {orig_audio.shape}')
    # now load in the audio separately because torchvision video clips is broken for audio
    video_clip = VideoFileClip(video_path)
    num_frames = int(video_clip.fps * video_clip.duration)
    frame_idx_start = i * frame_hop
    frame_idx_end = frame_idx_start + clip_len


    print(f'start sec {frame_idx_start/video_clip.fps} end sec {frame_idx_end/video_clip.fps}')
    video_clip = video_clip.subclip(frame_idx_start/video_clip.fps, (frame_idx_end/video_clip.fps)+buff_secs)
    audio = video_clip.audio
    audio = audio.to_soundarray()
    audio = torch.as_tensor(audio[:expected_samps, :])
    audio = torch.sum(audio, axis=-1)

    print(f'vid {video.shape} audio {audio.shape} exp shape {expected_samps} fps {info["video_fps"]} sr {info["audio_fps"]}')

    audio = audio.unsqueeze(0)

    torchvision.io.write_video(f"test_vids/example_{i}.mp4",
                            video,
                            fps=info["video_fps"],
                            video_codec="h264",
                            audio_array=audio,
                            audio_fps=info["audio_fps"],
                            audio_codec="aac")




# for i, a in enumerate(all_vids[:10]):
#     video = VideoFileClip(a)
#     num_frames = int(video.fps * video.duration)
#     frame_idx_start = np.random.randint(0, num_frames-clip_len-1)
#     frame_idx_end = frame_idx_start + clip_len

#     print(f'start sec {frame_idx_start/video.fps} end sec {frame_idx_end/video.fps}')
#     video_clip = video.subclip(frame_idx_start/video.fps, (frame_idx_end/video.fps)+buff_secs)
#     audio = video_clip.audio
#     audio = audio.to_soundarray()

#     clip_iter = video_clip.iter_frames()
    
#     frames = [c for c in clip_iter]

#     audio = torch.as_tensor(audio[:expected_samps, :])
#     audio = torch.sum(audio, axis=-1)
#     # audio = torch.sum(audio, axis=-1).type(torch.float32)
#     print(audio.dtype)
#     frames = torch.as_tensor(frames[:clip_len])

#     print(f'video clip shape {frames.shape} audio clip shape {audio.shape} expected samps {expected_samps}')

#     torchvision.io.write_video(f"test_vids/example_{i}.mp4",
#                             frames.type(torch.uint8),
#                             fps=video.fps,
#                             video_codec="h264",
#                             audio_array=audio.unsqueeze(-1),
#                             audio_fps=44100,
#                             audio_codec="aac")
    
    # v = torchvision.io.read_video(a,)
                                #    start_pts=pts[frame_idx_start],
                                #    end_pts=pts[frame_idx_start+clip_len],
                                #    pts_unit="pts")

    # print(f"video {v[0].shape} audio {v[1].shape} sr {v[2]['audio_fps']}")
    # for t in ts[0][:100]:
    #     print(double(t))