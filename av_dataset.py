import torchvision
import torchvision.transforms as pt_transforms
import torch
import torchaudio
import glob
import numpy as np
from torchvision.transforms.transforms import Grayscale
from video_attention import VideoAttention
from moviepy.editor import VideoFileClip
import random
import utils
import pickle
import os
import time

class AV_Dataset():

    # num vid frames -> how many video frames to extract flow data from, size of 4D tensor
    def __init__(self, 
                 frames_per_clip=4, 
                 frame_hop=2,
                 framerate=30,
                 framesize=256,
                 samplerate=16000, 
                 fft_len=512,
                 hop=2,
                 hops_per_frame=None,
                 noise_std=0.01,
                 center_fft=True, 
                 use_polar=True, 
                 normalize_input_fft=True,
                 autocontrast=False,
                 shuffle_files=True,
                 num_workers=1, 
                 data_path="./data/raw"):

        # set attention extractor parameters
        self.attention_extractor = VideoAttention(
          patch_size=8,
          threshold=0.6
        )
        self.frames_per_clip = frames_per_clip
        self.frame_hop = frame_hop
        self.samplerate = samplerate
        self.fft_len=fft_len
        self.noise_std = noise_std
        self.normalize_input_fft = normalize_input_fft
        # self.fft_len = int((frames_per_clip/framerate) * samplerate)
        self.hop = hop
        self.hops_per_frame = hops_per_frame
        self.center_fft = center_fft
        self.use_polar = use_polar
        self.autocontrast = autocontrast

        # filter out clips that are not 30 fps
        if not os.path.isfile("clipcache/valid_clips.obj"):
          all_vids = utils.get_all_files(data_path, "mp4")
          all_vids = utils.filter_valid_videos(all_vids, lower_lim=29.97002997002996, upper_lim=30.)
          utils.save_cache_obj("clipcache/valid_clips.obj", all_vids)
        else:
          all_vids = utils.load_cache_obj("clipcache/valid_clips.obj")

        print(f"number of videos found: {len(all_vids)}")

        self.window = torch.hamming_window(self.fft_len)

        self.transform = pt_transforms.Compose([
          pt_transforms.RandomResizedCrop(framesize, scale=(0.6,1.0)),
          pt_transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        if shuffle_files:
          random.shuffle(all_vids)

        self.video_clips = utils.extract_clips(all_vids[:7],
                                              frames_per_clip,
                                              frame_hop,
                                              None)

        # self.audio_sample_len = int((samplerate/framerate) * frames_per_clip)
        self.audio_sample_len = int(hops_per_frame * hop * frames_per_clip)

        self.save_output_examples = True

    def stft(self, audio, normalize=True, polar=False):
      # hop = window.shape[0]//hop_ratio
      # consider removing +1 bin to make divisible by 2
      spec = torchaudio.functional.spectrogram(audio, 
                                              pad=0,
                                              window=self.window, 
                                              n_fft=self.fft_len, 
                                              hop_length=self.hop, 
                                              win_length=self.window.shape[0], 
                                              power=None, 
                                              normalized=self.normalize_input_fft, 
                                              onesided=True)
      # fft size should = (..., 2, fft_len/2+1, num_frames * a)
      # remove extra bin as well as extra frame
      spec = spec[:-1, :-1, :]
      print(f'SPEC SHAPE {spec.shape}')
      if self.use_polar:
        spec = torchaudio.functional.magphase(spec)
      return spec

    def istft(self, stft):
      # remember to add back removed bins with padding
      return torchaudio.functional.istft(stft)

    def audio_transforms(self, audio, sr, normalize=True, compress=False):
      if normalize:
        audio *= torch.max(torch.abs(audio))
      if sr != self.samplerate:
        resamp = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.samplerate)
        audio = resamp(audio)
      if compress:
        audio = torchaudio.functional.contrast(audio) # applies compression
      return audio

    def add_noise(self, tensor):
      noise = torch.randn(tensor.shape) * self.noise_std
      a_noise = tensor + noise
      return a_noise

    def get_av_pair(self, idx):
      video, _, info, video_idx, clip_idx = self.video_clips.get_clip(idx)
      video_path = self.video_clips.video_paths[video_idx]
      audio_path = utils.get_paired_audio(video_path, extract=True)
      audio, sr = torchaudio.load(audio_path)
      seconds_start = (clip_idx * self.frame_hop) / info["video_fps"]
      samples_start = round(seconds_start * sr)
      audio = audio[:, samples_start:samples_start+self.audio_sample_len]
      audio = torch.sum(audio, dim=0)
      return video, audio, info["video_fps"], sr

    def save_example(self, attn, audio, video, fps, sr, idx):

      audio_out = audio.unsqueeze(0)
      video_out = video * (1/torch.max(video)) # re-normalize it because norm transform is std
      video_out = torch.clip(video_out, 0., 1.)
      video_out = (video_out.permute(0,2,3,1) * 255).type(torch.uint8)
      attn_out = (attn.permute(0,2,3,1) * 255).type(torch.uint8).repeat(1,1,1,3)

      # print(f'video {video_out.shape} attn {attn.shape} audio {audio_out.shape}')
      torchvision.io.write_video(f"test_vids/example_{idx}.mp4",
                                  video_out,
                                  fps=fps,
                                  video_codec="h264",
                                  audio_array=audio_out,
                                  audio_fps=sr,
                                  audio_codec="aac")

      torchvision.io.write_video(f"test_vids/example_{idx}_ATTN.mp4",
                            attn_out,
                            fps=fps,
                            video_codec="h264",
                            audio_array=audio_out,
                            audio_fps=sr,
                            audio_codec="aac")

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):

      valid_example = False

      while not valid_example:
        # video, audio, info, video_idx = self.video_clips.get_clip(idx)
        video, audio, fps, sr = self.get_av_pair(idx)

        if audio.shape[0] != 0:
          valid_example = True
          # raise Exception('spam', 'eggs')
        else:
          idx += 1

      audio = self.audio_transforms(audio, sr)

      y_stft = self.stft(audio)
      # permute dims [n_fft, timesteps, channels] -> [channels, timesteps, n_fft]
      # timesteps now will line up with 3D tensor when its WxH are flattened
      y_stft = y_stft.permute(2, 1, 0)

      # new dimensionality: time dimension will match video time dim
      # y_stft = y_stft.permute(1,0,2)
      x_stft = self.add_noise(y_stft)

      video = video.permute(0, 3, 1, 2).type(torch.float32)
      video = video / 255.
      video = self.transform(video)
      if self.autocontrast:
        video = pt_transforms.functional.autocontrast(video)

      # get the video's attention map using DINO model
      attn = self.attention_extractor._inference(video)

      if self.save_output_examples:
        self.save_example(attn, audio, video, fps, sr, idx)

      video = video.permute(1, 0, 2, 3)
      attn = attn.permute(1,0,2,3)
      return x_stft, y_stft, attn, audio, video
        

if __name__ == "__main__":
    dataset = AV_Dataset(
        frames_per_clip=10,
        frame_hop=2,
        framerate=30,
        framesize=256,
        fft_len=256,
        hop_ratio=4,
        use_polar=False
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    dataloader = iter(dataloader)

    for i in range(10):
        print(next(dataloader)[0].shape)

                
            

