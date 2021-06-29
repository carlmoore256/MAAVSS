import torchvision
import torchvision.transforms as pt_transforms
from torchvision.datasets.video_utils import VideoClips
import torch
import torchaudio
import glob
import numpy as np
from torchvision.transforms.transforms import Grayscale
from video_attention import VideoAttention
import random
import utils
import pickle
import os


class AV_Dataset():

    # num vid frames -> how many video frames to extract flow data from, size of 4D tensor
    def __init__(self, 
                #  batch_size,
                 frames_per_clip=4, 
                 frame_hop=2,
                 framerate=30,
                 framesize=256,
                 samplerate=16000, 
                 fft_len=512,
                 hop_ratio=2,
                 noise_std=0.01,
                 center_fft=True, 
                 use_polar=True, 
                 normalize_input_fft=True,
                 autocontrast=False,
                 shuffle_files=True,
                 num_workers=1, 
                 data_path="./data/raw"):

        # assert batch_size > 1

        self.attention_extractor = VideoAttention()
        self.frames_per_clip = frames_per_clip
        self.samplerate = samplerate
        self.fft_len=fft_len
        self.noise_std = noise_std
        self.normalize_input_fft = normalize_input_fft
        # self.fft_len = int((frames_per_clip/framerate) * samplerate)
        self.hop=fft_len//hop_ratio
        self.center_fft = center_fft
        self.use_polar = use_polar
        # self.example_idx = 0
        self.autocontrast = autocontrast
        all_vids = utils.get_all_files(data_path, "mp4")
        print(f"number of videos found: {len(all_vids)}")

        self.window = torch.hamming_window(self.fft_len)

        self.transform = pt_transforms.Compose([
          #  pt_transforms.Resize(size)
          # pt_transforms.Grayscale(),
          pt_transforms.RandomResizedCrop(framesize, scale=(0.6,1.0)),
          # pt_transforms.Normalize(
          #           (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # if shuffle_files:
        #   random.shuffle(all_vids)

        self.extract_clips(all_vids, frames_per_clip, frame_hop, framerate)

        self.audio_sample_len = (samplerate//framerate) + 32


    def extract_clips(self, all_vids, frames_per_clip, frame_hop, framerate):
      if not os.path.isfile("clipcache/video_clips.obj"):
        self.video_clips = VideoClips(
            all_vids[:10],
            clip_length_in_frames=frames_per_clip,
            frames_between_clips=frame_hop,
            frame_rate=framerate,
            # num_workers=num_workers
        )

        filehandler = open("clipcache/video_clips.obj", 'wb')
        pickle.dump(self.video_clips, filehandler)
      else:
        print("loading video clip slices from cache")
        filehandler = open("clipcache/video_clips.obj", 'rb') 
        self.video_clips = pickle.load(filehandler)

    def stft(self, audio, normalize=True, polar=False):
      # hop = window.shape[0]//hop_ratio
      spec = torchaudio.functional.spectrogram(audio, 
                                              pad=0,
                                              window=self.window, 
                                              n_fft=self.fft_len, 
                                              hop_length=self.hop, 
                                              win_length=self.window.shape[0], 
                                              power=None, 
                                              normalized=self.normalize_input_fft, 
                                              onesided=True)
      if self.use_polar:
        spec = torchaudio.functional.magphase(spec)
      return spec

    def audio_transforms(self, audio, sr, normalize=True, compress=False):
      # if normalize:
      #   audio = torchaudio.functional.gain(audio, 
      audio = torch.sum(audio, dim=0)
      print(f'AUDIO SHAPE {audio.shape} AVG LEN {self.audio_sample_len} ratio {audio.shape[0]/self.audio_sample_len}')

      # print(f"SAMPLERATES this sr {sr} self.sr {self.samplerate} audio in {audio.shape}")
      if sr != self.samplerate:
        resamp = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.samplerate)
        audio = resamp(audio)
      # print(f'AUDIO OUT {audio.shape}')
      if compress:
        audio = torchaudio.functional.contrast(audio) # applies compression


      # if audio.shape[0] != self.audio_sample_len:
      #   audio


      return audio

    def add_noise(self, tensor):
      noise = torch.randn(tensor.shape) * self.noise_std
      a_noise = tensor + noise
      return a_noise

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):

      try:
        valid_example = False
        video, audio, info, video_idx = self.video_clips.get_clip(idx)

        if audio.shape[1] == 0:
          # valid_example = True
          print("INVALID EXAMPLE, SKIPPING")
          raise Exception('spam', 'eggs')
          # pass
            
        video = video.permute(0, 3, 1, 2)

        sr = info["audio_fps"]
        # print(f'AUDIO FPS {info["audio_fps"]}')

        audio = self.audio_transforms(audio, sr)

        y_stft = self.stft(audio)
        y_stft = y_stft.permute(2, 0, 1)
        x_stft = self.add_noise(y_stft)

        video = video.type(torch.float)
        # if self.transform is not None:
        video = self.transform(video)

        attn = torch.unsqueeze(self.attention_extractor._inference(video), 0)

        attn = attn / 255.

        video = video.permute(1, 0, 2, 3)

        # video = pt_transforms.functional.autocontrast(video)

        return x_stft, y_stft, attn, audio, video
      except Exception as e:
        print(e)
        

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

                
            

