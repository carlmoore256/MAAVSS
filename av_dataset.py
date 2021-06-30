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
        self.hop=fft_len//hop_ratio
        self.center_fft = center_fft
        self.use_polar = use_polar
        # self.example_idx = 0
        self.autocontrast = autocontrast
        # filter out clips that are not 30 fps
        if not os.path.isfile("clipcache/valid_clips.obj"):
          all_vids = utils.get_all_files(data_path, "mp4")
          all_vids = utils.filter_valid_fps(all_vids)
          # save to cache because filtering can be time consuming
          utils.save_cache_obj("clipcache/valid_clips.obj", all_vids)
        else:
          all_vids = utils.load_cache_obj("clipcache/valid_clips.obj")

        print(f"number of videos found: {len(all_vids)}")

        self.window = torch.hamming_window(self.fft_len)

        self.transform = pt_transforms.Compose([
          #  pt_transforms.Resize(size)
          # pt_transforms.Grayscale(),
          pt_transforms.RandomResizedCrop(framesize, scale=(0.6,1.0)),
          pt_transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        if shuffle_files:
          random.shuffle(all_vids)

        self.video_clips = utils.extract_clips(all_vids[:4], 
                                              frames_per_clip,
                                              frame_hop,
                                              None)
        # since this is pre-processing, all input audio is assumed to be 44.1k
        self.audio_sample_len = int((44100/framerate) * frames_per_clip)


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
      # audio = torch.sum(audio, dim=0)
      # print(f"SAMPLERATES this sr {sr} self.sr {self.samplerate} audio in {audio.shape}")
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

      # now load in the audio separately because torchvision video clips is broken for audio
      video_clip = VideoFileClip(video_path)
      # num_frames = int(video_clip.fps * video_clip.duration)
      frame_idx_start = clip_idx * self.frame_hop
      frame_idx_end = frame_idx_start + self.frames_per_clip

      video_clip = video_clip.subclip(frame_idx_start/video_clip.fps, (frame_idx_end/video_clip.fps)+0.0001)
      audio = video_clip.audio
      audio = audio.to_soundarray()
      audio = torch.as_tensor(audio[:self.audio_sample_len, :])
      audio = torch.sum(audio, axis=-1)
      # print(f'vid {video.shape} audio {audio.shape} exp shape {self.audio_sample_len} fps {info["video_fps"]} sr {info["audio_fps"]}')

      return video, audio, info
      
    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):

      valid_example = False

      while not valid_example:
        # video, audio, info, video_idx = self.video_clips.get_clip(idx)
        video, audio, info = self.get_av_pair(idx)

        if audio.shape[0] != 0:
          valid_example = True
          # raise Exception('spam', 'eggs')
        else:
          idx += 1

      audio_out = audio.unsqueeze(0)

      torchvision.io.write_video(f"test_vids/example_{idx}.mp4",
                                  video,
                                  fps=info["video_fps"],
                                  video_codec="h264",
                                  audio_array=audio_out,
                                  audio_fps=info["audio_fps"],
                                  audio_codec="aac")
          
      video = video.permute(0, 3, 1, 2)

      sr = info["audio_fps"]
      # print(f'AUDIO FPS {info["audio_fps"]}')

      audio = self.audio_transforms(audio, sr)

      y_stft = self.stft(audio)
      print(f'y_stft shape {y_stft.shape}')
      y_stft = y_stft.permute(2, 0, 1)
      x_stft = self.add_noise(y_stft)

      video = video.type(torch.float)
      # if self.transform is not None:
      video = self.transform(video)

      # attn = torch.unsqueeze(self.attention_extractor._inference(video), 0)
      attn = self.attention_extractor._inference(video)

      attn_save = attn.permute(0,2,3,1) * 255
      attn_save = attn_save.type(torch.uint8).repeat(1, 1, 1, 3)

      print(f"max attn {torch.max(attn)} max save {torch.max(attn_save)}")
      torchvision.io.write_video(f"test_vids/example_{idx}_ATTN.mp4",
                            attn_save,
                            fps=info["video_fps"],
                            video_codec="h264",
                            audio_array=audio_out,
                            audio_fps=info["audio_fps"],
                            audio_codec="aac")

      # attn = attn / 255.

      video = video.permute(1, 0, 2, 3)

      # video = pt_transforms.functional.autocontrast(video)

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

                
            

