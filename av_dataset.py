import torchvision
import torchvision.transforms as pt_transforms
import torch
import torchaudio
import numpy as np
from torchvision.transforms.transforms import Grayscale
from video_attention import VideoAttention
import random
import utilities
import os
import torch.nn.functional as F
import wandb
from PIL import Image

class AV_Dataset():

    # num vid frames -> how many video frames to extract flow data from, size of 4D tensor
    def __init__(self, 
                 num_frames=4, 
                 frame_hop=2,
                 framerate=30,
                 framesize=256,
                 samplerate=16000, 
                 fft_len=512,
                 hops_per_frame=8,
                 noise_std=0.01,
                 use_polar=False, 
                 attn_diff=False,
                 normalize_input_fft=True,
                 normalize_output_fft=True,
                 autocontrast=False,
                 compress_audio=False,
                 shuffle_files=True,
                 data_path="./data/raw",
                 max_clip_len=None,
                 gen_stft=True,
                 gen_video=True,
                 wandb_run=None,
                 trim_stft_end=True,
                 return_video_path=False,
                 attn_frames_path=None):
        
        self.gen_stft = gen_stft
        self.gen_video = gen_video

        self.hop, self.audio_sample_len, _ = utilities.calc_hop_size(num_frames, hops_per_frame, framerate, samplerate)

        # if gen_video:
          # set attention extractor parameters
        self.attention_extractor = VideoAttention(
          patch_size=8,
          threshold=0.6)

        self.num_frames = num_frames
        self.frame_hop = frame_hop
        self.samplerate = samplerate
        self.framerate = framerate
        self.fft_len=fft_len
        self.noise_std = noise_std
        self.normalize_input_fft = normalize_input_fft
        self.normalize_output_fft = normalize_output_fft
        # self.fft_len = int((num_frames/framerate) * samplerate)
        self.hops_per_frame = hops_per_frame
        # self.center_fft = center_fft
        self.use_polar = use_polar
        self.attn_diff = attn_diff # takes derivative of attention frames
        self.autocontrast = autocontrast
        self.compress_audio = compress_audio
        self.trim_stft_end = trim_stft_end
        self.return_video_path = return_video_path
        self.attn_frames_path = attn_frames_path

        self.cache_ratio = [0, 0]
        # if attn_frames_path is not None:
        #   self.all_attn_frames = {}
        #   all_frames = utilities.get_all_files(attn_frames_path, "jpg")
        #   for f in all_frames:
        #     subdir_name = os.path.split(os.path.split(f)[0])[-1]
        #     print(subdir_name)
        #     self.all_attn_frames[subdir_name] = {
        #       f,

        #     }
        self.imgLoadTransforms = pt_transforms.Compose([
          pt_transforms.ToTensor(),
          pt_transforms.Grayscale()
        ])


        self.backend = torchaudio.get_audio_backend()

        # filter out clips that are not 30 fps
        if not os.path.isfile("clipcache/valid_clips.obj"):
          all_vids = utilities.get_all_files(data_path, "mp4")
          all_vids = utilities.filter_valid_videos(all_vids, 
                                                fps_lower_lim=29.97002997002996, 
                                                fps_upper_lim=30., 
                                                max_frames=max_clip_len)

          utilities.save_cache_obj("clipcache/valid_clips.obj", all_vids)
        else:
          all_vids = utilities.load_cache_obj("clipcache/valid_clips.obj")
        self.all_vids = all_vids
        print(f"number of videos found: {len(self.all_vids)}")

        self.window = torch.hamming_window(self.fft_len)

        self.transform = pt_transforms.Compose([
          pt_transforms.RandomResizedCrop(framesize, scale=(0.6,1.0)),
          pt_transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        if shuffle_files:
          random.shuffle(all_vids)

        # if gen_video:
        self.video_clips = utilities.extract_clips(all_vids[:],
                                              num_frames,
                                              frame_hop,
                                              None)

        if wandb_run is not None:
          artifact = wandb.Artifact(f"video-clips{num_frames}f-{frame_hop}h", type='dataset', metadata={
            "num_frames" : num_frames,
            "frame_hop" : frame_hop,
            "num_clips" : len(self.video_clips.video_clips.video_paths)
          })
          with artifact.new_file('video_clips.obj') as f:
            f.write(self.video_clips)
          wandb_run.log_artifact(artifact)

        # self.audio_sample_len = int((samplerate/framerate) * num_frames)
        self.save_output_examples = False

        self.audio_memmap, self.audio_index_map = utilities.load_audio_map(data_path)

        if self.audio_memmap is None:
          self.use_audio_memmap = False
        else:
          self.use_audio_memmap = True
          vc_paths = [os.path.normpath(p) for p in self.video_clips.video_paths]
          self.mmp_paths = [os.path.normpath(p) for p in self.audio_index_map[0]]
          # at the index in this array is a map to mmp_paths; [video_clip_loc] -> [memmap_loc]
          self.vc_path_idxs = []
          for i, p in enumerate(vc_paths):
            self.vc_path_idxs.append(self.mmp_paths.index(p))

    def get_cache_ratio(self):
      return self.cache_ratio[0]/(self.cache_ratio[0]+self.cache_ratio[1])

    # change the output of the dataset between audio, video, and av
    def toggle_dataset_mode(self, a, v):
      self.gen_stft = a
      self.gen_video = v

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
      if self.trim_stft_end:
        spec = spec[:-1, :-1, :]
      else:
        spec = spec[:, :-1, :]

      if self.use_polar:
        spec = torchaudio.functional.magphase(spec)
        spec = torch.cat((spec[0].unsqueeze(0), spec[1].unsqueeze(0)), dim=0)
      return spec

    def istft(self, stft):
      # remember to add back removed bins with padding
      if self.trim_stft_end:
        stft = F.pad(stft, (0, 1)).permute(2,1,0)
      else:
        stft = stft.permute(2,1,0)
      if self.use_polar:
        mag = stft[:, :, 0]
        phase = stft[:, :, 1]
        rectangular = mag(torch.cos(phase) + (1j*torch.sin(phase)))
        stft = torch.view_as_real(rectangular)
        # stft = torchaudio.functional.magphase(stft)

      audio = torch.istft(stft.cpu().detach(), 
                          n_fft=self.fft_len, 
                          hop_length=self.hop, 
                          win_length=self.fft_len,
                          window=self.window,
                          normalized=self.normalize_input_fft,
                          onesided=True)
      return audio

    def audio_transforms(self, audio, sr, normalize=False):
      if audio.ndim > 1:
        if audio.shape[0] > 1:
          audio /= audio.shape[0]
        audio = torch.sum(audio, dim=0)
      if normalize:
        audio *= torch.max(torch.abs(audio))
      if sr != self.samplerate:
        resamp = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.samplerate)
        audio = resamp(audio)
      if self.compress_audio:
        audio = torchaudio.functional.contrast(audio) # applies compression
      return audio

    def add_noise(self, tensor):
      noise = torch.randn(tensor.shape) * self.noise_std
      a_noise = tensor + noise
      return a_noise

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

    def get_av_pair(self, idx):
      video, _, info, video_idx, clip_idx = self.video_clips.get_clip(idx)
      audio = self.get_audio(idx, info["video_fps"])
      return video, audio, info["video_fps"], self.samplerate

    # method for faster loading - get video frames from already existing attn frames
    def get_frames_cached(self, idx):
      video_idx, clip_idx = self.video_clips.get_clip_location(idx)
      video_path = self.video_clips.video_paths[video_idx]
      video_name = os.path.split(video_path)[-1][:-4]
      folder_path = os.path.join(self.attn_frames_path, video_name)
      true_idx = self.frame_hop * clip_idx
      img_paths = [os.path.join(folder_path, f'img_{i+true_idx:05d}.jpg') for i in range(self.num_frames)]
      attn = torch.zeros(1, self.num_frames, 256, 256)
      if utilities.verify_files(img_paths):
        self.cache_ratio[0] += 1
        attn = torch.zeros(1, self.num_frames, 256, 256)
        video = torch.zeros(3, self.num_frames, 256, 256)

        for i, path in enumerate(img_paths):
          img = Image.open(path)
          attn[:, i, :, :] = self.imgLoadTransforms(img)
        
        if self.attn_diff:
          attn = torch.diff(attn, )
      else: # generate the attention map
        self.cache_ratio[1] += 1
        attn, video = self.gen_video_example(idx)
        # cache files for later useage
        for i in range(attn.shape[1]):
          path = img_paths[i]
          frame = attn[:, i, :, :].repeat(3, 1, 1)
          torchvision.utils.save_image(frame, path)
      return attn, video

    def gen_av_example_cached(self, idx):
      attn, video = self.get_frames_cached(idx)
      x_stft, y_stft, audio = self.gen_stft_example(idx)
      return x_stft, y_stft, attn, audio, video

    def get_audio(self, idx, fps):
      video_idx, clip_idx = self.video_clips.get_clip_location(idx)
      seconds_start = (clip_idx * self.frame_hop) / fps
      samples_start = round(seconds_start * self.samplerate)
      if self.use_audio_memmap:
        idx_map_idx = self.vc_path_idxs[video_idx]
        mmap_idx = self.audio_index_map[1][idx_map_idx]
        audio = self.audio_memmap[mmap_idx[0]+samples_start:mmap_idx[0]+samples_start+self.audio_sample_len]
        audio = torch.as_tensor(audio)
        audio = self.audio_transforms(audio, self.samplerate, normalize=False)
      else:
        video_path = self.video_clips.video_paths[video_idx]
        audio_path = utilities.get_paired_audio(video_path, extract=True)
        audio, sr = torchaudio.load(audio_path, samples_start, num_frames=self.audio_sample_len)
        audio = self.audio_transforms(audio, self.samplerate, normalize=False)
      return audio

    def gen_av_example(self, idx):
      video, _, info, video_idx, clip_idx = self.video_clips.get_clip(idx)
      # audio = self.get_audio(idx, info["video_fps"])
      # y_stft = self.stft(audio)
      # if self.normalize_output_fft:
      #   y_stft *= 1/torch.max(torch.abs(y_stft) + 1e-7)
      # # permute dims [n_fft, timesteps, channels] -> [channels, timesteps, n_fft]
      # # timesteps now will line up with 3D tensor when its WxH are flattened
      # y_stft = y_stft.permute(2, 1, 0)
      # # new dimensionality: time dimension will match video time dim
      # # y_stft = y_stft.permute(1,0,2)
      # x_stft = self.add_noise(y_stft)
      x_stft, y_stft, audio = self.gen_stft_example(idx)
      video = video.permute(0, 3, 1, 2).type(torch.float32)
      video = video / 255.
      video = self.transform(video)
      if self.autocontrast:
        video = pt_transforms.functional.autocontrast(video)
      # get the video's attention map using DINO model
      attn = self.attention_extractor._inference(video)

      if self.attn_diff:
        pad = torch.zeros_like(attn[0:1])
        attn = torch.diff(attn, dim=0)
        attn = torch.cat((pad, attn), 0)

      attn *= 1/torch.max(attn)
      if self.save_output_examples:
        self.save_example(attn, audio, video, info["video_fps"], self.samplerate, idx)
      video = video.permute(1, 0, 2, 3)
      attn = attn.permute(1,0,2,3)
      return x_stft, y_stft, attn, audio, video

    def gen_stft_example(self, idx):
      audio = self.get_audio(idx, self.framerate)
      y_stft = self.stft(audio)
      y_stft = y_stft.permute(2, 1, 0)
      if self.normalize_output_fft:
        y_stft *= 1/torch.max(torch.abs(y_stft) + 1e-7)
      x_stft = self.add_noise(y_stft)
      return x_stft, y_stft, audio

    def gen_video_example(self, idx):
      video, _, _, vid_idx, clip_idx = self.video_clips.get_clip(idx)
      video = video.permute(0, 3, 1, 2).type(torch.float32)
      video = video / 255.
      video = self.transform(video)
      if self.autocontrast:
        video = pt_transforms.functional.autocontrast(video)
      # get the video's attention map using DINO model
      attn = self.attention_extractor._inference(video)
      attn *= 1/torch.max(attn)
      video = video.permute(1, 0, 2, 3)
      attn = attn.permute(1,0,2,3)
      # path return useful for getting file to do processing with
      if self.return_video_path:
        return attn, video, [self.video_clips.video_paths[vid_idx], clip_idx]
      else:
        return attn, video

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
      
      if self.gen_stft and self.gen_video:
        if self.attn_frames_path is not None:
          return self.gen_av_example_cached(idx)
        else:
          return self.gen_av_example(idx)
      
      if self.gen_stft and not self.gen_video:
        return self.gen_stft_example(idx)

      if self.gen_video and not self.gen_stft:
        return self.gen_video_example(idx)

########################################################################
# use to train stft autoencoder
class STFT_Dataset():
  def __init__(self,
              samplerate=16000,
              fft_len=512,
              hop=2,
              audio_sample_len=0,
              noise_std=0.1,
              normalize_input_fft=True,
              normalize_output_fft=False,
              use_polar=False,
              data_path='',
              max_clip_len=None,
              split=0.8):

    self.samplerate = samplerate
    self.fft_len = fft_len
    self.hop = hop
    self.audio_sample_len=audio_sample_len
    self.noise_std = noise_std
    self.normalize_input_fft = normalize_input_fft
    self.normalize_output_fft = normalize_output_fft
    self.use_polar = use_polar

    self.window = torch.hamming_window(self.fft_len)


    # filter out clips that are not 30 fps
    if not os.path.isfile("clipcache/valid_clips.obj"):
      all_vids = utilities.get_all_files(data_path, "mp4")
      all_vids = utilities.filter_valid_videos(all_vids, 
                                            fps_lower_lim=29.97002997002996, 
                                            fps_upper_lim=30., 
                                            max_frames=max_clip_len)

      utilities.save_cache_obj("clipcache/valid_clips.obj", all_vids)
    else:
      all_vids = utilities.load_cache_obj("clipcache/valid_clips.obj")

    # random.shuffle(all_vids) # shuffle between validation split
    # split_loc = int(len(all_vids)*split)
    # self.train_vids = all_vids[:split_loc]
    # self.val_vids = all_vids[split_loc:]
    self.all_vids = all_vids


  def add_noise(self, tensor):
    noise = torch.randn(tensor.shape) * self.noise_std
    a_noise = tensor + noise
    return a_noise

  def istft(self, stft):
    # remember to add back removed bins with padding
    stft = F.pad(stft, (0, 1)).permute(2,1,0)
    if self.use_polar:
      mag = stft[:, :, 0]
      phase = stft[:, :, 1]
      rectangular = mag(torch.cos(phase) + (1j*torch.sin(phase)))
      stft = torch.view_as_real(rectangular)
      # stft = torchaudio.functional.magphase(stft)

    audio = torch.istft(stft.cpu().detach(), 
                        n_fft=self.fft_len, 
                        hop_length=self.hop, 
                        win_length=self.fft_len,
                        window=self.window,
                        normalized=self.normalize_input_fft,
                        onesided=True)
    return audio

  def stft(self, audio, normalize=True, polar=False):
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
    spec = spec[:-1, :-1, :]

    if self.use_polar:
      spec = torchaudio.functional.magphase(spec)
      spec = torch.cat((spec[0].unsqueeze(0), spec[1].unsqueeze(0)), dim=0)
    return spec

  def get_example(self, idx):
    # if train:
    audio_path = utilities.get_paired_audio(self.all_vids[idx], extract=True)
    info = torchaudio.info(audio_path)
    sr = info.sample_rate
    samples_start = np.random.randint(0,high=info.num_frames-self.audio_sample_len-1)
    audio, sr = torchaudio.load(audio_path, samples_start, num_frames=self.audio_sample_len)
    audio = self.audio_transforms(audio, sr)
    y_stft = self.stft(audio)
    y_stft = y_stft.permute(2, 1, 0)
    if self.normalize_output_fft:
      y_stft *= 1/torch.max(torch.abs(y_stft))
    x_stft = self.add_noise(y_stft)
    return x_stft, y_stft, audio

  # def get_val_example(self, idx):
  #   return self.get_example(idx, train=False)
    
  def __len__(self):
    return len(self.all_vids)

  def __getitem__(self, idx):
    return self.get_example(idx)


class Video_Dataset():

    # num vid frames -> how many video frames to extract flow data from, size of 4D tensor
    def __init__(self, 
                 num_frames=4, 
                 frame_hop=2,
                 framesize=256,
                 autocontrast=False,
                 shuffle_files=True,
                 data_path="./data/raw",
                 max_clip_len=None):

        # set attention extractor parameters
        self.attention_extractor = VideoAttention(
          patch_size=8,
          threshold=0.6
        )
        self.num_frames = num_frames
        self.frame_hop = frame_hop
        self.autocontrast = autocontrast

        # filter out clips that are not 30 fps
        if not os.path.isfile("clipcache/valid_clips.obj"):
          all_vids = utilities.get_all_files(data_path, "mp4")
          all_vids = utilities.filter_valid_videos(all_vids, 
                                                fps_lower_lim=29.97002997002996, 
                                                fps_upper_lim=30., 
                                                max_frames=max_clip_len)

          utilities.save_cache_obj("clipcache/valid_clips.obj", all_vids)
        else:
          all_vids = utilities.load_cache_obj("clipcache/valid_clips.obj")

        print(f"number of videos found: {len(all_vids)}")

        self.transform = pt_transforms.Compose([
          pt_transforms.RandomResizedCrop(framesize, scale=(0.6,1.0)),
          pt_transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        if shuffle_files:
          random.shuffle(all_vids)

        self.video_clips = utilities.extract_clips(all_vids,
                                              num_frames,
                                              frame_hop,
                                              None)

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
      video, _, _, _, _ = self.video_clips.get_clip(idx)
      
      video = video.permute(0, 3, 1, 2).type(torch.float32)
      video = video / 255.
      video = self.transform(video)
      
      if self.autocontrast:
        video = pt_transforms.functional.autocontrast(video)

      # get the video's attention map using DINO model
      attn = self.attention_extractor._inference(video)

      attn *= 1/torch.max(attn)

      video = video.permute(1, 0, 2, 3)
      attn = attn.permute(1,0,2,3)

      return attn, video

if __name__ == "__main__":
    dataset = AV_Dataset(
        num_frames=10,
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

