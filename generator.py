import torch
import torchaudio
import utils
import cv2
import os
import numpy as np
import random
from video_attention import VideoAttention

class DataGenerator():

    # num vid frames -> how many video frames to extract flow data from, size of 4D tensor
    def __init__(self, 
                 batch_size,
                 num_vid_frames=4, 
                 framerate=30,
                 framesize=256,
                 samplerate=16000, 
                 max_vid_frames=100,
                 noise_std=0.01,
                 center_fft=True, 
                 use_polar=True, 
                 normalize_input_fft=True,
                 shuffle_files=True, 
                 data_path="./data/raw",
                 device=None):

        assert batch_size > 1

        self.attention_extractor = VideoAttention()

        self.max_vid_frames = max_vid_frames
        self.batch_size = batch_size
        self.num_vid_frames = num_vid_frames
        self.framerate = framerate
        self.framesize = framesize
        self.samplerate = samplerate

        self.noise_std = noise_std

        self.normalize_input_fft = normalize_input_fft

        self.fft_len = int((self.num_vid_frames/self.framerate) * self.samplerate)

        self.all_vids = utils.get_all_files(data_path, "mp4")
        print(f"number of videos found: {len(self.all_vids)}")

        if shuffle_files:
          random.shuffle(self.all_vids)

        # self.current_av_pair = self.load_example_pair(self.all_vids[0])

        self.example_idx = 0
        self.center_fft = center_fft
        self.use_polar = use_polar

        if device is None:
          self.device = torch.device('cuda:0') # MAKE THIS SWITCH
        else:
          self.device = device


    # waveform audio -> FFT[:, 0:len(fft)/2]
    def fft(self, audio):
        if self.normalize_input_fft:
          fft = torch.fft.fft(audio, dim=-1, norm="forward")
        else:
          fft = torch.fft.fft(audio, dim=-1, norm=None)

        # remove mirrored half of fft
        fft = fft[:, :fft.shape[-1]//2]
        return fft

    def ifft(self, fft):
        # add back mirrored half of fft as zeros
        fft_mirror = torch.zeros_like(fft)
        fft = torch.cat((fft, fft_mirror), -1)

        if self.normalize_input_fft:
          audio = torch.fft.ifft(fft, dim=-1, norm="forward")
        else:
          audio = torch.fft.ifft(fft, dim=-1, norm=None)
        audio = torch.real(audio)
        return audio

    # x + y(i) -> magnitude, angle (cartesian to polar)
    def cartesian_to_polar(self, cart):
        polar = torch.cat((torch.abs(cart), torch.angle(cart)), -1)
        return polar

    # don't forget here to implement only half the spec eventually!
    # returns cartesion in ri format
    def polar_to_cartesian(self, polar):
        # cartesian = torch.polar(polar[:, 0:1, :], polar[:, 1:2, :])
        cartesian = torch.polar(polar[:, :, 0:1], polar[:, :, 1:2])
        # ri_t = torch.cat(torch.real(cartesian), torch.imag(cartesian), 1)
        ri_t = torch.cat(torch.real(cartesian), torch.imag(cartesian), -1)
        return ri_t

    # cartesian notation to float32 tensor
    # [complex,] -> [real, imaginary]
    def complex_to_ri(self, tensor):
        ri_t = torch.view_as_real(tensor)
        # ri_t = torch.swapaxes(ri_t, 1, 2)
        return ri_t

    # float32 tensor to cartesian notation:
    # [real, imaginary] -> [complex,]
    def ri_to_complex(self, tensor):      
        # complex_t = torch.swapaxes(tensor, 1, 2)
        return torch.view_as_complex(tensor)

    # center fft by interlacing freqs and concatenating mirror
    # this may improve training, with more information density towards the center of the vector,
    # and not to the sides, where convolution artifacts occur, and network density reduces
    # another goal is to achieve greater gaussian distribution by interleaving frequencies
    # in the network during the split/mirror process
    def center_fft_bins(self, fft_tensor):
        left = fft_tensor[:, ::2, :]
        right = fft_tensor[:, 1::2, :]
        left = torch.flip(left, [1])
        centered_fft = torch.cat((left, right), 1)
        return centered_fft

    # reverse process of center_data()
    # un-mirrors and de-interlaces fft_tensors
    def decenter_fft_bins(self, fft_tensor):
        de_interlaced = torch.zeros_like(fft_tensor)
        left = fft_tensor[:, :fft_tensor.shape[1]//2, :]
        right = fft_tensor[:, fft_tensor.shape[1]//2:, :] 
        left = torch.flip(left, [1])
        de_interlaced[:, ::2, :] = left
        de_interlaced[:, 1::2, :] = right
        return de_interlaced

    def reverse_process_fft(self, fft_tensor):
      if self.use_polar:
        fft_tensor = self.polar_to_cartesian(fft_tensor)

      # this step must be done after pol_to_car
      if self.center_fft:
        # fft_tensor = self.complex_to_ri(fft_tensor)
        fft_tensor = self.decenter_fft_bins(fft_tensor)
      
      fft_tensor = self.ri_to_complex(fft_tensor)
      return fft_tensor


    # load audio and video exaple pair
    def load_example_pair(self, vid_path):
      while True:
        try:
          split_path = os.path.split(vid_path)
          name = split_path[-1][:-4]
          name = name + ".wav"
          audio_path = os.path.join(os.path.split(split_path[0])[0], "audio/", name)
          frames = self.load_video(vid_path)
          audio = self.load_audio(audio_path, int(self.samplerate * (self.max_vid_frames/self.framerate)))
          frames = torch.as_tensor(frames)
          audio = torch.as_tensor(audio)
          # print(f'FRAMES SHAPE {frames.shape}')
          # print(f'AUDIO SHAPE {audio.shape}')
          return frames, audio
        except:
          print("error loading audio and video pair, moving to next...")

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        i = 0

        while(cap.isOpened()):
            if i >= self.max_vid_frames:
              break
            ret, frame = cap.read()
            frame = cv2.resize(frame, (self.framesize, self.framesize))
            frames.append(frame)
            if ret is False:
              break
            i += 1
        cap.release()
        frames = np.asarray(frames)
        return frames

    def load_audio(self, path, length=-1):
        waveform, sr = torchaudio.load(path)
        waveform = waveform[0, :length]
        return waveform

    def add_noise(self, audio):
        noise = torch.randn(audio.shape) * self.noise_std
        a_noise = audio + noise
        return a_noise

    # returns a random num_vid_frames frames from video, returns FFT and vid
    def get_random_clip(self, frames, audio):
        frame_start = np.random.randint(0, frames.shape[0]-self.num_vid_frames-1)
        frame_end = frame_start+self.num_vid_frames
        v_clip = frames[frame_start:frame_end]
        sample_start = int((frame_start/self.framerate) * self.samplerate)
        sample_end = sample_start + self.fft_len
        a_clip = audio[sample_start:sample_end]
        return v_clip, a_clip

    def generator(self):
        while True:
            self.example_idx += 1
            if self.example_idx > len(self.all_vids):
                self.example_idx = 0

            frames, audio = self.load_example_pair(self.all_vids[self.example_idx])

            frame_idxs = torch.randint(0, high=len(frames) - self.num_vid_frames - 1, size=(self.batch_size, 1))
            samp_idxs = (frame_idxs.type(torch.double)/self.framerate) * self.samplerate
            samp_idxs = samp_idxs.type(torch.long)

            frame_idxs = torch.cat((frame_idxs, frame_idxs + self.num_vid_frames), -1)
            samp_idxs = torch.cat((samp_idxs, samp_idxs + self.fft_len), -1)

            vid = torch.cat([torch.unsqueeze(frames[index[0]:index[1], :, :], 0) for index in frame_idxs], dim=0)

            vid = torch.cat([torch.unsqueeze(self.attention_extractor._inference(clip), 0) for clip in vid], dim=0)
            vid = torch.unsqueeze(vid, -1)

            vid = vid.type(torch.float) / 255.

            y_audio = torch.cat([torch.unsqueeze(audio[index[0]:index[1]], 0) for index in samp_idxs], dim=0)

            # input data augmentation - add noise (current implementation)
            x_ft = self.fft(self.add_noise(y_audio))
            y_ft = self.fft(y_audio)


            if self.use_polar:
              x_ft = self.cartesian_to_polar(x_ft)
              y_ft = self.cartesian_to_polar(y_ft)
            else:
              # convert tensors to real notation [batch_size, num_bins, real/imag]
              x_ft = self.complex_to_ri(x_ft)
              y_ft = self.complex_to_ri(y_ft)

            if self.center_fft:
              x_ft = self.center_fft_bins(x_ft)
              y_ft = self.center_fft_bins(y_ft)


          
            print(x_ft.shape)
            # have to do this because pytorch is channels first
            x_ft = x_ft.permute(0, 2, 1)
            y_ft = y_ft.permute(0, 2, 1)

            vid = vid.permute(0, 4, 1, 2, 3)

            print(f"SHAPES! x_ft:{x_ft.shape} y_ft:{y_ft.shape} vid:{vid.shape}")


            vid = vid.to(self.device)
            x_ft = x_ft.to(self.device)
            y_ft = y_ft.to(self.device)

            yield [[x_ft, vid], [y_ft, vid]]