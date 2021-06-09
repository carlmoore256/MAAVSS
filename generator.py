import tensorflow as tf
import utils
import cv2
import glob
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
                 shuffle_files=True, 
                 data_path="./data/raw"):

        assert batch_size > 1

        self.attention_extractor = VideoAttention()
        
        self.max_vid_frames = max_vid_frames
        self.batch_size = batch_size
        self.num_vid_frames = num_vid_frames
        self.framerate = framerate
        self.framesize = framesize
        self.samplerate = samplerate

        self.noise_std = noise_std

        self.normalize_input_fft = False

        self.fft_len = int((self.num_vid_frames/self.framerate) * self.samplerate)

        self.all_vids = utils.get_all_files(data_path, "mp4")
        print(f'number of videos found: {len(self.all_vids)}')

        if shuffle_files:
          random.shuffle(self.all_vids)

        # self.current_av_pair = self.load_example_pair(self.all_vids[0])

        self.example_idx = 0
        self.center_fft = center_fft
        self.use_polar = use_polar



    # waveform audio -> FFT (tf.complex64 dtype)
    def fft(self, audio):
        audio = tf.cast(audio, dtype=tf.complex64)
        fft = tf.signal.fft(tf.transpose(audio))
        if self.normalize_input_fft:
            fft = self.normalize_fft(fft)
        # fft = tf.transpose(fft)
        fft = fft[:, :fft.shape[-1]//2]
        return fft

    def ifft(self, fft):
        audio = tf.signal.ifft(fft)
        audio = tf.math.real(audio)
        return audio

    # normalize fft by 1/(length/2) 
    def normalize_fft(self, fft):
        scalar = 1.0/(fft.shape[-1] // 2)
        normalized_fft = tf.math.multiply(fft, scalar)
        return normalized_fft

    def reverse_normalize_fft(self, normalized_fft):
        return normalized_fft * (normalized_fft.shape[-1] * 2)

    # x + y(i) -> magnitude, angle (cartesian to polar)
    def cartesian_to_polar(self, cartesian, concat_axis=0):
        magnitude = tf.abs(cartesian)
        angle = tf.math.angle(cartesian)
        polar = tf.concat([magnitude, angle], axis=concat_axis)
        return polar

    # don't forget here to implement only half the spec eventually!
    # returns cartesion in ri format
    def polar_to_cartesian(self, polar):
        real = polar[:, 0:1, :] * np.sin(polar[:, 1:, :])
        imag = polar[:, 0:1, :] * np.cos(polar[:, 1:, :])
        ri_t = tf.concat([real, imag], axis=1)
        # complex_t = tf.dtypes.complex(real, imag)
        return ri_t

    # cartesian notation to float32 tensor
    # [complex,] -> [real, imaginary]
    def complex_to_ri(self, tensor, concat_axis=0):      
        real = tf.math.real(tensor)
        imag = tf.math.imag(tensor)
        ri_t = tf.concat([real, imag], axis=0)
        return ri_t

    # float32 tensor to cartesian notation:
    # [real, imaginary] -> [complex,]
    def ri_to_complex(self, tensor):      
        real = tensor[:,0,:]
        imag = tensor[:,1,:]
        # account for FFT mirror cutoff at N/2+1
        mirror = tf.zeros_like(real)
        real = tf.concat([real,mirror], axis=-1)
        imag = tf.concat([imag,mirror], axis=-1)
        complex_t = tf.dtypes.complex(real, imag)
        return complex_t

    # center fft by interlacing freqs and concatenating mirror
    # this may improve training, with more information density towards the center of the vector,
    # and not to the sides, where convolution artifacts occur, and network density reduces
    # another goal is to achieve greater gaussian distribution by interleaving frequencies
    # in the network during the split/mirror process
    def center_fft_bins(self, fft_tensor):
        left = fft_tensor[:, :, ::2]
        right = fft_tensor[:, :, 1::2]
        left = tf.reverse(left, axis=[-1])
        centered_fft = tf.concat([left, right], axis=-1)
        return centered_fft

    # reverse process of center_data()
    # un-mirrors and de-interlaces fft_tensors
    def decenter_fft_bins(self, fft_tensor):
        de_interlaced = np.zeros_like(fft_tensor)
        left = fft_tensor[:, :, :fft_tensor.shape[-1]//2]
        right = fft_tensor[:, :, fft_tensor.shape[-1]//2:]
        left = tf.reverse(left, axis=[-1])
        de_interlaced[:, :, ::2] = left
        de_interlaced[:, :, 1::2] = right
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
          print(f'FRAMES SHAPE {frames.shape}')
          print(f'AUDIO SHAPE {audio.shape}')
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
        raw_audio = tf.io.read_file(path)
        waveform = tf.audio.decode_wav(raw_audio, desired_channels=1, desired_samples=length)
        return waveform[0]

    def add_noise(self, audio):
        noise = tf.random.normal(audio.shape, mean=0.0, stddev=self.noise_std, dtype=tf.dtypes.float32)
        a_noise = tf.math.add(audio, noise)
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

    # main process for generating x,y pairs
    def gen_xy(self, frames, audio):
        v_clip, a_clip = self.get_random_clip(frames, audio)

        # USE ATTENTION MODEL TO EXTRACT ATTENTION HEATMAP
        v_clip = self.attention_extractor._inference(v_clip)
        v_clip = tf.expand_dims(v_clip, -1)

        a_clip_noise = self.add_noise(a_clip)
        ft_x = self.fft(a_clip_noise)
        ft_y = self.fft(a_clip)
        return ft_x, ft_y, v_clip

    def generator(self):
        while True:
            frames, audio = self.load_example_pair(self.all_vids[self.example_idx])

            self.example_idx += 1

            if self.example_idx > len(self.all_vids):
                self.example_idx = 0

            x_ft = []
            y_ft = []
            vid = []

            # creates batch on the same video
            for i in range(self.batch_size):
              print(f"generating batch {i} / {self.batch_size}")
              this_x_ft, this_y_ft, this_v_clip = self.gen_xy(frames, audio)
              x_ft.append(this_x_ft)
              y_ft.append(this_y_ft)
              vid.append(this_v_clip)
                
            x_ft = tf.convert_to_tensor(x_ft)
            y_ft = tf.convert_to_tensor(y_ft)

            if self.use_polar:
              x_ft = self.cartesian_to_polar(x_ft, concat_axis=1)
              y_ft = self.cartesian_to_polar(y_ft, concat_axis=1)
            else:
              x_ft = self.complex_to_ri(x_ft, 1)
              y_ft = self.complex_to_ri(y_ft, 1)

            if self.center_fft:
              x_ft = self.center_fft_bins(x_ft)
              y_ft = self.center_fft_bins(y_ft)

            vid = tf.convert_to_tensor(vid)
            vid = tf.cast(vid, dtype=tf.float32)
            vid /= 255.

            print(f"x ft {x_ft.shape} y ft {y_ft.shape} vid {vid.shape}")
            
            yield [[x_ft, vid], [y_ft, vid]]