import tensorflow as tf
import utils
import cv2
import glob
import os
import numpy as np

class DataGenerator():

    # num vid frames -> how many video frames to extract flow data from, size of 4D tensor
    def __init__(self, batch_size, num_vid_frames=4, data_path="./data/raw"):
        self.batch_size = batch_size
        self.num_vid_frames = num_vid_frames

        self.all_vids = utils.get_all_files(data_path, "mp4")
        self.all_audio = utils.get_all_files(data_path, "wav")
        self.current_av_pair = self.load_example_pair(self.all_vids[0])


    # load audio and video exaple pair
    def load_example_pair(self, vid_path):

        split_path = os.path.split(vid_path)
        name = split_path[-1][:-4]
        name = name + ".wav"
        audio_path = os.path.join(split_path[0], "audio/", name)
        print(audio_path)

        frames = self.load_video(vid_path)
        audio = self.load_audio(audio_path)

        print(f'FRAMES SHAPE {frames.shape}')
        print(f'AUIDO SHAPE {audio.shape}')

        return frames, audio


    # returns a random num_vid_frames frames from video, returns spectrogram and vid
    def get_random_example(self, frames, audio):


    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            frames.append(frame)
            if ret is False:
                break
        cap.release()
        # frames = tf.convert_to_tensor(frames)
        frames = np.asarray(frames)
        print("\n LOADED VID \n")
        return frames

    def load_audio(self, path):
        raw_audio = tf.io.read_file(path)
        waveform = tf.audio.decode_wav(raw_audio, desired_channels=1)
        return waveform[0]

dg = DataGenerator(32)