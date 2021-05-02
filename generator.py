import tensorflow as tf
import glob

class DataGenerator():

    def __init__(self, batch_size, data_path="./data/raw"):
        self.batch_size = batch_size
        all_vids = glob.glob(f"{data_path}/*/**.mp4", recursive=True)


    def load_video

