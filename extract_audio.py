# save audio from videos to separate audio directory
import utils
import subprocess
import os
import argparse


def extract(all_vids):
    for v in all_vids:
        directory = os.path.split(v)[0]
        directory = f'{os.path.split(v)[0]}/audio/'

        if not os.path.exists(directory):
            print(f'making directory: {directory}')
            os.makedirs(directory)

        filename = os.path.split(v)[-1]
        filename = filename[:-4]
        filename = f"{filename}.wav"
        output_file = f'{directory}{filename}'

        if os.path.isfile(output_file):
            print(f"{output_file} already converted, skipping")
        else:
            # -vn : no video
            # -ar : audio samplerate
            result = subprocess.Popen(["ffmpeg", "-i", v, "-vn", "-ar", "16000", output_file],stdout=subprocess.PIPE)
            result.wait()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default='./data/raw', help="save directory")
    args = parser.parse_args()
    extract(utils.get_all_files(args.dir, "mp4"))
