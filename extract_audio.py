# save audio from videos to separate audio directory
import utils
import subprocess
import os

all_vids = utils.get_all_files("./data/raw", "mp4")

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
        result = subprocess.Popen(["ffmpeg", "-i", v, "-vn", "-ar", "44100", output_file],stdout=subprocess.PIPE)
        result.wait()