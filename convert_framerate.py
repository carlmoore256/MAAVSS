# converts video framerates all to specified rate
import utils
import subprocess
import os

all_vids = utilities.get_all_files("./data/raw", "mp4")

for v in all_vids:
    directory = os.path.split(v)[0]
    directory = f'{os.path.split(v)[0]}/video/'

    if not os.path.exists(directory):
        print(f'making directory: {directory}')
        os.makedirs(directory)

    filename = os.path.split(v)[-1]
    # filename = filename[:-4]
    # filename = f"{filename}.wav"
    output_file = f'{directory}{filename}'

    result = subprocess.Popen(["ffmpeg", "-i", v, "-filter:v", "fps=30", output_file],stdout=subprocess.PIPE)
    result.wait()