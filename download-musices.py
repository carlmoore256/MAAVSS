# download videos from the musices dataset
import os
import utils
import subprocess
import glob
import argparse


def download_link(vid_id, directory):
    url = "http://www.youtube.com/watch?v=" + vid_id
    print(f"downloading {url}")
    result = subprocess.Popen(["youtube-dl", "-o", f"{directory}/%(title)s.%(ext)s", url],stdout=subprocess.PIPE)
    result.wait()
    print(result.communicate()[0])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type=str, default='./data/raw', help="save directory")

    args = parser.parse_args()

    links = utils.load_json("MUSICES.json")

    classes = links["videos"]

    for k in classes.keys():
        directory = os.path.join(args.dir, k)
        if not os.path.exists(directory):
            print(f'making directory: {directory}')
            os.makedirs(directory)

        this_class = classes[k]
        files_in_dir = glob.glob(f'{directory}/*', recursive=True)

        files_in_dir = [os.path.split(f)[-1] for f in files_in_dir]

        for vid_id in this_class:
            try:
                title = utils.get_video_title(vid_id)
            except:
                print("error loading url, skipping")
                continue
            
            if title in files_in_dir:
                print(f'{title} already in {directory} skipping')
            else:
                download_link(vid_id, directory)