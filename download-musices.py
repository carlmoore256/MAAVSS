# download videos from the musices dataset
import os
import utils
import subprocess
import glob
import argparse
import random
import threading
import time

def download_link(vid_id, directory):
    url = "http://www.youtube.com/watch?v=" + vid_id
    print(f"downloading {url}")
    # --postprocessor-args
    # -f '(mp4,webm)[height<480]'
    # -f "[filesize>10M]"
    # -f mp4/worstvideo/[filesize<10M]
    result = subprocess.Popen(["youtube-dl", "-o", f"{directory}/%(title)s.%(ext)s", url, "-f", "mp4/worstvideo/[filesize<10M]", "--socket-timeout", "5", "--restrict-filenames", "--abort-on-unavailable-fragment"],stdout=subprocess.PIPE)
    result.wait()
    print(result.communicate()[0])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type=str, default='./data/raw', help="save directory")

    args = parser.parse_args()

    links = utils.load_json("MUSICES.json")

    classes = links["videos"]

    all_keys = list(classes.keys())

    random.shuffle(all_keys)

    for k in all_keys:
        directory = os.path.join(args.dir, k)
        if not os.path.exists(directory):
            print(f'making directory: {directory}')
            os.makedirs(directory)

        this_class = classes[k]

        random.shuffle(this_class)

        files_in_dir = glob.glob(f'{directory}/*', recursive=True)

        files_in_dir = [os.path.split(f)[-1] for f in files_in_dir]

        for vid_id in this_class:
            # try:
            url = "http://www.youtube.com/watch?v=" + vid_id
            result = subprocess.Popen(["youtube-dl", "-o", f"{directory}/%(title)s.%(ext)s", url, "-f", "mp4/worstvideo/[filesize<10M]", "--socket-timeout", "5", "--restrict-filenames", "--get-filename"],stdout=subprocess.PIPE)
            result.wait()
            title = result.communicate()[0].decode("utf-8")
            title = os.path.split(title)[-1]
            title = str(title)[:-1]

            # title = utils.get_video_title(vid_id)
            # print(f"\n FILES IN DIR: {files_in_dir} \n")
            # except:
            #     print("error loading url, skipping")
            #     continue

            if title in files_in_dir:
                print(f'{title} already in {directory} skipping')
            elif title != "":
                t = threading.Thread(target=download_link, args=(vid_id, directory))
                # download_link(vid_id, directory)
                t.start()
                # set a timeout of 60 seconds
                t.join(60)