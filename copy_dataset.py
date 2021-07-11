import utilities
import os
import random
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--destination', type=str, default="/content/MUSICES")
    parser.add_argument('-s', '--source', type=str, default="/content/drive/MyDrive/MagPhaseLAVSE/MUSICES")
    parser.add_argument('-b', '--cp_buff', type=int, default=32*1024*1)

    args = parser.parse_args()

    copy_buffer = args.cp_buff
    new_root = args.destination
    all_files = utilities.get_all_files(args.source, "mp4")
    random.shuffle(all_files)

    for i, f in enumerate(all_files):
        print(f'copying file {i} / {len(all_files)}')
        audio_path = utilities.get_paired_audio(f, extract=False)
        vid_split = f.split(os.sep)
        new_path_vid = os.path.join(new_root, vid_split[-2], vid_split[-1])
        audio_split = audio_path.split(os.sep)
        audio_subdir = os.path.join(new_root, vid_split[-2], audio_split[-2])
        new_path_audio = os.path.join(audio_subdir, audio_split[-1])

        print(new_path_vid)
        print(new_path_audio)

        if not os.path.exists(audio_subdir):
            print(f'making audio directory: {audio_subdir}')
            os.makedirs(audio_subdir)

        if os.path.exists(new_path_vid):
            print(f'skipping {new_path_vid}, file already exists...')
        else:
            print(f'copying video to {new_path_vid}')
            with open(f, 'rb') as fda:
                with open(new_path_vid, 'wb') as fdb:
                    shutil.copyfileobj(fda, fdb, length=copy_buffer)

        if os.path.exists(new_path_audio):
            print(f'skipping {new_path_audio}, file already exists...')
        else:
            print(f'copying audio to {new_path_audio}')
            with open(audio_path, 'rb') as fda:
                with open(new_path_audio, 'wb') as fdb:
                    shutil.copyfileobj(fda, fdb, length=copy_buffer)
