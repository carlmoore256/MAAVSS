# save audio from videos to separate audio directory
import utilities
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default='./data/raw', help="save directory")
    parser.add_argument("--reject", type=str, default='./data/reject', help="reject directory")
    args = parser.parse_args()
    all_files = utilities.get_all_files(args.dir, "mp4")
    audio_paths = [[utilities.get_paired_audio(f), f] for f in all_files]

    print(f'finished extracting {len(audio_paths)} audio streams')
    for a in audio_paths:
        if a[0] is None:
            filename = os.path.split(a[1])[-1]
            # move the file to the reject folder
            os.rename(a[1], os.path.join(args.reject, filename))