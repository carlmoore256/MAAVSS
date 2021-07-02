# save audio from videos to separate audio directory
import utils
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default='./data/raw', help="save directory")
    args = parser.parse_args()
    all_files = utils.get_all_files(args.dir, "mp4")
    audio_paths = [utils.get_paired_audio(f) for f in all_files]
    print(f'finished extracting {len(audio_paths)} audio streams')