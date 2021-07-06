# save audio from videos to separate audio directory
import utilities
import subprocess
import os
import argparse

# def extract_audio(video_path, save_dir, sr):
#     filename = os.path.split(video_path)[-1]
#     filename = filename[:-4]
#     filename = f"{filename}.wav"
#     output_file = os.path.join(save_dir, filename)
#     # output_file = f'{save_dir}{filename}'

#     if os.path.isfile(output_file):
#         print(f"{output_file} already converted, skipping")
#     else:
#         # -vn : no video
#         # -ar : audio samplerate
#         result = subprocess.Popen(["ffmpeg", "-i", v, "-vn", "-ar", str(sr), output_file],stdout=subprocess.PIPE)
#         result.wait()

def convert_framerate(video_path, save_dir, framerate,sr):
    filename = os.path.split(video_path)[-1]
    output_file = os.path.join(save_dir, filename)
    fps = f'fps={framerate}'
    # -an : removes audio
    # command = ["ffmpeg", "-i", v, "-filter:v", "-an", fps, output_file]
    # print(command)
    result = subprocess.Popen(["ffmpeg", "-hwaccel", "cuda", "-i", v, "-filter:v", fps, "-ar", str(sr), output_file], stdout=subprocess.PIPE)
    result.wait()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dlpath", type=str, default='./data/raw', help="path to downloaded videos")
    parser.add_argument("--outpath", type=str, default='./data/processed', help="path to output saved files")
    parser.add_argument("-sr", type=int, default=16000, help="samplerate to convert to")
    parser.add_argument("-fr", type=int, default=30, help="framerate to convert to")

    args = parser.parse_args()

    all_vids = utilities.get_all_files(args.dlpath, "mp4")
    all_vids += utilities.get_all_files(args.dlpath, "mkv")

    for v in all_vids:
        directory = os.path.split(v)[0]

        class_name = os.path.split(directory)[-1]
        print(class_name)

        # audio_dir = os.path.join(args.outpath, class_name, "audio")
        video_output_dir = os.path.join(args.outpath, class_name)
        # print(f'audio dir {audio_dir} video dir {video_dir}')

        # if not os.path.exists(audio_dir):
        #     print(f'creating dir {audio_dir}')
        #     os.makedirs(audio_dir)

        if not os.path.exists(video_output_dir):
            print(f'creating dir {video_output_dir}')
            os.makedirs(video_output_dir)

        # extract_audio(v, audio_dir, args.sr)
        print(f"output dir {video_output_dir}")
        convert_framerate(v, video_output_dir, args.fr, args.sr)