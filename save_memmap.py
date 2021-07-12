import numpy as np
import torch
import torchaudio
import utilities
import argparse
import pickle
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="E:/MUSICES", help="path to dataset")
    parser.add_argument('--output_file', type=str, default="audio_memmap.memmap")

    args = parser.parse_args()

    all_files = utilities.get_all_files(args.data_path, "mp4")
    audio_paths = []
    audio_lengths = []
    indexes = []
    total_len = 0

    for i, f in enumerate(all_files):
        path = utilities.get_paired_audio(f)
        audio_paths.append(path)
        info = torchaudio.info(path)
        audio_lengths.append(info.num_frames)
        indexes.append([total_len, total_len+info.num_frames]) # append starting position
        total_len += info.num_frames
        
    indexes = np.asarray(indexes)
    filename = os.path.join(args.data_path, args.output_file)
    print(f'creating memmap file {filename}')
    map = np.memmap(filename, dtype='float32', mode='w+', shape=(total_len))

    idx = 0
    for p, loc in zip(audio_paths, indexes):
        print(f'{idx}/{len(audio_paths)} writing {p} to memmap')
        audio = torchaudio.load(p)[0]
        if audio.shape[0] > 1:
            audio /= 2.
        audio = torch.sum(audio, dim=0)
        map[loc[0]:loc[1]] = audio
        idx += 1
    index_map = [all_files, indexes]
    utilities.save_cache_obj(os.path.join(args.data_path, "audio_index_map.obj"), index_map)
    print(f"finished writing audio to memmap")

