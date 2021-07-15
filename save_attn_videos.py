import enum
from av_dataset import AV_Dataset
from run_config import model_args
import wandb
import torch
import os
import glob
import torchvision

def save_frames(frames, save_path, offset=0, check_exists=False):
    attn = frames.squeeze(0)
    for i, frame in enumerate(attn):
        img_filename = os.path.join(save_path, f'img_{i+offset:05d}.jpg')
        if check_exists:
           if os.path.isfile(img_filename):
               print(f'{img_filename} already exists, skipping...')
               continue
        img_path = os.path.join(save_path, img_filename)
        torchvision.utils.save_image(frame, img_path)

if __name__ == "__main__":

    base_dir = "E:/MUSICES_ATTN"

    train_path = os.path.join(base_dir, "train")
    val_path = os.path.join(base_dir, "val")

    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    if not os.path.isdir(val_path):
        os.mkdir(val_path)

    args = model_args()
    with wandb.init(project='AV-Fusion-AVSE', entity='carl_m', config=args):
        config = wandb.config

        dataset = AV_Dataset(
            num_frames=64,
            frame_hop=64,
            framerate=config.framerate,
            samplerate = config.samplerate,
            fft_len=config.fft_len,
            hops_per_frame=config.hops_per_frame,
            noise_std=config.noise_scalar,
            use_polar = config.use_polar,
            normalize_input_fft = config.normalize_fft,
            normalize_output_fft = config.normalize_output_fft,
            autocontrast=config.autocontrast,
            compress_audio=config.compress_audio,
            shuffle_files=True,
            data_path=config.data_path,
            max_clip_len=config.max_clip_len,
            gen_stft=False,
            gen_video=True,
            trim_stft_end=False,
            return_video_path=True
        )

        train_split = int(len(dataset)*config.split)
        val_split = len(dataset) - train_split
        train_dset, val_dset = torch.utils.data.random_split(dataset, [train_split, val_split])


        train_gen = torch.utils.data.DataLoader(train_dset,
                                            batch_size=config.batch_size,
                                            shuffle=False,
                                            num_workers=0)
        val_gen = torch.utils.data.DataLoader(val_dset,
                                                batch_size=config.batch_size,
                                                shuffle=False,
                                                num_workers=0)

        t_gen = iter(train_gen)
        v_gen = iter(val_gen)


        for i, [attns, _, [paths, clip_idxs]] in enumerate(t_gen):

            for attn, p, idx in zip(attns, paths, clip_idxs):
                name = os.path.split(p)[-1]
                frames_new_path = os.path.join(train_path, name[:-4])

                true_idx = config.frame_hop * idx

                if not os.path.isdir(frames_new_path):
                    print(f'saving frames for {name} in {frames_new_path}')
                    os.mkdir(frames_new_path)
                    save_frames(attn, frames_new_path, offset=true_idx)
                else:
                    save_frames(attn, frames_new_path, offset=true_idx, check_exists=True)
