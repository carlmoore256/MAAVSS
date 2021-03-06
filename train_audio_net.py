from enum import auto
from random import sample
import torch
from torch.utils import data
from av_dataset import AV_Dataset, STFT_Dataset
from avse_model import AV_Fusion_Model
import argparse
import matplotlib.pyplot as plt
import numpy as np
import wandb
import utilities
import torchvision.transforms.functional as TF
from run_config import model_args
import time

if __name__ == "__main__":
    wandb.init(project='AV_Fusion', entity='carl_m', config={"dataset":"MUSIC"})
    config = wandb.config
    args = model_args()
    wandb.config.update(args)

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config.hop, audio_sample_len, config.num_fft_frames = utilities.calc_hop_size(
        config.num_frames,
        config.hops_per_frame, 
        config.framerate, 
        config.samplerate
    )

    preview_dims=(512, 4096)
    
    dataset = AV_Dataset(
        num_frames=config.num_frames,
        frame_hop=config.frame_hop,
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
        gen_stft=True,
        gen_video=False
    )

    train_split = int(len(dataset)*config.split)
    val_split = len(dataset) - train_split
    train_dset, val_dset = torch.utils.data.random_split(dataset, 
                                                    [train_split, val_split])

    train_gen = torch.utils.data.DataLoader(train_dset,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          num_workers=0)
    val_gen = torch.utils.data.DataLoader(val_dset,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          num_workers=0)

    x_stft, y_stft, audio = next(iter(train_gen))

    model = AV_Fusion_Model([config.batch_size, 2, config.num_fft_frames, config.fft_len//2], 
                        [config.batch_size, 1, config.num_frames, config.p_size*config.p_size],
                        config.hops_per_frame).to(DEVICE)

    mse_loss = torch.nn.MSELoss()

    model.toggle_phasegram_ae_grads(False)
    model.toggle_stft_ae_grads(True)
    model.toggle_fusion_grads(False)

    optimizer = torch.optim.Adam(model.stft_autoencoder.parameters(), lr=config.learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    ##################################
    
    if config.saved_model != None:
        print(f'Loading model weights from {config.saved_model}')
        model.load_state_dict(torch.load(config.saved_model), strict=False)
    if args.c or args.checkpoint is not None:
        utilities.load_checkpoint(model, optimizer, args.cp_dir, args.c, args.checkpoint, config.cp_load_opt)
        

    best_loss = 1e5

    ##################################

    for e in range(config.epochs):

        model.train()
        t1 = time.perf_counter()
        for i, d in enumerate(train_gen):
            optimizer.zero_grad()
            x_stft = d[0]
            y_stft = d[1]
            if y_stft.shape[0] != config.batch_size:
                continue
            audio = d[2]
            y_stft = y_stft.to(DEVICE)

            yh_stft = model.audio_ae_forward(y_stft)
            loss = mse_loss(yh_stft, y_stft)
            loss.backward()
            optimizer.step()
            
            wandb.log({ "loss": loss } )

            if i % config.cb_freq == 0:
                t2 = time.perf_counter()
                print(f'epoch {e} step {i}/{len(train_gen)} loss {loss.sum()} time {t2 - t1}')
                stft_plot = utilities.stft_ae_image_callback(y_stft[0], yh_stft[0])
                p_audio = dataset.istft(yh_stft[0].cpu().detach())

                wandb.log( {
                    "stft_train": wandb.Image(stft_plot),
                    "audio_input_train": wandb.Audio(audio[0], sample_rate=config.samplerate),
                    "audio_output_train": wandb.Audio(p_audio, sample_rate=config.samplerate)
                } )
                t1 = time.perf_counter()
            
            if not args.no_save and config.cp_freq != 0 and i % config.cp_freq == 0:
                print(f'saving {wandb.run.name} checkpoint')
                utilities.save_checkpoint(model.state_dict(), 
                                optimizer.state_dict(),
                                e, loss,
                                f'train_{wandb.run.name}',
                                config.cp_dir)
        
        
        model.eval()
        avg_loss = 0

        # validation
        for i, d_v in enumerate(val_gen):
            x_stft_val = d_v[0]
            y_stft_val = d_v[1]
            audio_val = d_v[2]
            y_stft_val = y_stft_val.to(DEVICE)
            with torch.no_grad():
                yh_stft_val = model.audio_ae_forward(y_stft_val)
                val_loss = mse_loss(yh_stft_val, y_stft_val)
            avg_loss += val_loss
            wandb.log({ "val_loss": val_loss })

        avg_loss /= len(val_gen)

        if avg_loss < best_loss and e > 0:
            best_loss = avg_loss
            if not args.no_save:
                print(f'saving {wandb.run.name} checkpoint - {best_loss} best avg loss (val)')
                utilities.save_checkpoint(model.state_dict(), 
                                optimizer.state_dict(),
                                e, avg_loss,
                                wandb.run.name,
                                config.cp_dir)
        
        stft_plot = utilities.stft_ae_image_callback(y_stft_val[0], yh_stft_val[0])
        p_audio = dataset.istft(yh_stft_val[0].cpu().detach())

        wandb.log( {
            "stft_val": wandb.Image(stft_plot),
            "audio_input_val": wandb.Audio(audio_val[0], sample_rate=config.samplerate),
            "audio_output_val": wandb.Audio(p_audio, sample_rate=config.samplerate)
        } )
    if not args.no_save:
        utilities.save_model(f"saved_models/avf-a-ae-{wandb.run.name}", model, overwrite=True)