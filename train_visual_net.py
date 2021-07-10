from random import sample
import torch
from torch.utils import data
from av_dataset import Video_Dataset
from avse_model import AV_Fusion_Model
import argparse
import matplotlib.pyplot as plt
import numpy as np
import wandb
import utilities
import torchvision.transforms.functional as TF
from run_config import model_args

if __name__ == "__main__":
    wandb.init(project='AV_Fusion', entity='carl_m', config={"dataset":"MUSIC"})
    config = wandb.config
    args = model_args()
    wandb.config.update(args)

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config.hop = int((config.samplerate/config.framerate)/config.hops_per_frame)
    audio_sample_len = int(config.hops_per_frame * config.hop * config.num_frames)
    num_fft_frames = audio_sample_len // config.hop

    preview_dims=(512, 4096)
    
    dataset = Video_Dataset(
        frames_per_clip = config.num_frames,
        frame_hop = config.frame_hop,
        framesize = config.framesize,
        autocontrast = config.autocontrast,
        data_path = config.data_path,
        max_clip_len=config.max_clip_len
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

    attn, video = next(iter(train_gen))

    model = AV_Fusion_Model([config.batch_size, 2, num_fft_frames, config.fft_len//2], 
                        [config.batch_size, 1, config.num_frames, config.p_size*config.p_size],
                        config.hops_per_frame).to(DEVICE)

    mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.phasegram_autoencoder.parameters(), lr=config.learning_rate)

    ##################################
    
    if config.saved_model != None:
        print(f'Loading model weights from {config.saved_model}')
        model.load_state_dict(torch.load(config.saved_model), strict=False)
    if args.c or args.checkpoint is not None:
        utilities.load_checkpoint(model, optimizer, args.cp_dir, args.c, args.checkpoint, config.cp_load_opt)

    model.toggle_enc_grads(True)
    model.toggle_fusion_grads(False)

    t_gen = iter(train_gen)
    v_gen = iter(val_gen)
    last_loss = 1e6

    for e in range(config.epochs):
        if e + 1 * config.steps_per_epoch > len(train_gen):
            t_gen = iter(train_gen)
        if e + 1 * config.val_steps > len(val_gen):
            v_gen = iter(val_gen)

        model.train()

        for i in range(config.steps_per_epoch):
            optimizer.zero_grad()
            attn, video = next(t_gen)
            y_attn = attn.to(DEVICE)
            # attention frames generate the phasegram
            y_phasegram = utilities.video_phasegram(y_attn, 
                                    resize=(config.p_size, config.p_size),
                                    diff=True,
                                    cumulative=True)
            yh_phasegram = model.visual_ae_forward(y_phasegram)

            loss = mse_loss(yh_phasegram, y_phasegram)
            loss.backward()
            optimizer.step()
            
            wandb.log({ "loss": loss } )

            if i % config.cb_freq == 0:
                print(f'epoch {e} step {i}/{config.steps_per_epoch} loss {loss.sum()}')
                frame_plot = utilities.video_phasegram_image(
                    y_phasegram[0], yh_phasegram[0], attn[0], preview_dims)
                wandb.log( {"frames": wandb.Image(frame_plot)} )
        
        model.eval()
        avg_loss = 0

        # validation
        for i in range(config.val_steps):
            attn_val, video_val = next(v_gen)
            y_attn_val = attn_val.to(DEVICE)
            with torch.no_grad():
                y_pgram_val = utilities.video_phasegram(y_attn_val, 
                        resize=(config.p_size, config.p_size),
                        diff=True,
                        cumulative=True)
                yh_pgram_val = model.visual_ae_forward(y_pgram_val)
                val_loss = mse_loss(yh_pgram_val, y_pgram_val)
            avg_loss += val_loss
            wandb.log({ "val_loss": val_loss })

        avg_loss /= config.val_steps

        if avg_loss < last_loss:
            if not args.no_save:
                print(f'saving {wandb.run.name} checkpoint - {avg_loss} avg loss (val)')
                utilities.save_checkpoint(model.state_dict(), 
                                        optimizer.state_dict(),
                                        e, avg_loss,
                                        wandb.run.name,
                                        config.cp_dir)
        last_loss = avg_loss
        
        frame_plot = utilities.video_phasegram_image(
            y_pgram_val[0], yh_pgram_val[0], y_attn_val[0], preview_dims)
        wandb.log( {
            "video_frames_val": wandb.Image(frame_plot)
        } )
    if not args.no_save:
        utilities.save_model(f"saved_models/avf-v-ae-{wandb.run.name}", model, overwrite=True)