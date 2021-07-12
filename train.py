from random import sample
import torch
from torch.utils import data
from av_dataset import AV_Dataset
from avse_model import AV_Fusion_Model
import matplotlib.pyplot as plt
import numpy as np
import wandb
import utilities
from run_config import model_args
import torchvision.transforms.functional as TF

def stft_ae_step(y_stft, model, optimizer, loss_f):
    optimizer.zero_grad()
    yh_stft = model.audio_ae_forward(y_stft)
    loss = loss_f(yh_stft, y_stft)
    loss.backward()
    optimizer.step()
    return loss, yh_stft

def phasegram_step(attn, p_size, model, optimizer, loss_f):
    # attention frames generate the phasegram
    y_phasegram = utilities.video_phasegram(attn, 
                            resize=(p_size, p_size),
                            diff=True,
                            cumulative=True)
    yh_phasegram = model.visual_ae_forward(y_phasegram)

    loss = loss_f(yh_phasegram, y_phasegram)
    loss.backward()
    optimizer.step()
    return loss, yh_phasegram

def av_fusion_step(x_stft, y_stft, attn, p_size, model, optimizer, loss_f):
    # attention frames generate the phasegram
    y_phasegram = utilities.video_phasegram(attn, 
                            resize=(p_size, p_size),
                            diff=True,
                            cumulative=True)

    yh_stft, yh_phasegram, av_fused = model(x_stft, y_phasegram)

    a_loss = loss_f(yh_phasegram, y_phasegram)
    v_loss = loss_f(yh_stft, y_stft)
    loss = a_loss + v_loss
    loss.backward()
    optimizer.step()
    return loss, a_loss, v_loss, yh_stft, yh_phasegram, av_fused
    

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

    model = AV_Fusion_Model([config.batch_size, 2, config.num_fft_frames, config.fft_len//2], 
                        [config.batch_size, 1, config.num_frames, config.p_size*config.p_size],
                        config.hops_per_frame,
                        ).to(DEVICE)

    mse_loss = torch.nn.MSELoss()
        
    model.toggle_phasegram_ae_grads(False)
    model.toggle_stft_ae_grads(False)
    model.toggle_fusion_grads(True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    if config.saved_model != None:
        print(f'Loading model weights from {config.saved_model}')
        model.load_state_dict(torch.load(config.saved_model), strict=False)
    if args.c or args.checkpoint is not None:
        utilities.load_checkpoint(model, optimizer, args.cp_dir, args.c, args.checkpoint, config.cp_load_opt)
        
    t_gen = iter(train_gen)
    v_gen = iter(val_gen)
    last_loss = 1e5

    train_a = True
    train_v = False
    train_av = False

    for e in range(config.epochs):
        if e + 1 * config.steps_per_epoch > len(train_gen):
            t_gen = iter(train_gen)
        if e + 1 * config.val_steps > len(val_gen):
            v_gen = iter(val_gen)

        model.train()

        for i in range(config.steps_per_epoch):
            optimizer.zero_grad()

            # if i == len(t_gen):
            #     t_gen = iter(train_gen)

            if train_a:
                _, y_stft, audio = next(t_gen)
                loss, yh_stft = stft_ae_step(
                    y_stft.to(DEVICE),
                    model,
                    optimizer,
                    mse_loss
                )
                wandb.log({"stft loss": loss})
            
            if train_v:
                attn, video = next(t_gen)
                loss, yh_phasegram = phasegram_step(
                    attn.to(DEVICE),
                    config.p_size,
                    model,
                    optimizer,
                    mse_loss
                )
                wandb.log({"phasegram_loss":loss } )

            if train_av:
                x_stft, y_stft, attn, audio, video = next(t_gen)
                loss, a_loss, v_loss, yh_stft, yh_phasegram, av_fused = av_fusion_step(
                    x_stft.to(DEVICE), 
                    y_stft.to(DEVICE), 
                    attn.to(DEVICE), 
                    config.p_size, 
                    model, 
                    optimizer, 
                    mse_loss
                )

            # wandb.log({ "loss": loss,
            #             "stft_loss":a_loss,
            #             "phasegram_loss":v_loss } )

            # if i % config.cb_freq == 0:
            #     print(f'epoch {e} step {i}/{config.steps_per_epoch} loss {loss.sum()} a_loss {a_loss} v_loss {v_loss}')
            #     stft_plot = utilities.stft_ae_image_callback(y_stft[0], yh_stft[0])
            #     frame_plot = utilities.video_phasegram_image(
            #         y_phasegram[0], yh_phasegram[0], attn[0], preview_dims)
            #     wandb.log( {"frames": wandb.Image(frame_plot),
            #                 "stft": wandb.Image(stft_plot)} )
        
        model.eval()
        avg_loss = 0

        # validation
        for i in range(config.val_steps):
            x_stft_v, y_stft_v, attn_v, audio_v, video_v = next(v_gen)
            x_stft_v = x_stft_v.to(DEVICE)
            y_stft_v = y_stft_v.to(DEVICE)
            attn_v = attn_v.to(DEVICE)
            with torch.no_grad():
                y_pgram_v = utilities.video_phasegram(attn_v, 
                        resize=(config.p_size, config.p_size),
                        diff=True,
                        cumulative=True)
                y_pgram_v = y_pgram_v.to(DEVICE)
                yh_stft_v, yh_pgram_v, av_fused_v = model(x_stft_v, y_pgram_v)
                # print(f"yh_stft_v {yh_stft_v.device} y_stft_v {y_stft_v.device}")
                a_loss_val = mse_loss(yh_stft_v, y_stft_v.to(DEVICE))
                v_loss_val = mse_loss(yh_pgram_v, y_pgram_v)
                val_loss = a_loss_val + v_loss_val
            avg_loss += val_loss
            wandb.log({ "val_loss": val_loss,
                        "val_loss_stft":a_loss_val,
                        "val_loss_phasegram":v_loss_val })

        avg_loss /= config.val_steps

        if avg_loss < last_loss and e > 0:
            if not args.no_save:
                utilities.save_checkpoint(model.state_dict(), 
                                        optimizer.state_dict(),
                                        e, avg_loss,
                                        wandb.run.name,
                                        config.cp_dir)
        last_loss = avg_loss
        
        frame_plot = utilities.video_phasegram_image(
            y_pgram_v[0], yh_pgram_v[0], attn_v[0], preview_dims)
        stft_plot = utilities.stft_ae_image_callback(y_stft_v[0], yh_stft_v[0])
        p_audio = dataset.istft(yh_stft_v[0].cpu().detach())
        latent_plot = utilities.latent_fusion_image_callback(av_fused_v[0].cpu().detach().numpy())

        wandb.log( {
            "video_frames_val": wandb.Image(frame_plot),
            "stft_frames_val": wandb.Image(stft_plot),
            "latent_plot": wandb.Image(latent_plot),
            "audio_target": wandb.Audio(audio_v[0], sample_rate=config.samplerate),
            "audio_output": wandb.Audio(p_audio, sample_rate=config.samplerate)
        } )
    if not args.no_save:
        utilities.save_model(f"saved_models/avf-v-ae-{wandb.run.name}.pt", model, overwrite=True)