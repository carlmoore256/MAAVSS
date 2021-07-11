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
    
    dataset = AV_Dataset(
        frames_per_clip = config.num_frames,
        frame_hop = config.frame_hop,
        framesize = config.framesize,
        samplerate=config.samplerate,
        fft_len=config.fft_len,
        hop=config.hop,
        hops_per_frame=config.hops_per_frame,
        noise_std=config.noise_scalar,
        use_polar=config.use_polar,
        normalize_input_fft=config.normalize_fft,
        normalize_output_fft=config.normalize_output_fft,
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

    x_stft, y_stft, attn, audio, video = next(iter(train_gen))

    model = AV_Fusion_Model([config.batch_size, 2, num_fft_frames, config.fft_len//2], 
                        [config.batch_size, 1, config.num_frames, config.p_size*config.p_size],
                        config.hops_per_frame,
                        ).to(DEVICE)

    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # disable the encoder/decoder head grads for the network
    # model.toggle_enc_grads(False)
    # model.toggle_dec_grads(False)
    # model.toggle_fusion_grads(True)


    # for name, param in model.named_parameters():
    #     print(f"\nname")
    #     if param.requires_grad:
    #         print(param.shape)

    if config.saved_model != None:
        print(f'Loading model weights from {config.saved_model}')
        model.load_state_dict(torch.load(config.saved_model), strict=False)

    print(f'LOAD OPT {args.cp_load_opt}')
    if args.c or args.checkpoint is not None:
        utilities.load_checkpoint(model, optimizer, args.cp_dir, args.c, args.checkpoint, config.cp_load_opt)

    for param in model.stft_autoencoder.parameters():
        param.requires_grad = False
    for param in model.phasegram_autoencoder.parameters():
        param.requires_grad = False

        
    t_gen = iter(train_gen)
    v_gen = iter(val_gen)
    last_loss = 1e5

    for e in range(config.epochs):
        if e + 1 * config.steps_per_epoch > len(train_gen):
            t_gen = iter(train_gen)
        if e + 1 * config.val_steps > len(val_gen):
            v_gen = iter(val_gen)

        model.train()

        for i in range(config.steps_per_epoch):
            optimizer.zero_grad()
            x_stft, y_stft, attn, audio, video = next(t_gen)
            y_attn = attn.to(DEVICE)
            x_stft = x_stft.to(DEVICE)
            y_stft = y_stft.to(DEVICE)
            # attention frames generate the phasegram
            y_phasegram = utilities.video_phasegram(y_attn, 
                                    resize=(config.p_size, config.p_size),
                                    diff=True,
                                    cumulative=True)

            yh_stft, yh_phasegram, av_fused = model(x_stft, y_phasegram)

            a_loss = mse_loss(yh_phasegram, y_phasegram)
            v_loss = mse_loss(yh_stft, y_stft)
            loss = a_loss + v_loss
            loss.backward()
            optimizer.step()
            
            wandb.log({ "loss": loss,
                        "stft_loss":a_loss,
                        "phasegram_loss":v_loss } )

            if i % config.cb_freq == 0:
                print(f'epoch {e} step {i}/{config.steps_per_epoch} loss {loss.sum()} a_loss {a_loss} v_loss {v_loss}')
                stft_plot = utilities.stft_ae_image_callback(y_stft[0], yh_stft[0])
                frame_plot = utilities.video_phasegram_image(
                    y_phasegram[0], yh_phasegram[0], attn[0], preview_dims)
                wandb.log( {"frames": wandb.Image(frame_plot),
                            "stft": wandb.Image(stft_plot)} )
        
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