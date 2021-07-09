from random import sample
import torch
from torch.utils import data
from av_dataset import STFT_Dataset
from avse_model import AV_Fusion_Model
import argparse
import matplotlib.pyplot as plt
import numpy as np
import wandb
import utilities
import torchvision.transforms.functional as TF

if __name__ == "__main__":
    wandb.init(project='AV_Fusion', entity='carl_m', config={"dataset":"MUSIC"})

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=4, metavar='N')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument("-e", '--epochs', type=int, default=10, help="epochs")
    parser.add_argument('--cb_freq', type=int, default=100, help="wandb callback frequency in epochs")
    parser.add_argument('--split', type=float, default=0.8, help="train/val split")
    
    parser.add_argument('--num_frames', type=int, default=6, help="number of consecutive video frames (converted to attention maps)")
    parser.add_argument('--frame_hop', type=int, default=2, help="hop between each clip example in a video")
    parser.add_argument('--framerate', type=int, default=30, help="video fps")
    parser.add_argument('--framesize', type=int, default=256, help="scaled video frame dims (converted to attention maps)")
    parser.add_argument('--p_size', type=int, default=64, help="downsampled phasegram size")
    parser.add_argument('--max_clip_len', type=int, default=None, help="maximum clip length to load (speed up loading)")
    
    parser.add_argument('--fft_len', type=int, default=256, help="size of fft")
    parser.add_argument('-a', '--hops_per_frame', type=int, default=8, help="num hops per frame (a)")
    parser.add_argument('--samplerate', type=int, default=16000, help="audio samplerate (dependent on dataset)")
    parser.add_argument('--normalize_fft', type=bool, default=True, help="normalize input fft by 1/n")
    parser.add_argument('--normalize_output_fft', type=bool, default=False, help="normalize output fft by 1/max(abs(fft))")
    parser.add_argument('--use_polar', type=bool, default=False, help="fft uses polar coordinates instead of rectangular")

    parser.add_argument('--data_path', type=str, default="E:/MUSICES", help="path to dataset")
    parser.add_argument('--saved_model', type=str, default=None, help="path to saved model state dict")
    parser.add_argument('--checkpoint', type=str, default=None, help="load model checkpoint")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints/", help="specify checkpoint save directory")

    args = parser.parse_args()
    config = wandb.config
    wandb.config.update(args)

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config.hop = int((config.samplerate/config.framerate)/config.hops_per_frame)
    audio_sample_len = int(config.hops_per_frame * config.hop * config.num_frames)
    num_fft_frames = audio_sample_len // config.hop

    preview_dims=(512, 4096)
    
    dataset = STFT_Dataset(
        samplerate = config.samplerate,
        fft_len = config.fft_len,
        hop = config.hop,
        audio_sample_len = audio_sample_len,
        normalize_input_fft = config.normalize_fft,
        normalize_output_fft = config.normalize_output_fft,
        use_polar = config.use_polar,
        data_path=config.data_path,
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

    x_stft, y_stft, audio = next(iter(train_gen))

    model = AV_Fusion_Model([config.batch_size, 2, num_fft_frames, config.fft_len//2], 
                        [config.batch_size, 1, config.num_frames, config.p_size*config.p_size],
                        config.hops_per_frame).to(DEVICE)
    


    mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.stft_autoencoder.parameters(), lr=config.learning_rate)
    
    if config.saved_model != None:
        print(f'Loading model weights from {config.saved_model}')
        model.load_state_dict(torch.load(config.saved_model))
    if config.checkpoint != None:
        print(f"loading model checkpoint from {config.checkpoint}")
        checkpoint = torch.load(config.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"error loading optimizer dict {e}")

    model.toggle_enc_grads(True)
    model.toggle_fusion_grads(False)

    best_loss = 1e5

    for e in range(config.epochs):

        model.train()

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
                print(f'epoch {e} step {i}/{len(train_gen)} loss {loss.sum()}')
        
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

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f'saving {wandb.run.name} checkpoint - {best_loss} best avg loss (val)')
            utilities.save_checkpoint(model.state_dict(), 
                            optimizer.state_dict(),
                            e, avg_loss,
                            wandb.run.name,
                            config.checkpoint_dir)
        
        stft_plot = utilities.stft_ae_image_callback(y_stft_val[0], yh_stft_val[0])
        p_audio = dataset.istft(yh_stft_val[0].cpu().detach())

        wandb.log( {
            "fft_frames_val": wandb.Image(stft_plot),
            "audio_input": wandb.Audio(audio_val[0], sample_rate=config.samplerate),
            "audio_output": wandb.Audio(p_audio, sample_rate=config.samplerate)
        } )

    utilities.save_model(f"saved_models/avf-a-ae-{wandb.run.name}", model, overwrite=True)