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

if __name__ == "__main__":
    wandb.init(project='AV_Fusion', entity='carl_m', config={"dataset":"MUSIC"})

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=4, metavar='N')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument("-e", '--epochs', type=int, default=10, help="epochs")
    parser.add_argument('--data_path', type=str, default="data/raw", help="path to dataset")
    parser.add_argument('--num_frames', type=int, default=6, help="number of consecutive video frames (converted to attention maps)")
    parser.add_argument('--frame_hop', type=int, default=2, help="hop between each clip example in a video")
    parser.add_argument('--framerate', type=int, default=30, help="video fps")
    parser.add_argument('--framesize', type=int, default=256, help="scaled video frame dims (converted to attention maps)")
    parser.add_argument('--p_size', type=int, default=64, help="downsampled phasegram size")
    parser.add_argument('--autocontrast', type=bool, default=False, help="automatic video contrast")
    
    parser.add_argument('--fft_len', type=int, default=256, help="size of fft")
    parser.add_argument('-a', '--hops_per_frame', type=int, default=8, help="num hops per frame (a)")
    parser.add_argument('--samplerate', type=int, default=16000, help="audio samplerate (dependent on dataset)")

    parser.add_argument('--cb_freq', type=int, default=100, help="wandb callback frequency in epochs")
    parser.add_argument('--max_clip_len', type=int, default=None, help="maximum clip length to load (speed up loading)")
    parser.add_argument('--split', type=float, default=0.8, help="train/val split")
    parser.add_argument('--saved_model', type=str, default=None, help="path to saved model state dict")

    args = parser.parse_args()
    config = wandb.config
    wandb.config.update(args)

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config.hop = int((config.samplerate/config.framerate)/config.hops_per_frame)
    audio_sample_len = int(config.hops_per_frame * config.hop * config.num_frames)
    num_fft_frames = audio_sample_len // config.hop
    
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
    
    if config.saved_model != None:
        print(f'Loading model weights from {config.saved_model}')
        model.load_state_dict(torch.load(config.saved_model))

    mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.phasegram_autoencoder.parameters(), lr=config.learning_rate)

    # model.toggle_av_grads(False)
    # model.toggle_audio_grads(False)

    for e in range(config.epochs):
        model.train()
        avg_loss = 0

        for i, d in enumerate(train_gen):
            optimizer.zero_grad()

            attn = d[0]
            video = d[1]
            y_attn = attn.to(DEVICE)
            # attention frames generate the phasegram
            y_phasegram = utilities.video_phasegram(y_attn)
            yh_phasegram = model.visual_ae_forward(y_phasegram)
            print(f'y_phasegram sh {y_phasegram.shape} yh {yh_phasegram.shape}')

            loss = mse_loss(yh_phasegram, y_phasegram)
            loss.backward()
            optimizer.step()
            
            wandb.log({ "loss": loss } )

            print(f'epoch {e} step {i}/{len(train_gen)} loss {loss.sum()}')

            
            if i % config.cb_freq == 0:
                fig=plt.figure(figsize=(config.num_frames, 3))
                plt.tight_layout()

                pg_img = y_phasegram.permute(0, 2, 1)

                # cols = config.num_frames
                # rows = 3
                # for a in range(cols * rows):
                #     if a < cols:
                #         img = video[0, 0, a, :, :].cpu().detach().numpy()
                #     elif a < cols * 2:
                #         img = attn[0, 0, a%cols, :, :].cpu().detach().numpy()
                #     else:
                #         img = yh_attn[0, 0, a%cols, :, :].cpu().detach().numpy()

                #     fig.add_subplot(rows, cols, a+1)
                #     plt.xticks([])
                #     plt.yticks([])
                #     plt.imshow(img)
                fig.canvas.draw()
                frame_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                frame_plot = frame_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                wandb.log( {
                    "video_frames": wandb.Image(frame_plot)
                } )
        
        model.eval()

        # validation
        for i, d in enumerate(val_gen):
            attn_val = d[0]
            video_val = d[1]
            y_attn_val = y_attn_val.to(DEVICE)
            with torch.no_grad():
                yh_attn_val = model.visual_ae_forward(y_attn_val)
                val_loss = mse_loss(yh_attn_val, y_attn_val)
            wandb.log({ "val_loss": val_loss } )

        avg_loss /= len(train_gen)
        if avg_loss < last_loss:
            print(f'saving {wandb.run.name} checkpoint - {avg_loss} avg loss (val)')
            utilities.save_model(f"checkpoints/av-fusion-pgram-ae-{wandb.run.name}", model)
        last_loss = avg_loss

        # if e % config.cb_freq == 0:
        fig=plt.figure(figsize=(8, 5))
        plt.tight_layout()

        cols = config.num_frames
        rows = 3
        for a in range(cols * rows):
            if a < cols:
                img = video_val[0, 0, a, :, :].cpu().detach().numpy()
            elif a < cols * 2:
                img = attn_val[0, 0, a%cols, :, :].cpu().detach().numpy()
            else:
                img = yh_attn_val[0, 0, a%cols, :, :].cpu().detach().numpy()

        fig.add_subplot(rows, cols, a+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
        fig.canvas.draw()
        frame_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        frame_plot = frame_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        wandb.log( {
            "video_frames_val": wandb.Image(frame_plot)
        } )

    utilities.save_model(f"saved_models/av-fusion-pgram-ae-{wandb.run.name}", model)