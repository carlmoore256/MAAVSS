from random import sample
import torch
from torch.utils import data
from av_dataset import STFT_Dataset
from avse_model import AV_Model_STFT
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
    parser.add_argument('--fft_len', type=int, default=256, help="size of fft")
    parser.add_argument('-a', '--hops_per_frame', type=int, default=8, help="num hops per frame (a)")
    parser.add_argument('--samplerate', type=int, default=16000, help="audio samplerate (dependent on dataset)")
    parser.add_argument('--center_fft', type=bool, default=True, help="interlace and center fft")
    parser.add_argument('--use_polar', type=bool, default=False, help="fft uses polar coordinates instead of rectangular")
    parser.add_argument('--normalize_fft', type=bool, default=True, help="normalize input fft by 1/n")
    parser.add_argument('--normalize_output_fft', type=bool, default=False, help="normalize output fft by 1/max(abs(fft))")
    parser.add_argument('--noise_scalar', type=float, default=0.1, help="scale gaussian noise by N for data augmentation (applied to x)")
    parser.add_argument('--cb_freq', type=int, default=100, help="wandb callback frequency in epochs")
    parser.add_argument('--max_clip_len', type=int, default=None, help="maximum clip length to load (speed up loading)")
    parser.add_argument('--split', type=float, default=0.8, help="train/val split")
    parser.add_argument('--saved_model', type=str, default=None, help="path to saved model state dict")

    args = parser.parse_args()
    config = wandb.config
    wandb.config.update(args)

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config.hop = int((config.samplerate/config.framerate)/config.hops_per_frame)

    dataset = STFT_Dataset(
        samplerate = config.samplerate,
        fft_len = config.fft_len,
        hop = config.hop,
        audio_sample_len = int(config.hops_per_frame * config.hop * config.num_frames),
        noise_std = config.noise_scalar,
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
                                          shuffle=True)
    val_gen = torch.utils.data.DataLoader(val_dset,
                                          batch_size=config.batch_size,
                                          shuffle=True)

    x_stft, y_stft, audio = next(iter(train_gen))

    model = AV_Model_STFT(x_stft.shape, 
                        [config.batch_size, 1, config.num_frames, config.framesize, config.framesize],
                        config.hops_per_frame).to(DEVICE)
    
    if config.saved_model != None:
        print(f'Loading model weights from {config.saved_model}')
        model.load_state_dict(torch.load(config.saved_model))

    mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.audio_ae.parameters(), lr=config.learning_rate)

    model.toggle_av_grads(False)
    model.toggle_visual_grads(False)

    for e in range(config.epochs):
        model.train()
        
        for i, d in enumerate(train_gen):
            optimizer.zero_grad()
            x_stft = d[0]
            y_stft = d[1]
            audio = d[2]
            y_stft = y_stft.to(DEVICE)
            yh_stft = model.audio_ae_forward(y_stft)
            loss = mse_loss(yh_stft, y_stft)
            loss.backward()
            optimizer.step()
            
            wandb.log({ "loss": loss } )

        print(f'epoch {e} step {i} loss {loss.sum()}')

        # validation
        for i, d in enumerate(val_gen):
            x_stft_val = d[0]
            y_stft_val = d[1]
            audio_val = d[2]
            y_stft_val = y_stft_val.to(DEVICE)
            model.eval()
            with torch.no_grad():
                yh_stft_val = model.audio_ae_forward(y_stft_val)
                val_loss = mse_loss(yh_stft_val, y_stft_val)
            wandb.log({ "val_loss": val_loss } )

        if e % config.cb_freq == 0:
            print(f'epoch {e} step {i} loss {loss.sum()}')
            fig=plt.figure(figsize=(6, 5))
            plt.tight_layout()

            y_stft_ex = y_stft_val[0].cpu().detach().numpy()
            plt.subplot(1,4,1)
            plt.axis("off")
            plt.title("y (real)")
            plt.imshow(y_stft_ex[0].T)
            plt.subplot(1,4,2)
            plt.axis("off")
            plt.title("y (imag)")
            plt.imshow(y_stft_ex[1].T)

            yh_stft_ex = yh_stft_val[0].cpu().detach().numpy()
            plt.subplot(1,4,3)
            plt.axis("off")
            plt.title("ŷ (real)")
            plt.imshow(yh_stft_ex[0].T)
            plt.subplot(1,4,4)
            plt.axis("off")
            plt.title("ŷ (imag)")
            plt.imshow(yh_stft_ex[1].T)

            fig.canvas.draw()
            fft_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            fft_plot = fft_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            p_audio = dataset.istft(yh_stft_val[0])

            wandb.log( {
                "fft_frames_val": wandb.Image(fft_plot),
                "audio_input": wandb.Audio(audio_val[0], sample_rate=config.samplerate),
                "audio_output": wandb.Audio(p_audio, sample_rate=config.samplerate)
            } )

    utilities.save_model(f"saved_models/autoencoder_{wandb.run.name}", model)