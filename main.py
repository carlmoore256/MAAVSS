# import avse_model
from avse_model import AVSE_Model
from generator import DataGenerator

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import wandb

wandb.init(project='MagPhaseLVASE', entity='carl_m', config={"dataset":"MUSIC"})

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=4, metavar='N')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
parser.add_argument("-e", '--epochs', type=int, default=1000, help="epochs")
parser.add_argument('--data_path', type=str, default="data/processed", help="path to dataset")
parser.add_argument('--num_frames', type=int, default=4, help="number of consecutive video frames (converted to attention maps)")
parser.add_argument('--framesize', type=int, default=256, help="scaled video frame dims (converted to attention maps)")
parser.add_argument('--samplerate', type=int, default=16000, help="audio samplerate (dependent on dataset)")
parser.add_argument('--center_fft', type=bool, default=True, help="interlace and center fft")
parser.add_argument('--use_polar', type=bool, default=False, help="fft uses polar coordinates instead of rectangular")
parser.add_argument('--normalize_fft', type=bool, default=True, help="normalize input fft by 1/n")
parser.add_argument('--noise_scalar', type=float, default=0.1, help="scale gaussian noise by N for data augmentation (applied to x)")
parser.add_argument('--cb_freq', type=int, default=100, help="wandb callback frequency in epochs")

args = parser.parse_args()

config = wandb.config
wandb.config.update(args)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dg = DataGenerator(
    batch_size=config.batch_size,
    num_vid_frames=config.num_frames, 
    framerate=30,
    framesize=config.framesize,
    samplerate=config.samplerate, 
    max_vid_frames=100, # accelerate video loading time
    noise_std=config.noise_scalar,
    center_fft=config.center_fft, 
    use_polar=config.use_polar, 
    shuffle_files=config.normalize_fft, 
    data_path = config.data_path
)

gen = dg.generator()
x_example, y_example, _, _ = next(gen)
a_shape = x_example[0].shape
v_shape = x_example[1].shape

model = AVSE_Model(a_shape, v_shape).to(DEVICE)
print(model)

wandb.watch(model)

mse_loss = torch.nn.MSELoss()
cosine_loss = torch.nn.CosineSimilarity()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) <- caused memory issues
optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

for i in range(config.epochs):
    optimizer.zero_grad()

    x, y, vid, audio = next(gen)

    yh_a, yh_v = model(x[0].to(DEVICE), x[1].to(DEVICE))

    a_loss = mse_loss(yh_a, y[0].to(DEVICE)).sum()
    v_loss = mse_loss(yh_v, y[1].to(DEVICE))

    loss = a_loss + v_loss

    loss.backward()

    print(f"step:{i} loss: {loss} a_loss:{a_loss} v_loss:{v_loss}")

    optimizer.step()

    wandb.log({ "loss": loss,
                "a_loss": a_loss,
                "v_loss": v_loss} )

    if i % config.cb_freq == 0:
        
        fig=plt.figure(figsize=(8, 6))
        plt.tight_layout()

        cols = config.num_frames
        rows = 3
        
        for i in range(cols * rows):
            if i < cols:
                img = vid[0, i]
            elif i < cols * 2:
                img = x[1][0, 0, i%cols, :, :].cpu().detach().numpy()
            else:
                img = yh_v[0, 0, i%cols, :, :].cpu().detach().numpy()
            fig.add_subplot(rows, cols, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
        fig.canvas.draw()
        frame_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        frame_plot = frame_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        fig=plt.figure(figsize=(8, 7))

        fig.add_subplot(3, 1, 1)
        plt.title("x - noisy fft")
        plt.tight_layout()
        plt.plot(x[0][0, 0, :].cpu().detach().numpy())
        plt.plot(x[0][0, 1, :].cpu().detach().numpy())

        fig.add_subplot(3, 1, 2)
        plt.title("y - target fft")
        plt.plot(y[0][0, 0, :].cpu().detach().numpy())
        plt.plot(y[0][0, 1, :].cpu().detach().numpy())

        fig.add_subplot(3, 1, 3)
        plt.title("yhat - predicted fft")
        plt.plot(yh_a[0, 0, :].cpu().detach().numpy())
        plt.plot(yh_a[0, 1, :].cpu().detach().numpy())

        fig.canvas.draw()
        fft_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        fft_plot = fft_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        p_audio = dg.inference_to_audio(yh_a.cpu().detach())

        wandb.log( {
            "video_frames": wandb.Image(frame_plot),
            "fft_frames": wandb.Image(fft_plot),
            "audio_input": wandb.Audio(audio[0], sample_rate=config.samplerate),
            "audio_output": wandb.Audio(p_audio[0], sample_rate=config.samplerate)
        } )
