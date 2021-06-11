# import avse_model
from avse_model import AVSE_Model
from train import TrainLoop
import torch
from generator import DataGenerator
import matplotlib.pyplot as plt
import numpy as np
import wandb

wandb.init(project='MagPhaseLVASE', entity='carl_m')
config = wandb.config

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

config.learning_rate = 1e-5
config.batch_size = 4
config.num_vid_frames = 4
config.epochs = 1000

config.center_fft = False
config.use_polar=False
config.normalize_fft=True

img_freq = 50

# loss_coefficient = 0.001 # loss = a_loss + loss_coefficient * v_loss

dg = DataGenerator(
    batch_size = config.batch_size,
    num_vid_frames=config.num_vid_frames, 
    framerate=30,
    framesize=256,
    samplerate=16000, 
    max_vid_frames=100,
    noise_std=0.01,
    center_fft=config.center_fft, 
    use_polar=config.use_polar, 
    shuffle_files=config.normalize_fft, 
    data_path = "data/processed"
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

    # loss = a_loss + loss_coefficient * v_loss
    loss = a_loss + v_loss

    loss.backward()

    print(f"step:{i} loss: {loss} a_loss:{a_loss} v_loss:{v_loss}")

    optimizer.step()

    wandb.log({ "loss": loss,
                "a_loss": a_loss,
                "v_loss": v_loss} )

    if i % img_freq == 0:
        
        fig=plt.figure(figsize=(8, 6))
        plt.tight_layout()

        cols = config.num_vid_frames
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
            "audio_input": wandb.Audio(audio[0], sample_rate=16000),
            "audio_output": wandb.Audio(p_audio[0], sample_rate=16000)
        } )
