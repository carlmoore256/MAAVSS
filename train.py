from random import sample
import torch
from torch.utils import data
from av_dataset import AV_Dataset
from avse_model import AV_Model_STFT
import argparse
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torchaudio

if __name__ == "__main__":
  wandb.init(project='MagPhaseLVASE', entity='carl_m', config={"dataset":"MUSIC"})

  # IF CHANING NUM_FRAMES, delete cached video frames

  parser = argparse.ArgumentParser()
  parser.add_argument('-b', '--batch_size', type=int, default=4, metavar='N')
  parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
  parser.add_argument("-e", '--epochs', type=int, default=1000, help="epochs")
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
  parser.add_argument('--noise_scalar', type=float, default=0.1, help="scale gaussian noise by N for data augmentation (applied to x)")
  parser.add_argument('--cb_freq', type=int, default=100, help="wandb callback frequency in epochs")
  parser.add_argument('--max_clip_len', type=int, default=None, help="maximum clip length to load (speed up loading)")

  args = parser.parse_args()
  config = wandb.config
  wandb.config.update(args)

  DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  config.hop = int((config.samplerate/config.framerate)/config.hops_per_frame)
  # therefore, fft size should = (..., 2, fft_len/2+1, num_frames * a)
  # make sure num_frames > hop
  
  dataset = AV_Dataset(
    frames_per_clip=config.num_frames,
    frame_hop=config.frame_hop,
    framerate=config.framerate,
    framesize=config.framesize,
    fft_len=config.fft_len,
    hop=config.hop,
    hops_per_frame=config.hops_per_frame,
    samplerate=config.samplerate,
    noise_std=config.noise_scalar,
    center_fft=config.center_fft,
    use_polar=config.use_polar,
    normalize_input_fft=config.normalize_fft,
    shuffle_files=True,
    num_workers=1,
    data_path=config.data_path,
    max_clip_len=config.max_clip_len
  )

  dataloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=config.batch_size,
                                          shuffle=True)
  dataloader = iter(dataloader)

  x_stft, y_stft, attn, audio, video = next(dataloader)


  model = AV_Model_STFT(x_stft.shape, attn.shape, config.hops_per_frame).to(DEVICE)

  print(model)

  mse_loss = torch.nn.MSELoss()
  # cosine_loss = torch.nn.CosineSimilarity()
  optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

  training_ae = True

  for i in range(config.epochs):
      optimizer.zero_grad()

      try:
        x_stft, y_stft, attn, audio, video = next(dataloader)
      except Exception as e:
        print("ERROR LOADING EXAMPLE, SKIPPING...")
        continue

      if i < 300: # train the ae
        yh_stft, yh_attn = model(y_stft.to(DEVICE), attn.to(DEVICE), train_ae=True)
      else:
        yh_stft, yh_attn = model(x_stft.to(DEVICE), attn.to(DEVICE), train_ae=False)

      a_loss = mse_loss(yh_stft, y_stft.to(DEVICE)).sum()
      v_loss = mse_loss(yh_attn, attn.to(DEVICE))

      loss = a_loss + v_loss

      loss.backward()

      print(f"step:{i} loss: {loss} a_loss:{a_loss} v_loss:{v_loss}")

      optimizer.step()

      wandb.log({ "loss": loss,
                  "a_loss": a_loss,
                  "v_loss": v_loss} )

      if i % config.cb_freq == 0:
        fig=plt.figure(figsize=(8, 5))
        plt.tight_layout()

        cols = config.num_frames
        rows = 3

        for i in range(cols * rows):
          if i < cols:
              img = video[0, 0, i, :, :].cpu().detach().numpy()
          elif i < cols * 2:
              img = attn[0, 0, i%cols, :, :].cpu().detach().numpy()
          else:
              img = yh_attn[0, 0, i%cols, :, :].cpu().detach().numpy()
          fig.add_subplot(rows, cols, i+1)
          plt.xticks([])
          plt.yticks([])
          plt.imshow(img)
        fig.canvas.draw()
        frame_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        frame_plot = frame_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        fig=plt.figure(figsize=(8, 3))
        plt.tight_layout()

        x_stft_ex = x_stft[0].cpu().detach().numpy()
        plt.subplot(1,6,1)
        plt.axis("off")
        plt.title("x (real)")
        plt.imshow(x_stft_ex[0])

        plt.subplot(1,6,2)
        plt.axis("off")
        plt.title("x (imag)")
        plt.imshow(x_stft_ex[1])

        y_stft_ex = y_stft[0].cpu().detach().numpy()
        plt.subplot(1,6,3)
        plt.axis("off")
        plt.title("y (real)")
        plt.imshow(y_stft_ex[0])
        plt.subplot(1,6,4)
        plt.axis("off")
        plt.title("y (imag)")
        plt.imshow(y_stft_ex[1])

        yh_stft_ex = yh_stft[0].cpu().detach().numpy()
        plt.subplot(1,6,5)
        plt.axis("off")
        plt.title("yh (real)")
        plt.imshow(yh_stft_ex[0])
        plt.subplot(1,6,6)
        plt.axis("off")
        plt.title("yh (imag)")
        plt.imshow(yh_stft_ex[1])

        fig.canvas.draw()
        fft_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        fft_plot = fft_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # window = torch.hamming_window(config.fft_len)
        # p_audio = torch.istft(yh_stft[0].cpu().detach().permute(1,2,0), 
        #                       n_fft=config.fft_len, 
        #                       hop_length=config.hop, 
        #                       win_length=config.fft_len,
        #                       window=window,
        #                       normalized=config.normalize_fft,
        #                       onesided=True)
        p_audio = dataset.istft(yh_stft[0])

        wandb.log( {
            "video_frames": wandb.Image(frame_plot),
            "fft_frames": wandb.Image(fft_plot),
            "audio_input": wandb.Audio(audio[0], sample_rate=config.samplerate),
            "audio_output": wandb.Audio(p_audio, sample_rate=config.samplerate)
        } )

torch.save(model.state_dict(), f"saved_models/model_e{config.epochs}_l{loss}")
