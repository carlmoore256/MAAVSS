from random import sample
import torch
from torch.utils import data
from av_dataset import AV_Dataset
from avse_model import AV_Model_STFT
import argparse
import matplotlib.pyplot as plt
import wandb

if __name__ == "__main__":
  wandb.init(project='MagPhaseLVASE', entity='carl_m', config={"dataset":"MUSIC"})

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
  parser.add_argument('--hop_ratio', type=int, default=8, help="divisions of fft_len for hop")
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

  dataset = AV_Dataset(
    frames_per_clip=config.num_frames,
    frame_hop=config.frame_hop,
    framerate=config.framerate,
    framesize=config.framesize,
    fft_len=config.fft_len,
    hop_ratio=config.hop_ratio,
    samplerate=config.samplerate,
    noise_std=config.noise_scalar,
    center_fft=config.center_fft,
    use_polar=config.use_polar,
    normalize_input_fft=config.normalize_fft,
    shuffle_files=True,
    num_workers=1,
    data_path=config.data_path
  )

  dataloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=config.batch_size,
                                          shuffle=True)
  dataloader = iter(dataloader)

  x_stft, y_stft, video, audio = next(dataloader)


  model = AV_Model_STFT(x_stft.shape, video.shape).to(DEVICE)

  mse_loss = torch.nn.MSELoss()
  # cosine_loss = torch.nn.CosineSimilarity()
  optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

  for i in range(config.epochs):
      optimizer.zero_grad()

      try:
        x_stft, y_stft, video, audio = next(dataloader)
      except Exception as e:
        print("ERROR LOADING EXAMPLE, SKIPPING...")
        continue

      yh_a, yh_v = model(x_stft.to(DEVICE), video.to(DEVICE))

      a_loss = mse_loss(yh_a, y_stft.to(DEVICE)).sum()
      v_loss = mse_loss(yh_v, video.to(DEVICE))

      loss = a_loss + v_loss

      loss.backward()

      print(f"step:{i} loss: {loss} a_loss:{a_loss} v_loss:{v_loss}")

      optimizer.step()

      wandb.log({ "loss": loss,
                  "a_loss": a_loss,
                  "v_loss": v_loss} )

      # if i % config.cb_freq == 0:
          
      #     fig=plt.figure(figsize=(8, 6))
      #     plt.tight_layout()

      #     cols = config.num_frames
      #     rows = 3
          
      #     for i in range(cols * rows):
      #         if i < cols:
      #             img = vid[0, i]
      #         elif i < cols * 2:
      #             img = x[1][0, 0, i%cols, :, :].cpu().detach().numpy()
      #         else:
      #             img = yh_v[0, 0, i%cols, :, :].cpu().detach().numpy()
      #         fig.add_subplot(rows, cols, i+1)
      #         plt.xticks([])
      #         plt.yticks([])
      #         plt.imshow(img)
      #     fig.canvas.draw()
      #     frame_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
      #     frame_plot = frame_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

      #     fig=plt.figure(figsize=(8, 7))

      #     fig.add_subplot(3, 1, 1)
      #     plt.title("x - noisy fft")
      #     plt.tight_layout()
      #     plt.plot(x[0][0, 0, :].cpu().detach().numpy())
      #     plt.plot(x[0][0, 1, :].cpu().detach().numpy())

      #     fig.add_subplot(3, 1, 2)
      #     plt.title("y - target fft")
      #     plt.plot(y[0][0, 0, :].cpu().detach().numpy())
      #     plt.plot(y[0][0, 1, :].cpu().detach().numpy())

      #     fig.add_subplot(3, 1, 3)
      #     plt.title("yhat - predicted fft")
      #     plt.plot(yh_a[0, 0, :].cpu().detach().numpy())
      #     plt.plot(yh_a[0, 1, :].cpu().detach().numpy())

      #     fig.canvas.draw()
      #     fft_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
      #     fft_plot = fft_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
          
      #     p_audio = dg.inference_to_audio(yh_a.cpu().detach())

      #     wandb.log( {
      #         "video_frames": wandb.Image(frame_plot),
      #         "fft_frames": wandb.Image(fft_plot),
      #         "audio_input": wandb.Audio(audio[0], sample_rate=config.samplerate),
      #         "audio_output": wandb.Audio(p_audio[0], sample_rate=config.samplerate)
      #     } )
