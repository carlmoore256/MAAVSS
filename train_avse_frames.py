from random import sample
import torch
from torch.utils import data
from av_dataset import AV_Dataset
from avse_model_final import AV_Fusion_Model_Frames
import matplotlib.pyplot as plt
import numpy as np
import wandb
import utilities
from run_config import model_args
import torchvision.transforms.functional as TF


def train():
    args = model_args()
    with wandb.init(project='AV-Fusion-AVSE', entity='carl_m', config=args):
      config = wandb.config

      wandb.config.update({"phasegram" : False})

      DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

      config.hop, audio_sample_len, config.num_fft_frames = utilities.calc_hop_size(
          config.num_frames,
          config.hops_per_frame, 
          config.framerate, 
          config.samplerate
      )
      
      dataset = AV_Dataset(
          num_frames= config.num_seq + config.num_frames,
          frame_hop=config.frame_hop,
          framerate=config.framerate,
          samplerate = config.samplerate,
          fft_len=config.fft_len,
          hops_per_frame=config.hops_per_frame,
          noise_std=config.noise_scalar,
          use_polar = config.use_polar,
          attn_diff=config.attn_diff,
          normalize_input_fft = config.normalize_fft,
          normalize_output_fft = config.normalize_output_fft,
          autocontrast=config.autocontrast,
          compress_audio=config.compress_audio,
          shuffle_files=True,
          data_path=config.data_path,
          max_clip_len=config.max_clip_len,
          gen_stft=True,
          gen_video=True,
          trim_stft_end=False,
          attn_frames_path="E:/MUSICES_ATTN/train"
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

      stft_shape = [config.batch_size, 2, config.num_fft_frames, config.fft_len//2 + 1]     
      attn_shape = [config.batch_size, 1, config.num_frames, config.framesize, config.framesize]

      x_stft_ex, _, attn_ex, _, _ = next(iter(train_gen))
      
      stft_zeros = torch.zeros_like(x_stft_ex).to(DEVICE)
      attn_zeros = torch.zeros_like(attn_ex).to(DEVICE)

      model = AV_Fusion_Model_Frames(stft_shape, 
                          attn_shape,
                          config.hops_per_frame,
                          latent_channels=config.latent_chan,
                          fc_size=config.fc_size
                          ).to(DEVICE)
      # model_artifact = wandb.Artifact(
      #   "AV-Fusion-Convnet",
      #   type='model', 
      #   metadata=dict(config))
      # save_name = f"saved_models/{wandb.run.name}.pt"
      # utilities.save_model(save_name, model, overwrite=True)
      # model_artifact.add_file(save_name)
      # wandb.save(save_name)
      # run.log_artifact(model_artifact)
      
      mse_loss = torch.nn.MSELoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

      if config.saved_model != None:
          print(f'Loading model weights from {config.saved_model}')
          model.load_state_dict(torch.load(config.saved_model), strict=False)
      if args.c or args.checkpoint is not None:
          utilities.load_checkpoint(model, optimizer, args.cp_dir, args.c, args.checkpoint, config.cp_load_opt)
          
      t_gen = iter(train_gen)
      v_gen = iter(val_gen)
      last_loss = 1e5

      train_mode = 2 # 0 = audio, 1 = visual, 2 = av
      idx_middle_frame = (config.num_seq - 1) // 2

      try:
        wandb.watch(model, torch.nn.functional.mse_loss, log="all", log_freq=300)
      except Exception as e:
        print(f'error with wandb.watch: {e}')

      for e in range(config.epochs):
          if e + 1 * config.steps_per_epoch > len(train_gen):
              t_gen = iter(train_gen)
          if e + 1 * config.val_steps > len(val_gen):
              v_gen = iter(val_gen)

          model.train()

          for i in range(config.steps_per_epoch):

              x_stft, y_stft, attn, audio, video = next(t_gen)
              
              y_attn = attn

              if train_mode == 0:
                x_attn = attn_zeros
                if config.objective_zeros:
                    y_attn = attn_zeros
              else:
                x_attn = attn            

              if train_mode == 1:
                x_stft = stft_zeros
                if config.objective_zeros:
                    y_stft = stft_zeros

              x_stft = x_stft.to(DEVICE)
              y_stft = y_stft.to(DEVICE)
              x_attn = x_attn.to(DEVICE)
              y_attn = y_attn.to(DEVICE)

              if i % config.cb_freq == 0:
                attn_orig = attn
                stft_start = idx_middle_frame * config.hops_per_frame
                stft_end = stft_start + (config.hops_per_frame * config.num_seq)
                output_stft = torch.zeros_like(x_stft[:, :, stft_start:stft_end, :])
                output_attn = torch.zeros_like(attn[:, :, idx_middle_frame:idx_middle_frame+config.num_seq, :])

              for j in range(config.num_seq):

                x_attn_batch = x_attn[:, :, j:j+config.num_frames, :]
                y_attn_batch = y_attn[:, :, j+idx_middle_frame, :]            

                stft_pos_start = config.hops_per_frame*j
                stft_pos_end = stft_pos_start+(config.hops_per_frame * config.num_frames)

                stft_pos_start_mid = stft_pos_start + (config.hops_per_frame * idx_middle_frame)
                stft_pos_end_mid = stft_pos_start_mid + config.hops_per_frame

                x_stft_batch = x_stft[:, :, stft_pos_start:stft_pos_end, :]
                y_stft_batch = y_stft[:, :, stft_pos_start_mid:stft_pos_end_mid, :]

                yh_stft, yh_attn, latent = model(x_stft_batch, x_attn_batch)

                a_loss = mse_loss(yh_stft, y_stft_batch)
                v_loss = mse_loss(yh_attn, y_attn_batch)
                loss = a_loss + config.loss_coeff * v_loss

                loss /= config.num_seq
                loss.backward()

                if i % config.cb_freq == 0:
                    stft_start_op = j*config.hops_per_frame
                    stft_end_op = stft_start_op + config.hops_per_frame
                    output_stft[:, :, stft_start_op:stft_end_op, :] = yh_stft
                    output_attn[:, :, j, :, :] = yh_attn
                  

              optimizer.step()
              optimizer.zero_grad()

              wandb.log({
                  "loss" : loss,
                  "a_loss" : a_loss,
                  "v_loss" : v_loss,
                  "mode" : train_mode,
                  "cache_ratio" : dataset.get_cache_ratio()
              })

              if i % config.cb_freq == 0:
                  print(f'epoch {e} step {i}/{config.steps_per_epoch} loss {loss} a_loss {a_loss} v_loss {v_loss}')
                  
                  # output_stft = output_stft[:, :, idx_middle_frame*config.hops_per_frame:(config.num_seq * config.hops_per_frame)]
                  y_stft = y_stft[:, :, stft_start:stft_end, :]
                  attn =  attn[:, :, idx_middle_frame:idx_middle_frame+config.num_seq, :]
                  video =  video[:, :, idx_middle_frame:idx_middle_frame+config.num_seq, :]
                  
                  audio_input = dataset.istft(y_stft[0].cpu().detach()).unsqueeze(0)
                  audio_output = dataset.istft(output_stft[0].cpu().detach()).unsqueeze(0)

                  input_mag, input_phase = utilities.plot_waveform_specgram(audio_input, 16000)
                  output_mag, output_phase = utilities.plot_waveform_specgram(audio_output, 16000)
                  
                    
                  wandb.log({
                      "stft" : utilities.stft_ae_image_callback(y_stft[0], output_stft[0]),
                      "input_mag" : input_mag,
                      "input_phase" : input_phase,
                      "output_mag" : output_mag,
                      "output_phase" : output_phase,
                      "attention_frames" : utilities.video_frames_image(attn[0], output_attn[0], video[0]),
                      "audio_input" : wandb.Audio(audio_input.squeeze(0), sample_rate=16000),
                      "audio_output" : wandb.Audio(audio_output.squeeze(0), sample_rate=16000),
                  })
          

          # multi-input training scheme - switch between training modes
          if e % config.mode_freq == 0:
            train_mode = np.random.randint(0,2)
          
          utilities.save_checkpoint(model.state_dict(), 
                                  optimizer.state_dict(),
                                  e, loss,
                                  wandb.run.name,
                                  config.cp_dir)
      if not args.no_save:
          utilities.save_model(f"saved_models/avf-v-ae-{wandb.run.name}.pt", model, overwrite=True)

if __name__ == "__main__":
    train()
