import json
import urllib
import urllib.request
import glob
import matplotlib.pyplot as plt
import torch
import pickle
import subprocess
import os
import numpy as np
import torchvision.transforms.functional as TF

def get_all_files(base_dir, ext):
    return glob.glob(f'{base_dir}/*/**/**.{ext}', recursive=True)

def save_json(out_path, data, indent=3):
  with open(out_path, 'w') as outfile:
    json.dump(data, outfile, sort_keys=False, indent=indent)
  print(f'wrote json to {out_path}')

def load_json(path):
  with open(path) as json_file:
      jfile = json.load(json_file)
  return jfile

def get_video_title(vid_id):
    params = {"format": "json", "url": "https://www.youtube.com/watch?v=%s" % vid_id}
    query_string = urllib.parse.urlencode(params)
    url = "https://www.youtube.com/oembed"
    url = url + "?" + query_string

    with urllib.request.urlopen(url) as response:
        response_text = response.read()
        data = json.loads(response_text.decode())
        return data['title']

# save object as cached obj
def save_cache_obj(path, obj):
  filehandler = open(path, 'wb')
  pickle.dump(obj, filehandler)

def load_cache_obj(path):
  filehandler = open(path, 'rb') 
  obj = pickle.load(filehandler)
  return obj

def extract_audio_from_video(input_video, output_file, sr=16000):
  print(f'extracting audio from {input_video}')
  result = subprocess.Popen(["ffmpeg", "-i", input_video, "-vn", "-ar", str(sr), output_file], stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE,
                      universal_newlines=True)

  # result.wait()
  err = result.stderr
  # get the last output of error line
  err_result = list(err)[-1]    
  if "does not contain any stream" in err_result:
    return False
  else:
    return True

# get path to a video examples corresponding audio file
def get_paired_audio(video_path, extract=True):
  split_path = os.path.split(video_path)
  audio_path = os.path.join(split_path[0], "audio/", f"{split_path[1][:-4]}.wav")
  
  if os.path.isfile(audio_path):
    return audio_path

  elif extract:
    directory = os.path.split(video_path)[0]
    directory = f'{os.path.split(video_path)[0]}/audio/'
    if not os.path.exists(directory):
      print(f'making audio directory: {directory}')
      os.makedirs(directory)
    if extract_audio_from_video(video_path, audio_path):
      return audio_path
    else: # no valid audio stream found
      return None
  else:
    return None

def filter_valid_videos(all_vids, fps_lower_lim=29.97002997002996, fps_upper_lim=30., max_frames=None):
  import cv2
  print(f"filtering valid clips")
  valid_clips = []

  for i, v in enumerate(all_vids):
      video = cv2.VideoCapture(v)
      fps = video.get(cv2.CAP_PROP_FPS)
      video.release()
      # fps_info.append(np.array([fps, num_frames]))
      if fps >= fps_lower_lim and fps <=fps_upper_lim:

        if max_frames is not None:
          num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
          if num_frames <= max_frames:
            valid_clips.append(v)
        else:
          valid_clips.append(v)
      if i % 10 == 0:
        print(f"gathering info, file {i}/{len(all_vids)}")

  return valid_clips


def extract_clips(all_vids, frames_per_clip, frame_hop, framerate):

  def process_clips():
    from video_utils_custom import VideoClips
    print(f"processing video clips, this could take some time...")
    video_clips = VideoClips(
        all_vids,
        clip_length_in_frames=frames_per_clip,
        frames_between_clips=frame_hop,
        frame_rate=framerate,
        # num_workers=num_workers
    )
    config = [frames_per_clip, frame_hop, framerate]
    save_cache_obj("clipcache/video_clips.obj", video_clips)
    # save a config
    save_cache_obj("clipcache/clip_config.obj", config)
    return video_clips

  if not os.path.isfile("clipcache/video_clips.obj"):
    return process_clips()

  elif os.path.isfile("clipcache/clip_config.obj"):
    config = load_cache_obj("clipcache/clip_config.obj")
    if config != [frames_per_clip, frame_hop, framerate]:
      return process_clips()
    else:
      print("loading video clip slices from cache")
      return load_cache_obj("clipcache/video_clips.obj")

def save_model(path, model):
  while os.path.isfile(path):
    path = f'{path}_(1)'
  torch.save(model.state_dict(), path)

def video_phasegram(frames, resize=None, diff=True, cumulative=True):
  frames = torch.squeeze(frames, 1)
  if resize is not None:
    frames = TF.resize(frames, resize)
  fft = torch.fft.fftshift(torch.fft.fft2(frames))
  p = torch.angle(fft)
  p_flat = torch.flatten(p, start_dim=-2, end_dim=-1)
  if cumulative:
    p_flat = torch.cumsum(p_flat, dim=-1)
    p_flat /= 2. * np.pi * p_flat.shape[-1] # normalize
  else:
    p_flat += np.pi
    p_flat /= np.pi * 2.
  if diff:
    p_diff = torch.diff(p_flat, dim=-2)
    pad = torch.zeros_like(p_diff[:, 0:1, :])
    phasegram = torch.cat((pad, p_diff), dim=1)
  else:
    phasegram = p_flat
  phasegram = torch.unsqueeze(phasegram, 1)
  return phasegram

def generate_filmstrip(frames, dims):
  frames = frames.squeeze(0)
  filmstrip = torch.zeros((frames.shape[2], frames.shape[1] * frames.shape[0]))
  for i, f in enumerate(frames):
    filmstrip[:, i*frames.shape[1]:i*frames.shape[1]+frames.shape[2]] = f
  filmstrip = TF.resize(filmstrip.unsqueeze(0), dims, interpolation=TF.InterpolationMode.NEAREST)
  return filmstrip.squeeze(0)

# generate image of phasegram and frames
def video_phasegram_image(y_phasegram, yh_phasegram, frames, dims=(512, 2048)):

    pg_y_img = y_phasegram.permute(0, 2, 1)
    pg_y_img = TF.resize(pg_y_img, dims, interpolation=TF.InterpolationMode.NEAREST)
    pg_y_img = pg_y_img.cpu().detach().numpy()

    pg_yh_img = yh_phasegram.permute(0, 2, 1)
    pg_yh_img = TF.resize(pg_yh_img, dims, interpolation=TF.InterpolationMode.NEAREST)
    pg_yh_img = pg_yh_img.cpu().detach().numpy()
    filmstrip = generate_filmstrip(frames, dims)

    fig=plt.figure(figsize=(7, 3))
    plt.tight_layout()
    plt.subplots_adjust(wspace=1)

    plt.subplot(3,1,1)
    plt.title("video frames")
    plt.imshow(filmstrip)
    plt.axis("off")

    plt.subplots_adjust(wspace=1)
    plt.subplot(3,1,2)
    plt.title("phasegram (y)")
    plt.imshow(pg_y_img[0])
    plt.axis("off")

    plt.subplot(3,1,3)
    plt.title("phasegram (ŷ)")
    plt.imshow(pg_yh_img[0])
    plt.axis("off")

    fig.canvas.draw()
    frame_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    frame_plot = frame_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return frame_plot