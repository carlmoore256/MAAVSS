import json
import urllib
import urllib.request
import glob
import matplotlib.pyplot as plt
import torch
import pickle
import os

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

def filter_valid_fps(all_vids, lower_lim=29.97002997002996, upper_lim=30.):
  import cv2
  print(f"filtering valid clips")
  valid_clips = []

  for v in all_vids:
      video = cv2.VideoCapture(v)
      fps = video.get(cv2.CAP_PROP_FPS)
      # num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
      video.release()
      # fps_info.append(np.array([fps, num_frames]))
      if fps >= lower_lim and fps <=upper_lim:
          valid_clips.append(v)

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