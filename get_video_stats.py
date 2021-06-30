import cv2
import glob
import numpy as np
import pickle

all_vids = glob.glob("E:/MUSICES/*/**.mp4", recursive=True)

fps_info = []

valid_clips = []

for i, v in enumerate(all_vids):
    video = cv2.VideoCapture(v)
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    video.release()
    fps_info.append(np.array([fps, num_frames]))
    if fps >= 29.97002997002996 and fps <=30:
        valid_clips.append(v)

    if i % 10 == 0:
        print(f"gathering info, file {i}/{len(all_vids)}")

fps_info = np.asarray(fps_info)

np.save("fps_info.npy", fps_info)
print(f'saving {len(valid_clips)} valid clips')
filehandler = open("clipcache/valid_clips.obj", 'wb')
pickle.dump(valid_clips, filehandler)