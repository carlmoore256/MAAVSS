from re import sub
import torchvision
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import utilities
import glob
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    folder = "E:/MUSICES_ATTN/train/"
    output_dir = "D:/MagPhaseLAVSE/images/filmstrips"
    all_videos = utilities.get_all_files("E:/MUSICES", "mp4")

    all_video_names = [os.path.split(f)[-1][:-4] for f in all_videos]

    height = 256
    width = 256
    max_frames = 6

    all_files = glob.glob(os.path.join(folder, "*.jpg"))

    indexes = [int(f[-9:-4]) for f in all_files]
    clips_start = np.where(np.diff(indexes)>1)[0] + 1
    rand_clip_idx = np.random.randint(0,high=len(clips_start)-1)

    clips_list = all_files[clips_start[rand_clip_idx]:clips_start[rand_clip_idx+1]]
    clips_list = clips_list[:max_frames]

    subdir = os.path.split(os.path.split(clips_list[0])[0])[1]
    img_index = int(os.path.split(clips_list[0])[-1][-9:-4])
   

    matched_vid = None
    for name, path in zip(all_video_names, all_videos):
        if name == subdir:
            matched_vid = path

    image_tensor = torch.zeros((1, len(clips_list), 256, 256))
    video = torch.zeros((3, len(clips_list), 256, 256))

    video_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((height, width)),
        # transforms.RandomResizedCrop((height, width), scale=(0.6,1.0))
    ])

    last_pts = 0
    cap = cv2.VideoCapture(matched_vid)
    cap.set(1, img_index)
    success,frame = cap.read()
    count = 0
    while success:
        if count == max_frames:
            break
        success,frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = video_transform(frame)
        video[:, count, :, :] = frame.type(torch.float32)
        count += 1
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale()
    ])

    
    for i, f in enumerate(clips_list):
        img = Image.open(f)
        img = transform(img)
        image_tensor[:, i, :, :] = img

    # image_tensor *= 1/(torch.max(image_tensor) + 1e-7)

    filmstrip_attn = utilities.generate_filmstrip(image_tensor, dims=None, resize=False)
    filmstrip_video = utilities.generate_filmstrip(video, dims=None, resize=False)
    fs_attn_color = filmstrip_attn.unsqueeze(-1).repeat(1, 1, 3)

    
    filmstrip_combined = torch.cat((torch.clone(filmstrip_video), fs_attn_color), dim=0).permute(2,0,1)
    
    # torchvision.utils.save_image(filmstrip_combined, os.path.join(output_dir, f"{subdir}_{clips_start[rand_clip_idx]}_filmstrip_frames.jpg"))
    # torchvision.utils.save_image(filmstrip_attn, os.path.join(output_dir, f"{subdir}_{clips_start[rand_clip_idx]}_filmstrip.jpg"))
    # torchvision.utils.save_image(filmstrip_video.permute(2,0,1), os.path.join(output_dir, f"{subdir}_{clips_start[rand_clip_idx]}_video_raw.jpg"))
    print(image_tensor.shape)
    diff = torch.diff(image_tensor, dim=1)
    diff_filmstrip = utilities.generate_filmstrip(diff, dims=None, resize=False)

    # norm = (diff_filmstrip - torch.min(diff_filmstrip)) / (torch.max(diff_filmstrip) - torch.min(diff_filmstrip))
    norm = F.pad(diff_filmstrip, (width, 0))
    norm *= 1/(torch.max(torch.abs(diff_filmstrip)) + 1e-7)
    print(f'min  norm {torch.min(norm)} maxc {torch.max(norm)}')

    ch_r = torch.clone(norm)
    ch_r[ch_r < 0] = 0

    ch_g = torch.clone(norm)
    ch_g[ch_g >= 0] = 0
    ch_g *= -1
    
    rb_diffs = torch.zeros_like(fs_attn_color)

    # negative and positive values encoded in these channels
    rb_diffs[:, :, 0] = ch_r
    rb_diffs[:, :, 1] = ch_g

    movement_overlay = torch.clip(torch.clone(filmstrip_video) + torch.clone(rb_diffs), 0, 1).permute(2, 0, 1)

    torchvision.utils.save_image(movement_overlay, os.path.join(output_dir, f"{subdir}_{clips_start[rand_clip_idx]}_movement_overlay.jpg"))

    diffs_smooth = torch.clone(rb_diffs).permute(2, 0, 1)
    diffs_smooth = TF.gaussian_blur(diffs_smooth, kernel_size=(17, 17))
    filmstrip_video = filmstrip_video.permute(2, 0, 1)
    movement_overlay_smooth = torch.clip(filmstrip_video + diffs_smooth, 0, 1)

    torchvision.utils.save_image(movement_overlay_smooth, os.path.join(output_dir, f"{subdir}_{clips_start[rand_clip_idx]}_movement_overlay_smooth.jpg"))


    # diff_padded = F.pad(diff_filmstrip, (width, 0))
    # diff_padded = diff_padded.repeat(3, 1, 1).permute(1,2,0)
    # dp_thresh = diff_padded

    # dp_thresh *= 1/(torch.max(dp_thresh)+1e-7)
    # dp_thresh[dp_thresh > 0.4] = 1. 
    # masked = filmstrip_video + dp_thresh
    # masked = masked.permute(2, 0, 1)
    # masked = torch.clip(masked, 0, 1)

    # print(torch.max(masked))
    # torchvision.utils.save_image(masked, os.path.join(output_dir, f"{subdir}_{clips_start[rand_clip_idx]}_diff_mask.jpg"))




    # diff_rgb = filmstrip_attn.repeat(3, 1, 1) / 3.
    # diff_rgb[0, :, :] += diff_padded
    # diff_rgb *= 1/(torch.max(diff_rgb) + 1e-7)

    # filmstrip_combined2 = torch.cat((filmstrip_combined, diff_rgb), dim=1)
    # torchvision.utils.save_image(filmstrip_combined2, os.path.join(output_dir, f"{subdir}_{clips_start[rand_clip_idx]}_filmstrip_combined2.jpg"))
    
    # filmstrip_overlayed = filmstrip_video + diff_rgb.permute(1,2,0)
    # filmstrip_combined = torch.cat((fs_attn_color, filmstrip_overlayed), dim=0).permute(2,0,1)

    # torchvision.utils.save_image(filmstrip_combined, os.path.join(output_dir, f"{subdir}_{clips_start[rand_clip_idx]}_overlay.jpg"))




    # hop = 64
    # img_stack = torch.zeros((1, height+(hop * len(clips_list)), width+(hop * len(clips_list))))

    # print(f'image_tensor sh {image_tensor.shape}')
    # n_frames = image_tensor.shape[1]-1
    # for i in range(n_frames):
    #     hop_pos = ((n_frames-i)*hop) 
    #     img_stack[:, int(hop_pos * 0.5) : int(hop_pos * 0.5)+width, hop_pos:hop_pos+height] = image_tensor[:, i, :, :] / (n_frames - i)

    # torchvision.utils.save_image(img_stack, os.path.join(output_dir, f"{subdir}_{clips_start[rand_clip_idx]}_stack.jpg"))
