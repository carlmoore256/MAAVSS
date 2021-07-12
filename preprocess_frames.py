import torchvision
from video_attention import VideoAttention
import utilities
import numpy as np

if __name__ == "__main__":
    frame_buffer = 8
    frame_hop = 2
    output_dims = (256, 256)

    memmap_path = "E:/MUSICES/all_attention_frames.memmap"

    all_vids = utilities.get_all_files("E:/MUSICES", "mp4")
    video_clips = utilities.extract_clips(all_vids,
                                        frame_buffer,
                                        frame_hop,
                                        None)
    
    hops_per_buff = frame_buffer//frame_hop
    total_clips = video_clips.num_clips()//hops_per_buff
    total_frames = total_clips * frame_buffer
    total_pixels = total_frames * output_dims[0] * output_dims[1]
    print(f'total clips: {total_clips} total frames {total_frames} total pixels {total_pixels}')
    # 3991352
    # 100000
    map = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(2000000, output_dims[0], output_dims[1]))