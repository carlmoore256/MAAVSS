import utilities
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def load_audio_map(base_path):
    mmap_path = os.path.join(base_path, "audio_memmap.memmap")
    index_map_path = os.path.join(base_path, "audio_index_map.obj")
    index_map = utilities.load_cache_obj(index_map_path)
    map_len = index_map[1][-1, 1]
    map = np.memmap(mmap_path, dtype='float32', mode='r', shape=(map_len))
    return map, index_map
map, index_map = load_audio_map("E:/MUSICES")

index = np.random.randint(0,len(index_map[1]-1))
loc = index_map[1][index]
plt.plot(map[loc[0]:loc[1]])
plt.show()