from video_attention import VideoAttention



# set attention extractor parameters
self.attention_extractor = VideoAttention(
    patch_size=8,
    threshold=0.6
)