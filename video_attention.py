# use dino to extract video features
# https://github.com/facebookresearch/dino/blob/main/video_generation.py


import enum
import os


import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np

import dino.utils
import dino.vision_transformer as vits


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VideoAttention:
    def __init__(self,
                 patch_size=8, # Patch resolution of the self.model
                 threshold = 0.6,# masks obtained by thresh self-attention maps to keep xx percent of the mass
                 path_to_weights="dino_deitsmall8_pretrain.pth", # path to the pretrained weights avail at https://github.com/facebookresearch/dino
                 architecture="vit_small", # ["vit_tiny", "vit_small", "vit_base"]
                 resize=None):
        self.resize = resize
        self.threshold = threshold
        self.patch_size = patch_size
        self.checkpoint_key = "teacher" # Key to use in the checkpoint (example: "teacher")
        self.model = self.__load_model(architecture, path_to_weights)
        

    def _inference(self, frames):

        # attn_frames = []
        attn_frames = torch.zeros((frames.shape[0], 1, frames.shape[2], frames.shape[3]))

        for i, img in enumerate(frames):
            # make the image divisible by the patch size
            w, h = (
                img.shape[1] - img.shape[1] % self.patch_size,
                img.shape[2] - img.shape[2] % self.patch_size,
            )
            img = img[:, :w, :h].unsqueeze(0)

            w_featmap = img.shape[-2] // self.patch_size
            h_featmap = img.shape[-1] // self.patch_size
            
            attentions = self.model.get_last_selfattention(img.to(DEVICE))

            nh = attentions.shape[1]  # number of head

            # we keep only the output patch attention
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)

            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - self.threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
                
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = (
                nn.functional.interpolate(
                    th_attn.unsqueeze(0),
                    scale_factor=self.patch_size,
                    mode="nearest",
                )[0]
                # .cpu()
                # .numpy()
            )

            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = (
                nn.functional.interpolate(
                    attentions.unsqueeze(0),
                    scale_factor=self.patch_size,
                    mode="nearest",
                )[0]
                # .cpu()
                # .numpy()
            )
            # attentions = [attentions[i] * 1 / attentions.shape[0] for i in range(attentions.shape[0])]
            attentions *= 1/attentions.shape[0] # divide by total number of frames
            output_frame = torch.sum(attentions, dim=0) # sum attentions
            output_frame *= 1/torch.max(output_frame)

            # attn_frames.append(output_frame)
            attn_frames[i, 0, :, :] = output_frame
            # .to("cpu")
            # attn_frames[idx, :, :] = torch.as_tensor(output_frame).to(DEVICE)
            # attn_frames.append(output_frame)

        # attn_frames = np.asarray(attn_frames)
        # attn_frames = torch.as_tensor(attn_frames)
        return attn_frames


    def __load_model(self, arch, pretrained_weights):
        # build model
        model = vits.__dict__[arch](
            patch_size=self.patch_size, num_classes=0
        )
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(DEVICE)

        if os.path.isfile(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if (
                self.checkpoint_key is not None
                and self.checkpoint_key in state_dict
            ):
                print(
                    f"Take key {self.checkpoint_key} in provided checkpoint dict"
                )
                state_dict = state_dict[self.checkpoint_key]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print(
                "Pretrained weights found at {} and loaded with msg: {}".format(
                    pretrained_weights, msg
                )
            )
        else:
            print(
                "Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate."
            )
            url = None
            if arch == "vit_small" and self.patch_size == 16:
                url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif arch == "vit_small" and self.patch_size == 8:
                url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
            elif arch == "vit_base" and self.patch_size == 16:
                url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif arch == "vit_base" and self.patch_size == 8:
                url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            if url is not None:
                print(
                    "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
                )
                state_dict = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/dino/" + url
                )
                model.load_state_dict(state_dict, strict=True)
            else:
                print(
                    "There is no reference weights available for this model => We use random weights."
                )
        return model