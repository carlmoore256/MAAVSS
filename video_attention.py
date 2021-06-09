# use dino to extract video features
# https://github.com/facebookresearch/dino/blob/main/video_generation.py


import enum
import os


import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
# import numpy as np

import dino.utils
import dino.vision_transformer as vits


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VideoAttention:
    def __init__(self, resize=None):
        self.resize = resize
        self.model = self.__load_model()

    def _inference(self, frames):

        attn_frames = []

        for i, img in enumerate(frames):

            if self.resize is not None:
                transform = pth_transforms.Compose(
                    [
                        pth_transforms.ToTensor(),
                        pth_transforms.Resize(self.args.resize),
                        pth_transforms.Normalize(
                            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                        ),
                    ]
                )
            else:
                transform = pth_transforms.Compose(
                    [
                        pth_transforms.ToTensor(),
                        pth_transforms.Normalize(
                            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                        ),
                    ]
                )

            img = transform(img)

            # make the image divisible by the patch size
            w, h = (
                img.shape[1] - img.shape[1] % self.args.patch_size,
                img.shape[2] - img.shape[2] % self.args.patch_size,
            )
            img = img[:, :w, :h].unsqueeze(0)

            w_featmap = img.shape[-2] // self.args.patch_size
            h_featmap = img.shape[-1] // self.args.patch_size

            attentions = self.model.get_last_selfattention(img.to(DEVICE))

            nh = attentions.shape[1]  # number of head

            # we keep only the output patch attention
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - self.args.threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = (
                nn.functional.interpolate(
                    th_attn.unsqueeze(0),
                    scale_factor=self.args.patch_size,
                    mode="nearest",
                )[0]
                .cpu()
                .numpy()
            )

            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = (
                nn.functional.interpolate(
                    attentions.unsqueeze(0),
                    scale_factor=self.args.patch_size,
                    mode="nearest",
                )[0]
                .cpu()
                .numpy()
            )

            output_frame = sum(attentions[i] * 1 / attentions.shape[0] for i in range(attentions.shape[0]))

            attn_frames.append(output_frame)

            print(f"processed attention frame {i}/{len(frames)}")

        return attn_frames


    def __load_model(self):
        # build model
        model = vits.__dict__[self.args.arch](
            patch_size=self.args.patch_size, num_classes=0
        )
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(DEVICE)

        if os.path.isfile(self.args.pretrained_weights):
            state_dict = torch.load(self.args.pretrained_weights, map_location="cpu")
            if (
                self.args.checkpoint_key is not None
                and self.args.checkpoint_key in state_dict
            ):
                print(
                    f"Take key {self.args.checkpoint_key} in provided checkpoint dict"
                )
                state_dict = state_dict[self.args.checkpoint_key]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print(
                "Pretrained weights found at {} and loaded with msg: {}".format(
                    self.args.pretrained_weights, msg
                )
            )
        else:
            print(
                "Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate."
            )
            url = None
            if self.args.arch == "vit_small" and self.args.patch_size == 16:
                url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif self.args.arch == "vit_small" and self.args.patch_size == 8:
                url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
            elif self.args.arch == "vit_base" and self.args.patch_size == 16:
                url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif self.args.arch == "vit_base" and self.args.patch_size == 8:
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


# if __name__ == "__main__":
#     args = parse_args()

#     vg = VideoGenerator(args)
#     vg.run()