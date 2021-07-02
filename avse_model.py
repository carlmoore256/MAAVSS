# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv1D, SeparableConv1D, Conv3D, Reshape, Conv2D
# from tensorflow.keras.layers import concatenate, Dense, AveragePooling1D, AveragePooling3D, UpSampling3D
from math import e
from torch.functional import stft
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules import module
from torchsummary import summary
import copy

class AVSE_Model(nn.Module):
    def __init__(self, a_shape, v_shape):
        super(AVSE_Model, self).__init__()

        self.a_shape = a_shape
        self.v_shape = v_shape

        print(f"\nA SHAPE {a_shape}")
        print(f"V SHAPE {v_shape}\n")

        self.audio_net = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=10, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.AvgPool1d(2),
            nn.Conv1d(10, 4, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(4, 2, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(2),
            # nn.Flatten()
        )

        self.visual_net = nn.Sequential(
            nn.Conv3d(1, 12, kernel_size=3, stride=(1,2,2), padding_mode="zeros"),
            nn.ReLU(),
            nn.Conv3d(12, 6, kernel_size=(1, 3, 3), stride=(1,2,2), padding_mode="zeros"), # <- this is an issue, is the kernel doing anything in time?
            nn.ReLU(),
            nn.Conv3d(6, 2, kernel_size=(1, 3, 3), stride=(2,2,2), padding_mode="zeros"),
            nn.ReLU()
            # nn.Flatten()
        )

        self.av_fc1 = nn.Linear(2182, 512, bias=False)
        self.av_fc1_ln = nn.LayerNorm(512, elementwise_affine=True)

        # self.av_fc2 = nn.Linear(512, )

        self.a_fc_out = nn.Linear(512, a_shape[1] * a_shape[2])
        self.v_fc_out = nn.Linear(512, v_shape[1] * v_shape[2] * v_shape[3] * v_shape[4])

    def forward(self, x_a, x_v):

        y_a = self.audio_net(x_a)
        y_v = self.visual_net(x_v)

        # y_v = y_v.squeeze(2) # -> squeeze time dimension?

        # y_a = y_a.unsqueeze(-1) # -> add another dimension?

        # y_v = y_v.reshape((y_v.shape[0], y_v.shape[1], y_v.shape[2], y_v.shape[3] * y_v.shape[4]))
        y_v = y_v.flatten(start_dim=-2, end_dim=-1) # <- cool way to reshape
        y_v = y_v.squeeze(2)

        av_fc = torch.cat((y_a, y_v), -1)

        av_fc = av_fc.flatten(start_dim=1)
        
        av_fc = self.av_fc1(av_fc)
        av_fc = self.av_fc1_ln(av_fc)
        av_fc = F.leaky_relu(av_fc, negative_slope=0.3) # <- use functional here for simplicity

        y_a = self.a_fc_out(av_fc)
        y_a = torch.tanh(y_a)
        y_a = y_a.reshape(self.a_shape)

        # reconstruction of video attention frames
        y_v = self.v_fc_out(av_fc)
        y_v = F.leaky_relu(y_v, negative_slope=0.3) 
        # reshape tensor into original dimensions
        y_v = y_v.reshape(self.v_shape)

        return y_a, y_v

class AV_Model_STFT(nn.Module):
    def __init__(self, stft_shape, v_shape, alpha):
        super(AV_Model_STFT, self).__init__()

        self.stft_shape = stft_shape
        self.v_shape = v_shape

        time_dim = stft_shape[2]
        n_div = 0

        while time_dim > v_shape[2]:
            # print(st_sh)
            time_dim /= 2
            n_div += 1
        print(f'\n NDIV {n_div}')
        modules = []
        in_ch = 2
        for i in range(alpha):
            # out_ch = (alpha - (i + 1)) * 2 + 2
            out_ch = in_ch * 2
            modules.append(nn.ZeroPad2d((1, 1, 2, 0)))
            if i < n_div:
                modules.append(nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(2, 2)))
            else:
                modules.append(nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 2)))

            modules.append(nn.ReLU())
            # modules.append(nn.MaxPool2d((2,1)))
            in_ch = out_ch
        self.audio_net = nn.Sequential(*modules)
        # print(self.audio_net)
        summary(self.audio_net.to("cuda"), input_size=(stft_shape[1], stft_shape[2], stft_shape[3]))

        x_a = torch.rand(stft_shape).to("cuda")
        with torch.no_grad():
            x_a_enc = self.audio_net(x_a)

        modules = []
        spatial_dim = v_shape[3]
        in_ch = 1

        # while time_dim > audio_output.shape[-1]:
        while spatial_dim > x_a_enc.shape[-1]//2:
            out_ch = in_ch * 2
            modules.append(nn.Conv3d(in_ch, out_ch, kernel_size=(1,3,3), stride=(1,2,2), padding=(0, 1, 1), padding_mode="zeros"))
            spatial_dim /= 2
            in_ch = out_ch

        self.visual_net = nn.Sequential(*modules)
        summary(self.visual_net.to("cuda"), input_size=(v_shape[1], v_shape[2], v_shape[3], v_shape[4]))
        x_v = torch.rand(v_shape).to("cuda")

        with torch.no_grad():
            x_v_enc = self.visual_net(x_v)
            # concatenate along channel axis

        # flatten and pool to match size of encoded audio
        x_v_flat = torch.flatten(x_v_enc, start_dim=-2, end_dim=-1)
        print(f'flat - audio sh {x_v_flat.shape} {x_a_enc.shape}')
        flattened_dim = x_v_flat.shape[-1]

        # at bottom of both a and v net, find difference btwn latent sizes
        enc_spatial_diff = x_v_flat.shape[-1] - x_a_enc.shape[-1]
        
        if x_v_flat.shape[-1] > x_a_enc.shape[-1]:
            divs = x_v_flat.shape[-1] / x_a_enc.shape[-1]
            self.pool_v = True
        else:
            divs = x_a_enc.shape[-1] / x_v_flat.shape[-1]
            self.pool_v = False

        self.latentPool = nn.MaxPool2d((1, int(divs)))
        x_v_flat = self.latentPool(x_v_flat)

        print(f'x_v_flattened shape {x_v_flat.shape}')

        av_concat = torch.cat((x_a_enc, x_v_flat), dim=1)

        # now tensor is [..., channels, timesteps, spatial]
        in_ch = av_concat.shape[1]
        out_ch = in_ch//2

        modules = []

        while out_ch > 1:
            out_ch = in_ch//2
            modules.append(nn.ZeroPad2d((1, 1, 0, 0)))
            # kernel size convolves over shared spatial dimension (-1), doesn't touch temporal
            modules.append(nn.Conv2d(in_ch, out_ch, kernel_size=(1, 3), stride=(1, 1)))
            in_ch = out_ch

        self.av_featureNet = nn.Sequential(*modules)
        # print(self.av_featureNet)
        summary(self.av_featureNet.to("cuda"), input_size=(av_concat.shape[1], av_concat.shape[2], av_concat.shape[3]))

        with torch.no_grad():
            av_features = self.av_featureNet(av_concat)

        # we've condensed dims down to 1, squeeze it out
        av_features = av_features.squeeze(1)
        av_features = torch.flatten(av_features, start_dim=-2, end_dim=-1)

        print(f'av features {av_features.shape}')

        fc_output_neurons = 512

        self.av_fcNet = nn.Sequential(
            nn.Linear(av_features.shape[-1], fc_output_neurons, bias=False),
            nn.LayerNorm(fc_output_neurons, elementwise_affine=True),
            nn.LeakyReLU(0.3),
            # nn.Linear(512, fc_output_neurons, bias=False),
            # nn.LayerNorm(fc_output_neurons, elementwise_affine=True),
            # nn.LeakyReLU(),
        )

        summary(self.av_fcNet.to("cuda"), input_size=(av_features.shape[-1],))
        av_fc = self.av_fcNet(av_features)
        print(f'AV FC SHAPE {av_fc.shape}')
        # self.av_fc1 = nn.Linear(av_features.shape[-1], 512, bias=False).to('cuda')
        # self.av_fc1_ln = nn.LayerNorm(512, elementwise_affine=True).to('cuda')

        a_head_neurons = x_a_enc.shape[1] * x_a_enc.shape[2] * x_a_enc.shape[3]
        v_head_neurons = x_v_enc.shape[1] * x_v_enc.shape[2] * x_v_enc.shape[3] * x_v_enc.shape[4]
        self.a_fc_out = nn.Linear(fc_output_neurons, a_head_neurons).to('cuda')
        self.v_fc_out = nn.Linear(fc_output_neurons, v_head_neurons).to('cuda')

        x_a_head = self.a_fc_out(av_fc)
        x_v_head = self.v_fc_out(av_fc)

        x_a_head = torch.reshape(x_a_head, x_a_enc.shape) # remove batch dim
        x_v_head = torch.reshape(x_v_head, x_v_enc.shape)

        print(f"x_a_head {x_a_head.shape}")
        print(f"x_v_head {x_v_head.shape}")

        a_head_shape = x_a_head.shape[1:]

        self.audio_up1= nn.ConvTranspose2d(a_head_shape[0], a_head_shape[0]//2, kernel_size=(3, 3), stride=(2, 2), padding=1).to('cuda')
        self.audio_up2 = nn.ConvTranspose2d(a_head_shape[0]//2, a_head_shape[0]//4, kernel_size=(3, 3), stride=(2, 2), padding=1).to('cuda')
        self.audio_up3 = nn.ConvTranspose2d(a_head_shape[0]//4, a_head_shape[0]//8, kernel_size=(3, 3), stride=(1, 2), padding=1).to('cuda')
        self.audio_up4 = nn.ConvTranspose2d(a_head_shape[0]//8, 2, kernel_size=(3, 3), stride=(1, 2), padding=1).to('cuda')

        
        x_a_out = self.audio_up1(x_a_head, output_size=(a_head_shape[1] * 2, a_head_shape[2] * 2))
        x_a_out = self.audio_up2(x_a_out, output_size=(a_head_shape[1] * 4, a_head_shape[2] * 4))
        x_a_out = self.audio_up3(x_a_out, output_size=(a_head_shape[1] * 4, a_head_shape[2] * 8))
        x_a_out = self.audio_up4(x_a_out, output_size=(a_head_shape[1] * 4, a_head_shape[2] * 16))

        v_head_shape = x_v_head.shape[1:]

        self.video_up1= nn.ConvTranspose3d(v_head_shape[0], v_head_shape[0]//2, kernel_size=(1, 3, 3), stride=(1, 4, 4), padding=(0, 1, 1)).to('cuda')
        self.video_up2 = nn.ConvTranspose3d(v_head_shape[0]//2, v_head_shape[0]//4, kernel_size=(1, 3, 3), stride=(1, 4, 4), padding=(0, 1, 1)).to('cuda')
        self.video_up3 = nn.ConvTranspose3d(v_head_shape[0]//4, v_head_shape[0]//8, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)).to('cuda')
        self.video_up4 = nn.ConvTranspose3d(v_head_shape[0]//8, 1, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1).to('cuda')

        x_v_out = self.video_up1(x_v_head, output_size=(v_head_shape[1], v_head_shape[2] * 4, v_head_shape[3] * 4))
        x_v_out = self.video_up2(x_v_out, output_size=(v_head_shape[1], v_head_shape[2] * 16, v_head_shape[3] * 16))
        x_v_out = self.video_up3(x_v_out, output_size=(v_head_shape[1], v_head_shape[2] * 32, v_head_shape[3] * 32))
        x_v_out = self.video_up4(x_v_out, output_size=(v_head_shape[1], v_head_shape[2] * 64, v_head_shape[3] * 64))

        print(f'x v out sh {x_v_out.shape}')
        modules = []


        print("\n ######################################################################### \n")

    def forward(self, x_a, x_v):

        x_a_enc = self.audio_net(x_a)
        x_v_enc = self.visual_net(x_v)

        x_v_flat = torch.flatten(x_v_enc, start_dim=-2, end_dim=-1)

        if self.pool_v:
            x_v_flat = self.latentPool(x_v_flat)
        else:
            x_a_enc = self.latentPool(x_a_enc)

        # concatenate along channel axis
        av_concat = torch.cat((x_a_enc, x_v_flat), dim=1)

        # get features in 2d convnet
        av_features = self.av_featureNet(av_concat)

        # we've condensed dims down to 1, squeeze it out
        av_features = av_features.squeeze(1)
        av_features = torch.flatten(av_features, start_dim=-2, end_dim=-1)

        av_fc = self.av_fcNet(av_features)

        x_a_head = self.a_fc_out(av_fc)
        x_a_head = F.leaky_relu(x_a_head, negative_slope=0.3)
        x_a_head = x_a_head.reshape(x_a_enc.shape)
        a_head_shape = x_a_head.shape[1:]

        x_a_out = self.audio_up1(x_a_head, output_size=(a_head_shape[1] * 2, a_head_shape[2] * 2))
        x_a_out = self.audio_up2(x_a_out, output_size=(a_head_shape[1] * 4, a_head_shape[2] * 4))
        x_a_out = self.audio_up3(x_a_out, output_size=(a_head_shape[1] * 4, a_head_shape[2] * 8))
        x_a_out = self.audio_up4(x_a_out, output_size=(a_head_shape[1] * 4, a_head_shape[2] * 16))

        
        x_v_head = self.v_fc_out(av_fc)
        x_v_head = F.leaky_relu(x_v_head, negative_slope=0.3)
        x_v_head = torch.reshape(x_v_head, x_v_enc.shape)
        v_head_shape = x_v_head.shape[1:]

        x_v_out = self.video_up1(x_v_head, output_size=(v_head_shape[1], v_head_shape[2] * 4, v_head_shape[3] * 4))
        x_v_out = self.video_up2(x_v_out, output_size=(v_head_shape[1], v_head_shape[2] * 16, v_head_shape[3] * 16))
        x_v_out = self.video_up3(x_v_out, output_size=(v_head_shape[1], v_head_shape[2] * 32, v_head_shape[3] * 32))
        x_v_out = self.video_up4(x_v_out, output_size=(v_head_shape[1], v_head_shape[2] * 64, v_head_shape[3] * 64))
        # reconstruction of video attention frames
        # x_v = self.v_fc_out(x_v)
        # x_v = F.leaky_relu(x_v, negative_slope=0.3) 
        # # reshape tensor into original dimensions
        # x_v = x_v.reshape(self.v_shape)

        return x_a_out, x_v_out

# architecture - Hou et. al
# Audio-Visual Speech Enhancement Using Multimodal Deep Convolutional Neural Networks
# https://arxiv.org/pdf/1703.10893.pdf

# def build_model(in_1_shape, in_2_shape, out_1_shape, out_2_shape, padding="same"):

#     input_a = Input(shape=in_1_shape, name="audio-input")
#     input_v = Input(shape=in_2_shape, name="video-input")

#     an_0 = Conv1D(10, kernel_size=(9), activation="linear", padding=padding)(input_a)
#     an_0 = AveragePooling1D(pool_size=2)(an_0)
#     an_1 = Conv1D(4, kernel_size=5, activation="linear", padding=padding)(an_0)


#     vn_0 = Conv3D(12, 9, strides=(1,2,2), padding=padding)(input_v)
#     vn_1 = Conv3D(12, 9, strides=(2,2,2), padding=padding)(vn_0)
#     vn_2 = Conv3D(10, 9, strides=(2,2,2), padding=padding)(vn_1)
#     vn_3 = Conv3D(6, 7, strides=(2,2,2), padding=padding)(vn_2)
#     vn_4 = Conv3D(4, 5, strides=(2,2,2), padding=padding)(vn_3)
#     vn_5 = Conv3D(2, 5, strides=(2,2,2), padding=padding)(vn_4)

#     vn_rshp = Reshape((1, 32))(vn_5)

#     concat = concatenate([an_1, vn_rshp])

#     FC1 = Dense(1000, activation="sigmoid")(concat)
#     FC2 = Dense(512, activation="sigmoid")(FC1)

#     FC3a = Dense(out_1_shape[0] * (out_1_shape[1]//2), activation="linear")(FC2)
#     FC3a = Reshape((2, 533))(FC3a)
#     CoutA1 = Conv1D(out_1_shape[1], kernel_size=1, padding=padding)(FC3a)

#     CoutV1 = Reshape((1, 16, 32, 1))(FC2)
#     CoutV1 = Conv3D(8, kernel_size=5, padding=padding)(CoutV1)
#     CoutV1 = UpSampling3D(size=(2,4,2))(CoutV1)
#     CoutV2 = Conv3D(10, kernel_size=5, padding=padding)(CoutV1)
#     CoutV2 = UpSampling3D(size=(2,4,4))(CoutV2)
#     CoutV3 = Conv3D(3, kernel_size=3, padding=padding)(CoutV2)

#     model = Model(inputs=[input_a, input_v], outputs=[CoutA1, CoutV3])

#     return model
