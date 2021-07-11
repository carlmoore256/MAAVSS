from math import e
from torch.functional import stft
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules import module
from torch.nn.modules.activation import ReLU
from torchsummary import summary
import copy

class AVSE_Model(nn.Module):
    def __init__(self, a_shape, v_shape):
        super(AVSE_Model, self).__init__()

        self.a_shape = a_shape
        self.v_shape = v_shape

        print(f"\nA SHAPE {a_shape}")
        print(f"V SHAPE {v_shape}\n")

        self.audio_encoder = nn.Sequential(
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

        self.visual_encoder = nn.Sequential(
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

        y_a = self.audio_encoder(x_a)
        y_v = self.visual_encoder(x_v)

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

        modules = []
        in_ch = 2
        for i in range(alpha):
            # out_ch = (alpha - (i + 1)) * 2 + 2
            out_ch = in_ch * 2
            modules.append(nn.ZeroPad2d((2, 2, 3, 1))) # add 1 to each to increase k size by 2
            if i < n_div:
                modules.append(nn.Conv2d(in_ch, out_ch, kernel_size=(5, 5), stride=(2, 2)))
            else:
                modules.append(nn.Conv2d(in_ch, out_ch, kernel_size=(5, 5), stride=(1, 2)))
            
            modules.append(nn.BatchNorm2d(out_ch))
            modules.append(nn.Tanh())
            # modules.append(nn.MaxPool2d((2,1)))
            in_ch = out_ch
        self.audio_encoder = nn.Sequential(*modules)

        summary(self.audio_encoder.to("cuda"), input_size=(stft_shape[1], stft_shape[2], stft_shape[3]))

        x_a = torch.rand(stft_shape).to("cuda")
        with torch.no_grad():
            x_a_enc = self.audio_encoder(x_a)

        modules = []
        spatial_dim = v_shape[3]
        in_ch = 1

        # while time_dim > audio_output.shape[-1]:
        while spatial_dim > x_a_enc.shape[-1]//2:
            out_ch = in_ch * 2
            modules.append(nn.Conv3d(in_ch, out_ch, kernel_size=(3,3,3), stride=(1,1,1), padding=(1, 1, 1), padding_mode="zeros"))
            modules.append(nn.BatchNorm3d(out_ch))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool3d(kernel_size=(1,2,2)))
            spatial_dim /= 2
            in_ch = out_ch

        self.visual_encoder = nn.Sequential(*modules)
        summary(self.visual_encoder.to("cuda"), input_size=(v_shape[1], v_shape[2], v_shape[3], v_shape[4]))
        x_v = torch.rand(v_shape).to("cuda")

        with torch.no_grad():
            x_v_enc = self.visual_encoder(x_v)
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
            modules.append(nn.ReLU())
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
            nn.ReLU(),
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

        self.audio_up1= nn.ConvTranspose2d(a_head_shape[0], a_head_shape[0]//2, kernel_size=(5, 5), stride=(2, 2), padding=2).to('cuda')
        self.audio_up1_norm = nn.BatchNorm2d(a_head_shape[0]//2)
        self.audio_up2 = nn.ConvTranspose2d(a_head_shape[0]//2, a_head_shape[0]//4, kernel_size=(5, 5), stride=(2, 2), padding=2).to('cuda')
        self.audio_up2_norm = nn.BatchNorm2d(a_head_shape[0]//4)
        self.audio_up3 = nn.ConvTranspose2d(a_head_shape[0]//4, a_head_shape[0]//8, kernel_size=(5, 5), stride=(1, 2), padding=2).to('cuda')
        self.audio_up3_norm = nn.BatchNorm2d(a_head_shape[0]//8)
        self.audio_up4 = nn.ConvTranspose2d(a_head_shape[0]//8, 2, kernel_size=(3, 3), stride=(1, 2), padding=1).to('cuda')
        # self.audio_up4_norm = nn.BatchNorm2d(2)

        # self.audio_decoder = nn.Sequential(
        #     nn.ConvTranspose2d(a_head_shape[0], a_head_shape[0]//2, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1),
        #     nn.Tanh(),
        #     nn.ConvTranspose2d(a_head_shape[0]//2, a_head_shape[0]//4, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1),
        #     nn.Tanh(),
        #      nn.ConvTranspose2d(a_head_shape[0]//4, a_head_shape[0]//8, kernel_size=(3, 3), stride=(1, 2), padding=1)
        # )

        self.audio_ae = nn.Sequential(*self.audio_encoder,
                            self.audio_up1,
                            self.audio_up1_norm,
                            self.audio_up2,
                            self.audio_up2_norm,
                            self.audio_up3,
                            self.audio_up3_norm,
                            self.audio_up4)
        
        x_a_out = self.audio_up1(x_a_head, output_size=(a_head_shape[1] * 2, a_head_shape[2] * 2))
        x_a_out = self.audio_up2(x_a_out, output_size=(a_head_shape[1] * 4, a_head_shape[2] * 4))
        x_a_out = self.audio_up3(x_a_out, output_size=(a_head_shape[1] * 4, a_head_shape[2] * 8))
        x_a_out = self.audio_up4(x_a_out, output_size=(a_head_shape[1] * 4, a_head_shape[2] * 16))

        v_head_shape = x_v_head.shape[1:]

        self.video_up1= nn.ConvTranspose3d(v_head_shape[0], v_head_shape[0]//2, kernel_size=(3, 3, 3), stride=(1, 4, 4), padding=(1, 1, 1)).to('cuda')
        self.video_up1_norm = nn.BatchNorm3d(v_head_shape[0]//2).to('cuda')
        self.video_up2 = nn.ConvTranspose3d(v_head_shape[0]//2, v_head_shape[0]//4, kernel_size=(3, 3, 3), stride=(1, 4, 4), padding=(1, 1, 1)).to('cuda')
        self.video_up2_norm = nn.BatchNorm3d(v_head_shape[0]//4).to('cuda')
        self.video_up3 = nn.ConvTranspose3d(v_head_shape[0]//4, v_head_shape[0]//8, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)).to('cuda')
        self.video_up3_norm = nn.BatchNorm3d(v_head_shape[0]//8).to('cuda')
        self.video_up4 = nn.ConvTranspose3d(v_head_shape[0]//8, 1, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1).to('cuda')

        x_v_out = self.video_up1(x_v_head, output_size=(v_head_shape[1], v_head_shape[2] * 4, v_head_shape[3] * 4))
        x_v_out = self.video_up2(x_v_out, output_size=(v_head_shape[1], v_head_shape[2] * 16, v_head_shape[3] * 16))
        x_v_out = self.video_up3(x_v_out, output_size=(v_head_shape[1], v_head_shape[2] * 32, v_head_shape[3] * 32))
        x_v_out = self.video_up4(x_v_out, output_size=(v_head_shape[1], v_head_shape[2] * 64, v_head_shape[3] * 64))
        
        self.visual_ae = nn.Sequential(*self.visual_encoder,
                                        self.video_up1,
                                        self.video_up1_norm,
                                        self.video_up2,
                                        self.video_up2_norm,
                                        self.video_up3,
                                        self.video_up3_norm,
                                        self.video_up4)
        # print(f'x v out sh {x_v_out.shape}')        


        print("\n ######################################################################### \n")

    def toggle_av_grads(self, toggle):
        for param in self.av_featureNet.parameters():
            param.requires_grad = toggle
            
        for param in self.a_fc_out.parameters():
            param.requires_grad = toggle

        for param in self.v_fc_out.parameters():
            param.requires_grad = toggle

    def toggle_visual_grads(self, toggle):
        for param in self.visual_ae:
            param.requires_grad = toggle

    def toggle_audio_grads(self, toggle):
        for param in self.audio_ae:
            param.requires_grad = toggle
    
    def audio_ae_forward(self, x_a):
        x_a_enc = self.audio_encoder(x_a)
        a_head_shape = x_a_enc.shape[1:]
        x_a_out = self.audio_up1(x_a_enc, output_size=(a_head_shape[1] * 2, a_head_shape[2] * 2))
        x_a_out = self.audio_up1_norm(x_a_out)
        x_a_out = F.tanh(x_a_out)
        x_a_out = self.audio_up2(x_a_out, output_size=(a_head_shape[1] * 4, a_head_shape[2] * 4))
        x_a_out = self.audio_up2_norm(x_a_out)
        x_a_out = F.tanh(x_a_out)
        x_a_out = self.audio_up3(x_a_out, output_size=(a_head_shape[1] * 4, a_head_shape[2] * 8))
        x_a_out = self.audio_up3_norm(x_a_out)
        x_a_out = F.tanh(x_a_out)
        x_a_out = self.audio_up4(x_a_out, output_size=(a_head_shape[1] * 4, a_head_shape[2] * 16))
        # x_a_out = self.audio_up4_norm(x_a_out)
        return x_a_out

    def visual_ae_forward(self, x_v):
        x_v_enc = self.visual_encoder(x_v)
        v_head_shape = x_v_enc.shape[1:]
        x_v_out = self.video_up1(x_v_enc, output_size=(v_head_shape[1], v_head_shape[2] * 4, v_head_shape[3] * 4))
        x_v_out = self.video_up1_norm(x_v_out)
        x_v_out = F.relu(x_v_out)
        x_v_out = self.video_up2(x_v_out, output_size=(v_head_shape[1], v_head_shape[2] * 16, v_head_shape[3] * 16))
        x_v_out = self.video_up2_norm(x_v_out)
        x_v_out = F.relu(x_v_out)
        x_v_out = self.video_up3(x_v_out, output_size=(v_head_shape[1], v_head_shape[2] * 32, v_head_shape[3] * 32))
        x_v_out = self.video_up3_norm(x_v_out)
        x_v_out = F.relu(x_v_out)
        x_v_out = self.video_up4(x_v_out, output_size=(v_head_shape[1], v_head_shape[2] * 64, v_head_shape[3] * 64))
        x_v_out = F.sigmoid(x_v_out)
        return x_v_out

    def forward(self, x_a, x_v, train_ae=False):

        x_a_enc = self.audio_encoder(x_a)
        x_v_enc = self.visual_encoder(x_v)
        
        if not train_ae:
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
            av_fc = F.relu(av_fc)

            x_a_head = self.a_fc_out(av_fc)
            x_a_head = F.relu(x_a_head)
            x_a_head = x_a_head.reshape(x_a_enc.shape)

            x_v_head = self.v_fc_out(av_fc)
            x_v_head = F.relu(x_v_head)
            x_v_head = torch.reshape(x_v_head, x_v_enc.shape)
        else: # train the autoencoder
            x_a_head = x_a_enc
            x_v_head = x_v_enc

        a_head_shape = x_a_head.shape[1:]
        v_head_shape = x_v_head.shape[1:]

        x_a_out = self.audio_up1(x_a_head, output_size=(a_head_shape[1] * 2, a_head_shape[2] * 2))
        x_a_out = self.audio_up1_norm(x_a_out)
        x_a_out = F.tanh(x_a_out)
        x_a_out = self.audio_up2(x_a_out, output_size=(a_head_shape[1] * 4, a_head_shape[2] * 4))
        x_a_out = self.audio_up2_norm(x_a_out)
        x_a_out = F.tanh(x_a_out)
        x_a_out = self.audio_up3(x_a_out, output_size=(a_head_shape[1] * 4, a_head_shape[2] * 8))
        x_a_out = self.audio_up3_norm(x_a_out)
        x_a_out = F.tanh(x_a_out)
        x_a_out = self.audio_up4(x_a_out, output_size=(a_head_shape[1] * 4, a_head_shape[2] * 16))
        # x_a_out = self.audio_up4_norm(x_a_out)
        # x_a_out = F.tanh(x_a_out)

        
        x_v_out = self.video_up1(x_v_head, output_size=(v_head_shape[1], v_head_shape[2] * 4, v_head_shape[3] * 4))
        x_v_out = F.sigmoid(x_v_out)
        x_v_out = self.video_up2(x_v_out, output_size=(v_head_shape[1], v_head_shape[2] * 16, v_head_shape[3] * 16))
        x_v_out = F.sigmoid(x_v_out)
        x_v_out = self.video_up3(x_v_out, output_size=(v_head_shape[1], v_head_shape[2] * 32, v_head_shape[3] * 32))
        x_v_out = F.sigmoid(x_v_out)
        x_v_out = self.video_up4(x_v_out, output_size=(v_head_shape[1], v_head_shape[2] * 64, v_head_shape[3] * 64))
        x_v_out = F.sigmoid(x_v_out)

        # reconstruction of video attention frames
        # x_v = self.v_fc_out(x_v)
        # x_v = F.leaky_relu(x_v, negative_slope=0.3) 
        # # reshape tensor into original dimensions
        # x_v = x_v.reshape(self.v_shape)

        return x_a_out, x_v_out

##################################################################################################
# final version of the model
# takes in stft and phase motion vectors (phasegram)
class AV_Fusion_Model(nn.Module):

    def __init__(self, 
                stft_shape, 
                pgram_shape, 
                alpha, 
                latent_channels=64,
                fc_size=4096):

        super(AV_Fusion_Model, self).__init__()
        
        self.stft_shape = stft_shape
        self.pgram_shape = pgram_shape
        self.latent_channels = latent_channels

        ############### PHASEGRAM ENCODER #################

        modules = []
        tensor = torch.rand(pgram_shape)
        in_ch = 1

        # while tensor.shape[-1] > tensor.shape[-2]:
        while tensor.shape[-1] * tensor.shape[-2] * latent_channels > fc_size // 2:

            out_ch = in_ch * 2
            out_ch = min(out_ch, latent_channels)
            conv_layer = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 9), stride=(1,2), padding=(0, 4))
            with torch.no_grad():
                tensor = conv_layer(tensor)
            modules.append(conv_layer)
            modules.append(nn.BatchNorm2d(out_ch))
            modules.append(nn.Tanh())
            # modules.append(nn.LeakyReLU(negative_slope=0.3))
            in_ch = out_ch

        self.phasegram_encoder = nn.Sequential(*modules).to("cuda")

        ############### PHASEGRAM DECODER #################
        modules = []
        in_ch = latent_channels

        while tensor.shape[-1] < pgram_shape[-1]:
            out_ch = in_ch // 2
            out_ch = max(out_ch, 1)
            conv_layer = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(1, 9), stride=(1,2), padding=(0,4), output_padding=(0,1))
            modules.append(conv_layer)
            with torch.no_grad():
                tensor = conv_layer(tensor)
            if tensor.shape[-1] != pgram_shape[-1]:
                modules.append(nn.BatchNorm2d(out_ch))
                modules.append(nn.Tanh())
            in_ch = out_ch
        # modules.append(nn.Tanh())
        self.phasegram_decoder = nn.Sequential(*modules).to("cuda")

        x_v = torch.rand(pgram_shape).to("cuda")

        with torch.no_grad():
            x_v = self.phasegram_encoder(x_v)

        print(f'XV SHAPE {x_v.shape}')
        ############### STFT ENCODER #################

        modules = []
        in_ch = 2
        spatial_dim = stft_shape[-1]
        temporal_dim = stft_shape[-2]
        output_shape = [temporal_dim, spatial_dim]

        while output_shape != list(x_v.shape[2:]):
            out_ch = in_ch * 4
            if out_ch > latent_channels:
                out_ch = latent_channels
                
            stride = [1, 1]

            if output_shape[0] > x_v.shape[-2]:
                stride[0] = 2
                output_shape[0] = output_shape[0] // 2
            if output_shape[1] > x_v.shape[-1]:
                stride[1] = 2
                output_shape[1] = output_shape[1] // 2

            modules.append(nn.Conv2d(in_ch, out_ch, 
                                    kernel_size=(5, 5), 
                                    stride=tuple(stride), 
                                    padding=(2,2)))
            modules.append(nn.BatchNorm2d(out_ch))
            modules.append(nn.Tanh())
            in_ch = out_ch

        self.stft_encoder = nn.Sequential(*modules).to("cuda")

        x_a = torch.rand(stft_shape).to("cuda")
        with torch.no_grad():
            x_a = self.stft_encoder(x_a)

        ################ AV FUSION NET #################

        out_ch = fc_size // (output_shape[0] * output_shape[1])
        # our spatial dim must be low enough to accomadate fc size

        # number of channels out of the second fc layer
        fc2_out = latent_channels * (output_shape[0] * output_shape[1])

        # self.fusion_net = nn.Sequential(
        #     nn.Conv2d(in_channels=latent_channels * 2, 
        #               out_channels=out_ch, 
        #               kernel_size=(3,3), 
        #               padding=(1,1)),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(),
        #     nn.Flatten(1, 3),
        #     nn.Linear(fc_size, fc_size//2),
        #     nn.ReLU(),
        #     nn.Linear(fc_size//2, fc2_out),
        #     nn.ReLU()
        # ).to("cuda")
        # summary(self.fusion_net.to("cuda"), 
        #         input_size=(latent_channels*2, output_shape[0], output_shape[1]))
        
        # x_av_cat = torch.cat((x_v, x_a), dim=1)
        # re-arrange dimensions so that time dim is properly arranged for lstm
        x_v = x_v.permute(0, 2, 1, 3)
        x_a = x_a.permute(0, 2, 1, 3)
        x_av_cat = torch.cat((x_v, x_a), dim=2)
        print(f'X AV CAT SH {x_av_cat.shape}')
        x_av_cat = torch.flatten(x_av_cat, start_dim=-2, end_dim=-1)
        print(f'X AV CAT SH {x_av_cat.shape}')
        self.lstm = nn.LSTM(input_size=x_av_cat.shape[-1], hidden_size=256, num_layers=1, bias=False, batch_first=True, dropout=0, bidirectional=True).to("cuda")
        av = self.lstm(x_av_cat)[0]
        print(f'AV LSTM SH {av.shape}')
        av = torch.flatten(av, start_dim=1)
        print(f'AV FLATTENED SH {av.shape}')
        self.fc1 = nn.Linear(fc_size, fc_size//2).to("cuda")
        av = self.fc1(av)
        print(f'AV Lin1 SH {av.shape}')
        self.fc2 = nn.Linear(fc_size//2, fc2_out).to("cuda")
        av = self.fc2(av)
        print(f'AV Lin2 SH {av.shape}')

        with torch.no_grad():
            x_av_fused = self.av_fusion_forward(x_a, x_v)

        print(f'FUSED SHAPE {x_av_fused.shape}')

        ############### STFT DECODER #################

        modules = []
        in_ch = latent_channels
        encoded_shape = [x_a.shape[2], x_a.shape[3]]

        while encoded_shape != [temporal_dim, spatial_dim]:
            out_ch = in_ch // 4
            if out_ch < stft_shape[1]:
                out_ch = stft_shape[1]
            stride = [1, 1]
            out_padding = [0, 0]
            if encoded_shape[0] < temporal_dim:
                stride[0] = 2
                out_padding[0] = 1
                encoded_shape[0] = encoded_shape[0] * 2
            if encoded_shape[1] < spatial_dim:
                stride[1] = 2
                out_padding[1] = 1
                encoded_shape[1] = encoded_shape[1] * 2
            modules.append(nn.ConvTranspose2d(in_ch, out_ch, 
                                    kernel_size=(5, 5), 
                                    stride=tuple(stride), 
                                    padding=(2,2),
                                    output_padding=tuple(out_padding)))
            if encoded_shape != [temporal_dim, spatial_dim]:
                modules.append(nn.BatchNorm2d(out_ch))
                modules.append(nn.Tanh())
            in_ch = out_ch

        self.stft_decoder = nn.Sequential(*modules)

        ############### STFT & PHASEGRAM AUTOENCODERS #################

        self.stft_autoencoder = nn.Sequential(
            *self.stft_encoder,
            *self.stft_decoder
        ).to("cuda")

        print(f'\n ########## STFT AUTOENCODER ##########\n')
        print(f'Input shape {stft_shape}')
        summary(self.stft_autoencoder, 
                input_size=(stft_shape[1], stft_shape[2], stft_shape[3]))

        self.phasegram_autoencoder = nn.Sequential(
            *self.phasegram_encoder,
            *self.phasegram_decoder
        ).to("cuda")

        print(f'\n ########## PHASEGRAM AUTOENCODER ##########\n')
        print(f'Input shape {pgram_shape}')
        summary(self.phasegram_autoencoder, 
                input_size=(pgram_shape[1], pgram_shape[2], pgram_shape[3]))

    # enable these grads for training the fusion network
    def toggle_fusion_grads(self, toggle):
        self.lstm.requires_grad = toggle
        self.fc1.requires_grad = toggle
        self.fc2.requires_grad = toggle
        # for param in self.fusion_net:
        #     param.requires_grad = toggle

    # disable encoder grads for training the av fusion
    def toggle_enc_grads(self, toggle):
        for param in self.stft_encoder:
            param.requires_grad = toggle
        for param in self.phasegram_encoder:
            param.requires_grad = toggle

    def toggle_dec_grads(self, toggle):
        for param in self.stft_decoder:
            param.requires_grad = False
        for param in self.phasegram_decoder:
            param.requires_grad = False

    def av_fusion_forward(self, x_a, x_v):
            # re-arrange dimensions so that time dim is properly arranged for lstm
            x_v = x_v.permute(0, 2, 1, 3)
            x_a = x_a.permute(0, 2, 1, 3)
            x_av_cat = torch.cat((x_v, x_a), dim=2)
            x_av_cat = torch.flatten(x_av_cat, start_dim=-2, end_dim=-1)
            av = self.lstm(x_av_cat)[0]
            av = torch.flatten(av, start_dim=1)
            av = self.fc1(av)
            av = F.leaky_relu(av, negative_slope=0.3)
            av = self.fc2(av)
            av = F.leaky_relu(av, negative_slope=0.3)
            return av

    def visual_ae_forward(self, x_v):
        x_v = self.phasegram_autoencoder(x_v)
        return x_v
    
    def audio_ae_forward(self, x_a):
        x_a = self.stft_autoencoder(x_a)
        return x_a

    def forward(self, x_a, x_v):
        x_a = self.stft_encoder(x_a)
        x_v = self.phasegram_encoder(x_v)

        # x_av_cat = torch.cat((x_v, x_a), dim=1)

        x_av_fused = self.av_fusion_forward(x_a, x_v)
        # these both share the same dims
        x_av_fused = torch.reshape(x_av_fused,
            (x_av_fused.shape[0], self.latent_channels, x_a.shape[2], x_a.shape[3]))
       
        # x_a_latent = torch.reshape(x_av_fused,
                                # (x_av_fused.shape[0], self.latent_channels, x_a.shape[2], x_a.shape[3]))

        # x_v_latent = torch.reshape(x_av_fused, 
        #                         (x_av_fused.shape[0], self.latent_channels, x_v.shape[2], x_v.shape[3]))
        
        x_a_out = self.stft_decoder(x_av_fused)
        x_v_out = self.phasegram_decoder(x_av_fused)

        return x_a_out, x_v_out, x_av_fused



def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride[0]) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride[1]) + 1)
    return h, w
# architecture - Hou et. al
# Audio-Visual Speech Enhancement Using Multimodal Deep Convolutional Neural Networks
# https://arxiv.org/pdf/1703.10893.pdf

