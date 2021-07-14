from math import e
from torch.functional import stft
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules import module
from torch.nn.modules.activation import ReLU
from torchsummary import summary
import copy

##################################################################################################
# final version of the model
# takes in stft and phase motion vectors (phasegram)
class AV_Fusion_Model(nn.Module):

    def __init__(self, 
                stft_shape, 
                pgram_shape, 
                hops_per_frame, 
                latent_channels=16,
                fc_size=4096):

        super(AV_Fusion_Model, self).__init__()
        self.stft_shape = stft_shape
        self.pgram_shape = pgram_shape
        self.latent_channels = latent_channels
        self.output_stft_frames = hops_per_frame
        ############### PHASEGRAM ENCODER #################

        modules = []
        tensor = torch.rand(pgram_shape)
        in_ch = 1

        self.phasegram_encoder = nn.Sequential(
          nn.Conv2d(1, 32, (3, 9), 1, (1, 4), bias=False),
          nn.BatchNorm2d(32),
          nn.AvgPool2d((1, 4)),

          nn.Conv2d(32, 32, (3, 9), 1, (1, 4), bias=False),
          nn.BatchNorm2d(32),
          nn.AvgPool2d((1, 4)),

          nn.Conv2d(32, 16, (3, 9), 1, (1, 4), bias=False),
          nn.BatchNorm2d(16),
          nn.AvgPool2d((1, 4))
        ).to("cuda")

        tensor = self.phasegram_encoder(tensor.to("cuda"))

        print(f'tensor shape {tensor.shape}')

        # while tensor.shape[-1] > tensor.shape[-2]:
        # while tensor.shape[-1] * tensor.shape[-2] * latent_channels > fc_size // 2:

        #     out_ch = in_ch * 2
        #     out_ch = min(out_ch, latent_channels)
        #     conv_layer = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 9), stride=(1,2), padding=(0, 4), bias=False)
        #     with torch.no_grad():
        #         tensor = conv_layer(tensor)
        #     modules.append(conv_layer)
        #     modules.append(nn.BatchNorm2d(out_ch))
        #     modules.append(nn.Tanh())

        #     # modules.append(nn.LeakyReLU(negative_slope=0.3))
        #     in_ch = out_ch

        # self.phasegram_encoder = nn.Sequential(*modules).to("cuda")

        ############### PHASEGRAM DECODER #################
        modules = []
        in_ch = latent_channels

        self.phasegram_decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, (3, 9), stride=(1, 4), padding=(1, 4), output_padding=(0,3), bias=False),
            nn.BatchNorm2d(32),
            # nn.Tanh(),
            nn.ConvTranspose2d(32, 16, (3, 9), stride=(1, 4), padding=(1, 4), output_padding=(0,3), bias=False),
            nn.BatchNorm2d(16),
            # nn.Tanh(),
            nn.ConvTranspose2d(16, 1, (3, 9), stride=(1, 4), padding=(1, 4),output_padding=(0,3), bias=False),
            # nn.BatchNorm2d(1),
        ).to("cuda")

        tensor = self.phasegram_decoder(tensor)
        print(f'tensor dec shape {tensor.shape}')

        # while tensor.shape[-1] < pgram_shape[-1]:
        #     out_ch = in_ch // 2
        #     out_ch = max(out_ch, 1)
        #     conv_layer = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(1, 9), stride=(1,2), padding=(0,4), output_padding=(0,1), bias=False)
        #     modules.append(conv_layer)
        #     with torch.no_grad():
        #         tensor = conv_layer(tensor)
        #     if tensor.shape[-1] != pgram_shape[-1]:
        #         modules.append(nn.BatchNorm2d(out_ch))
        #         modules.append(nn.Tanh())
        #     in_ch = out_ch
        # # modules.append(nn.Tanh())
        # self.phasegram_decoder = nn.Sequential(*modules).to("cuda")

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
            out_ch = in_ch * 2
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
                                    kernel_size=(3, 9), 
                                    stride=tuple(stride), 
                                    padding=(1,4),
                                    bias=False))
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
        # fc2_out = latent_channels * (output_shape[0] * output_shape[1])

        # fc2_out = latent_channels * x_v.shape[-2] * x_v.shape[-1]
        fc2_out = 512

        # x_av_cat = torch.cat((x_v, x_a), dim=1)
        # re-arrange dimensions so that time dim is properly arranged for lstm
        x_av_cat = torch.cat((x_v.permute(0, 2, 1, 3), x_a.permute(0, 2, 1, 3)), dim=2)
        print(f'x_v {x_v.shape} x_a {x_a.shape}')
        x_av_cat = torch.flatten(x_av_cat, start_dim=-2, end_dim=-1)
        print(f'x_av_cat sh {x_av_cat.shape}')

        # conisder making bidirectional
        self.lstm = nn.LSTM(input_size=x_av_cat.shape[-1], hidden_size=256, num_layers=1, 
                            bias=False, batch_first=True, dropout=0, bidirectional=True).to("cuda")

        # self.lstm.flatten_parameters()
        av = self.lstm(x_av_cat)[0]
        print(f'AV LSTM SH {av.shape}')
        av = torch.flatten(av, start_dim=1)
        print(f'AV FLATTENED SH {av.shape} fc size {fc_size}')
        fc_size = av.shape[-1]
        self.fc1 = nn.Linear(fc_size, fc_size//2, bias=False).to("cuda")
        av = self.fc1(av)
        print(f'AV Lin1 SH {av.shape}')
        self.fc2 = nn.Linear(fc_size//2, fc2_out, bias=False).to("cuda")
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
            out_ch = in_ch // 2
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
                                    kernel_size=(3, 9), 
                                    stride=tuple(stride), 
                                    padding=(1,4),
                                    output_padding=tuple(out_padding),
                                    bias=False))
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


        # self.a_fc1 = nn.Linear(fc2_out, stft_shape[-2] * stft_shape[-1])
        self.a_fc1 = nn.Sequential(
          nn.Linear(fc2_out, 2 * self.output_stft_frames * stft_shape[-1], bias=False),
          # nn.LeakyReLU(negative_slope=0.3)
          # nn.Tanh()
        )

        self.v_fc1 = nn.Sequential(
          nn.Linear(fc2_out, 1 * pgram_shape[-1], bias=False),
          # nn.LeakyReLU(negative_slope=0.3)
          # nn.Tanh()
        )

    # enable these grads for training the fusion network
    def toggle_fusion_grads(self, toggle):
        self.lstm.requires_grad_(toggle)
        self.fc1.requires_grad_(toggle)
        self.fc2.requires_grad_(toggle)
        self.a_fc1.requires_grad_(toggle)
        self.v_fc1.requires_grad_(toggle)

    def toggle_stft_ae_grads(self, toggle):
        for param in self.stft_autoencoder:
            param.requires_grad_(toggle)

    def toggle_phasegram_ae_grads(self, toggle):
        for param in self.phasegram_autoencoder:
            param.requires_grad_(toggle)

    # disable encoder grads for training the av fusion
    def toggle_enc_grads(self, toggle):
        for param in self.stft_encoder:
            param.requires_grad_(toggle)
        for param in self.phasegram_encoder:
            param.requires_grad_(toggle)

    def toggle_dec_grads(self, toggle):
        for param in self.stft_decoder:
            param.requires_grad_(toggle)
        for param in self.phasegram_decoder:
            param.requires_grad_(toggle)

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

        x_a_enc = self.stft_encoder(x_a)
        x_v_enc = self.phasegram_encoder(x_v)

        x_av_fused = self.av_fusion_forward(x_a_enc, x_v_enc)

        # print(f'xav fused {x_av_fused.shape}')
        x_a_out = self.a_fc1(x_av_fused)
        # x_a_out = F.leaky_relu(x_a_out, negative_slope=0.3)
        x_v_out = F.tanh(x_a_out)

        x_v_out = self.v_fc1(x_av_fused)
        # x_v_out = F.leaky_relu(x_v_out, negative_slope=0.3)
        x_v_out = F.tanh(x_v_out)

        x_v_out = x_v_out.view(x_v.shape[0], 1, x_v.shape[-1])
        x_a_out = x_a_out.view(x_a.shape[0], 2, self.output_stft_frames, x_a.shape[-1])
        
        # x_a_out = self.stft_decoder(x_a_out)
        # x_v_out = self.phasegram_decoder(x_v_out)

        return x_a_out, x_v_out, x_av_fused


##################################################################################################
# final version of the model
# takes in stft and phase motion vectors (phasegram)
class AV_Fusion_Model_Frames(nn.Module):

    def __init__(self, 
                stft_shape, 
                frame_shape, 
                hops_per_frame, 
                latent_channels=16,
                fc_size=4096):
        
        super(AV_Fusion_Model_Frames, self).__init__()
        self.stft_shape = stft_shape
        self.frame_shape = frame_shape
        self.frame_channels = frame_shape[1]
        self.latent_channels = latent_channels
        self.output_stft_frames = hops_per_frame

        print(f'NUM FRAMES {frame_shape[2]}')
        modules = []
        in_ch = 1

        # self.visual_encoder = nn.Sequential(
        #   nn.Conv2d(1, 16, kernel_size=(5,5), stride=(2,2), padding=(2,2), bias=False),
        #   nn.BatchNorm2d(16),
        # #   nn.MaxPool2d((2, 2)),

        #   nn.Conv2d(16, 32, kernel_size=(5,5), stride=(2,2), padding=(2,2), bias=False),
        #   nn.BatchNorm2d(32),
        # #   nn.MaxPool2d((2, 2)),

        #   nn.Conv2d(32, 64, kernel_size=(5,5), stride=(2,2), padding=(2,2), bias=False),
        #   nn.BatchNorm2d(64),
        # #   nn.MaxPool2d((2, 2)),

        #   nn.Conv2d(64, 64, kernel_size=(5,5), stride=(2,2), padding=(2,2), bias=False),
        #   nn.BatchNorm2d(64),
        # #   nn.MaxPool2d((2, 2)),

        #   nn.Conv2d(64, latent_channels, kernel_size=(5,5), stride=(2,2), padding=(2,2), bias=False),
        #   nn.BatchNorm2d(latent_channels),
        # #   nn.MaxPool2d((2, 2))

        # ).to("cuda")
        self.visual_encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2), bias=False),
            nn.BatchNorm3d(8),
            # nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(8, 16, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2), bias=False),
            nn.BatchNorm3d(16),
            # nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2), bias=False),
            nn.BatchNorm3d(32),
            # nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3,7,7), stride=(1,3,3), padding=(1,3,3), bias=False),
            nn.BatchNorm3d(64),
            # nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(64, latent_channels, kernel_size=(3,7,7), stride=(1,3,3), padding=(1,3,3), bias=False),
            nn.BatchNorm3d(latent_channels),
            # nn.MaxPool3d((1, 2, 2)),
            nn.Flatten(start_dim=-2, end_dim=-1)
        ).to("cuda")

        summary(self.visual_encoder, 
                input_size=(frame_shape[1], frame_shape[2], frame_shape[3], frame_shape[4]))
        
        print(f'frame shape {frame_shape}')
        tensor = torch.rand(frame_shape)
        tensor = self.visual_encoder(tensor.to("cuda"))
        x_v = torch.rand(frame_shape).to("cuda")

        with torch.no_grad():
            x_v = self.visual_encoder(x_v)

        print(f'\nXV SHAPE {x_v.shape}')


        ############### STFT ENCODER #################

        modules = []
        in_ch = 2
        spatial_dim = stft_shape[-1]
        temporal_dim = stft_shape[-2]
        output_shape = [temporal_dim, spatial_dim]

        while output_shape != list(x_v.shape[2:]):
            out_ch = in_ch * 2
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
                                    kernel_size=(3, 9), 
                                    stride=tuple(stride), 
                                    padding=(1,4),
                                    bias=False))
            modules.append(nn.BatchNorm2d(out_ch))
            modules.append(nn.Tanh())
            in_ch = out_ch

        self.stft_encoder = nn.Sequential(*modules).to("cuda")
        
        x_a = torch.rand(stft_shape).to("cuda")
        with torch.no_grad():
            x_a = self.stft_encoder(x_a)
        
        summary(self.stft_encoder, 
                input_size=(stft_shape[1], stft_shape[2], stft_shape[3]))
        ################ AV FUSION NET #################
    
        out_ch = fc_size // (output_shape[0] * output_shape[1])
        # our spatial dim must be low enough to accomadate fc size

        fc2_out = 512

        # re-arrange dimensions so that time dim is properly arranged for lstm
        x_av_cat = torch.cat((x_v.permute(0, 2, 1, 3), x_a.permute(0, 2, 1, 3)), dim=2)
        print(f'x_v {x_v.shape} x_a {x_a.shape} permuted {x_v.permute(0, 2, 1, 3).shape}')
        print(f'x_av_cat {x_av_cat.shape}')

        x_av_cat = torch.flatten(x_av_cat, start_dim=-2, end_dim=-1)
        print(f'x_av_cat flat {x_av_cat.shape}')

        # conisder making bidirectional
        self.lstm = nn.LSTM(input_size=x_av_cat.shape[-1], hidden_size=256, num_layers=1, 
                            bias=False, batch_first=True, dropout=0, bidirectional=True).to("cuda")

        # self.lstm.flatten_parameters()
        av = self.lstm(x_av_cat)[0]
        print(f'AV LSTM SH {av.shape}')
        av = torch.flatten(av, start_dim=1)
        print(f'AV FLATTENED SH {av.shape} fc size {fc_size}')
        fc_size = av.shape[-1]
        self.fc1 = nn.Linear(fc_size, fc_size//2, bias=False).to("cuda")
        av = self.fc1(av)
        print(f'AV Lin1 SH {av.shape}')
        self.fc2 = nn.Linear(fc_size//2, fc2_out, bias=False).to("cuda")
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
            out_ch = in_ch // 2
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
                                    kernel_size=(3, 9), 
                                    stride=tuple(stride), 
                                    padding=(1,4),
                                    output_padding=tuple(out_padding),
                                    bias=False))
            if encoded_shape != [temporal_dim, spatial_dim]:
                modules.append(nn.BatchNorm2d(out_ch))
                modules.append(nn.Tanh())
            in_ch = out_ch

        self.stft_decoder = nn.Sequential(*modules)

        ############### STFT AUTOENCODER #################

        self.stft_autoencoder = nn.Sequential(
            *self.stft_encoder,
            *self.stft_decoder
        ).to("cuda")

        # self.a_fc1 = nn.Linear(fc2_out, stft_shape[-2] * stft_shape[-1])
        self.a_fc1 = nn.Sequential(
          nn.Linear(fc2_out, 2 * self.output_stft_frames * stft_shape[-1], bias=False),
          # nn.LeakyReLU(negative_slope=0.3)
          # nn.Tanh()
        )

        self.v_fc1 = nn.Sequential(
          nn.Linear(fc2_out, self.frame_channels * frame_shape[-2] * frame_shape[-1], bias=False),
          # nn.LeakyReLU(negative_slope=0.3)
          # nn.Tanh()
        )

    # enable these grads for training the fusion network
    def toggle_fusion_grads(self, toggle):
        self.lstm.requires_grad_(toggle)
        self.fc1.requires_grad_(toggle)
        self.fc2.requires_grad_(toggle)
        self.a_fc1.requires_grad_(toggle)
        self.v_fc1.requires_grad_(toggle)

    def toggle_stft_ae_grads(self, toggle):
        for param in self.stft_autoencoder:
            param.requires_grad_(toggle)

    # disable encoder grads for training the av fusion
    def toggle_enc_grads(self, toggle):
        for param in self.stft_encoder:
            param.requires_grad_(toggle)
        for param in self.visual_encoder:
            param.requires_grad_(toggle)


    def av_fusion_forward(self, x_a, x_v):
            # re-arrange dimensions so that time dim is properly arranged for lstm
            x_v = x_v.permute(0, 2, 1, 3)
            x_a = x_a.permute(0, 2, 1, 3)
            print(f'x_v shape {x_v.shape} x_a {x_a.shape}')
            x_av_cat = torch.cat((x_v, x_a), dim=2)
            x_av_cat = torch.flatten(x_av_cat, start_dim=-2, end_dim=-1)
            av = self.lstm(x_av_cat)[0]
            av = torch.flatten(av, start_dim=1)
            av = self.fc1(av)
            av = F.leaky_relu(av, negative_slope=0.3)
            av = self.fc2(av)
            av = F.leaky_relu(av, negative_slope=0.3)
            return av


    def audio_ae_forward(self, x_a):
        x_a = self.stft_autoencoder(x_a)
        return x_a

    def forward(self, x_a, x_v):
        x_a_enc = self.stft_encoder(x_a)
        x_v_enc = self.visual_encoder(x_v)

        x_av_fused = self.av_fusion_forward(x_a_enc, x_v_enc)

        x_a_out = self.a_fc1(x_av_fused)
        x_a_out = F.tanh(x_a_out)

        x_v_out = self.v_fc1(x_av_fused)
        x_v_out = F.leaky_relu(x_v_out, negative_slope=0.3)

        x_v_out = x_v_out.view(x_v.shape[0], self.frame_channels, self.frame_shape[-2], self.frame_shape[-1])
        x_a_out = x_a_out.view(x_a.shape[0], 2, self.output_stft_frames, x_a.shape[-1])

        return x_a_out, x_v_out, x_av_fused