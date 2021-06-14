# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv1D, SeparableConv1D, Conv3D, Reshape, Conv2D
# from tensorflow.keras.layers import concatenate, Dense, AveragePooling1D, AveragePooling3D, UpSampling3D
import torch.nn as nn
import torch.nn.functional as F
import torch

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
    def __init__(self, stft_shape, v_shape):
        super(AV_Model_STFT, self).__init__()

        self.stft_shape = stft_shape
        self.v_shape = v_shape

        print(f"\nA SHAPE {stft_shape}")
        print(f"V SHAPE {v_shape}\n")

        self.audio_net = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=10, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv1d(10, 4, kernel_size=(5,5), stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(4, 2, kernel_size=(5,5), stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
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

        self.av_fc1 = nn.Linear(3868, 512, bias=True)
        self.av_fc1_ln = nn.LayerNorm(512, elementwise_affine=True)

        # self.av_fc2 = nn.Linear(512, )

        self.a_fc_out = nn.Linear(512, stft_shape[1] * stft_shape[2] * stft_shape[3])
        self.v_fc_out = nn.Linear(512, v_shape[1] * v_shape[2] * v_shape[3] * v_shape[4])

    def forward(self, x_a, x_v):

        y_a = self.audio_net(x_a)
        y_v = self.visual_net(x_v)

        # y_v = y_v.squeeze(2) # -> squeeze time dimension?

        # y_a = y_a.unsqueeze(-1) # -> add another dimension?

        # y_v = y_v.reshape((y_v.shape[0], y_v.shape[1], y_v.shape[2], y_v.shape[3] * y_v.shape[4]))
        y_v = y_v.flatten(start_dim=-3, end_dim=-1) # <- cool way to reshape
        # y_v = y_v.squeeze(2)

        y_a = y_a.flatten(start_dim=2, end_dim=3)

        av_fc = torch.cat((y_a, y_v), -1)

        av_fc = av_fc.flatten(start_dim=1)
        
        av_fc = self.av_fc1(av_fc)
        av_fc = self.av_fc1_ln(av_fc)
        av_fc = F.leaky_relu(av_fc, negative_slope=0.3) # <- use functional here for simplicity

        y_a = self.a_fc_out(av_fc)
        y_a = torch.tanh(y_a)
        y_a = y_a.reshape(self.stft_shape)

        # reconstruction of video attention frames
        y_v = self.v_fc_out(av_fc)
        y_v = F.leaky_relu(y_v, negative_slope=0.3) 
        # reshape tensor into original dimensions
        y_v = y_v.reshape(self.v_shape)

        return y_a, y_v

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
