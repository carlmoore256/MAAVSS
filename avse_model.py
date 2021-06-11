# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv1D, SeparableConv1D, Conv3D, Reshape, Conv2D
# from tensorflow.keras.layers import concatenate, Dense, AveragePooling1D, AveragePooling3D, UpSampling3D
import torch.nn as nn
import torch.nn.functional as F


class AVSE_Model(nn.Module):
    def __init__(self):
        super(AVSE_Model, self).__init__()
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=18, kernel_size=3, stride=1)

        self.audio_net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.AvgPool1d(2),
            nn.Conv1d(10, 4, kernel_size=5, stride=1),
            nn.ReLU()
        )

        self.visual_net = nn.Sequential(
            nn.Conv3d(2, 12, kernel_size=9, stride=(1,2,2), padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv3d(12, 6, kernel_size=9, stride=(2,2,2), padding_mode="replicate"),
            nn.ReLU()
        )

    def forward(self, x_a, x_v):
        print(f"xa sh {x_a.shape}")
        print(f"xv sh {x_v.shape}")
        y_a = self.audio_net(x_a)
        y_v = self.visual_net(x_v)
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
