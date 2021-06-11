# import avse_model
from avse_model import AVSE_Model
from train import TrainLoop
import torch
from generator import DataGenerator
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


learning_rate = 1e-5
batch_size = 4
num_vid_frames = 4
epochs = 100

loss_coefficient = 0.001 # loss = a_loss + loss_coefficient * v_loss


dg = DataGenerator(
    batch_size = batch_size,
    num_vid_frames=num_vid_frames, 
    framerate=30,
    framesize=256,
    samplerate=16000, 
    max_vid_frames=100,
    noise_std=0.01,
    center_fft=True, 
    use_polar=False, 
    shuffle_files=True, 
    data_path = "data/processed"
)

gen = dg.generator()
x_example, y_example = next(gen)

print(f"\n x examp sh {x_example[0].shape} {x_example[1].shape} \n")


a_shape = x_example[0].shape
v_shape = x_example[1].shape

# out_1_shape = y_example[0][0].shape
# out_2_shape = y_example[1][0].shape

# model = avse_model.build_model(in_1_shape, in_2_shape, out_1_shape, out_2_shape)
# model.summary()

model = AVSE_Model(a_shape, v_shape).to(DEVICE)
print(model)

mse_loss = torch.nn.MSELoss()
cosine_loss = torch.nn.CosineSimilarity()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
losses = []

for i in range(epochs):
    optimizer.zero_grad()

    x, y = next(gen)

    yh_a, yh_v = model(x[0], x[1])

    a_loss = cosine_loss(yh_a, y[0]).sum()
    v_loss = mse_loss(yh_v, y[1])

    # loss = a_loss + loss_coefficient * v_loss
    loss = a_loss + v_loss

    loss.backward()

    print(f"step:{i} loss: {loss} a_loss:{a_loss} v_loss:{v_loss}")
    # a_loss.backward(retain_graph=True)
    # v_loss.backward()

    optimizer.step()

    print(f"step:{i} a_loss:{a_loss} v_loss:{v_loss}")


# trainLoop = TrainLoop(optimizer=Adam(learning_rate=learning_rate))

# for i in range(epochs):
#     print(f"EPOCH {i}")
#     trainLoop.train_step(gen, model)
