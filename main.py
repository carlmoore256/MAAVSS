import avse_model
from train import TrainLoop
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from generator import DataGenerator

learning_rate = 1e-5
batch_size = 4
num_vid_frames = 4
epochs = 100


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

print(f"x examp sh {x_example[0].shape} {x_example[1].shape}")

in_1_shape = x_example[0][0].shape
in_2_shape = x_example[1][0].shape

out_1_shape = y_example[0][0].shape
out_2_shape = y_example[1][0].shape

model = avse_model.build_model(in_1_shape, in_2_shape, out_1_shape, out_2_shape)
model.summary()

trainLoop = TrainLoop(optimizer=Adam(learning_rate=learning_rate))

for i in range(epochs):
    print(f"EPOCH {i}")
    trainLoop.train_step(gen, model)
