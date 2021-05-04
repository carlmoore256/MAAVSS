import avse_model
import train
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

learning_rate = 1e-5
batch_size = 32
num_vid_frames = 4
epochs = 10000

dg = DataGenerator(
    batch_size = batch_size,
    num_vid_frames=num_vid_frames, 
    framerate=30,
    samplerate=16000, 
    max_vid_frames=100,
    noise_std=0.01,
    center_fft=True, 
    use_polar=True, 
    shuffle_files=True, 
    data_path = "/content/drive/MyDrive/MagPhaseLAVSE/processed"
)

gen = dg.generator()

model = avse_model.build_model()
model.summary()


optimizer = Adam(learning_rate=learning_rate)

for i in range(epochs):
    train.train_step(gen, optimizer)
