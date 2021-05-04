import tensorflow as tf
from tensorflow.keras.losses import CosineSimilarity, MeanSquaredError

video_loss = MeanSquaredError()
fft_loss = CosineSimilarity()


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    ya, yv = model(inputs, training=True)

    a_loss = fft_loss(ya, targets[0])
    v_loss = video_loss(yv, targets[1])

    losses = [a_loss, v_loss]

  return losses, tape.gradient(losses, model.trainable_variables)

def train_step(gen, optimizer):
    x, y = next(gen)
    losses, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f'fft loss {losses[0]} video loss {losses[1]}')