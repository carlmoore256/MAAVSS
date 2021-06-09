import tensorflow as tf
from tensorflow.keras.losses import CosineSimilarity, MeanSquaredError


class TrainLoop():

  def __init__(self, model, optimizer):
    self.video_loss = MeanSquaredError()
    self.fft_loss = CosineSimilarity()
    self.optimizer = optimizer
    self.model = model

  def grad(self, model, inputs, targets):
    with tf.GradientTape() as tape:
      ya, yv = self.model(inputs, training=True)

      a_loss = self.fft_loss(ya, targets[0])
      v_loss = self.video_loss(yv, targets[1])

      losses = [a_loss, v_loss]

    return losses, tape.gradient(losses, self.model.trainable_variables)

  def train_step(self, gen):
      x, y = next(gen)
      losses, grads = self.grad(self.model, x, y)
      self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
      print(f'fft loss {losses[0]} video loss {losses[1]}')