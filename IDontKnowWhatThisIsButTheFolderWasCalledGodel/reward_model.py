import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_probability as tfp

class Reward_Model(object):
    def __init__(self, state_size):
        self.state_size = state_size
        self.model = self.build_model()

    def build_model(self):
        inputx = L.Input(shape=(self.state_size,))
        x = inputx

        for i in range(30):
            x = tfp.layers.DenseFlipout(32)(x)

        x = tfp.layers.DenseFlipout(1)(x)

        model = tf.keras.Model(inputx, x)
        model.compile(loss='mse', optimizer='adam')

        return model

    def predict(self, state):
        reward = self.model.predict(state)
        return reward

    def fit(self, state, reward):
        self.model.fit(state, reward, verbose=0, epochs=5)
