import tensorflow as tf
import tensorflow.keras.layers as L
from collections import deque
import random
import numpy as np
import tensorflow_probability as tfp

def set_random_seed():
    # tf.random.set_seed(1233)
    np.random.seed(1233)
    random.seed(1233)

set_random_seed()

def mod_sigmoid(x):
    return 1 / (1  + (1.0001 ** (x-10000)))

def mish(x):
    return tf.tanh(x) * x

def conv(x, units, kernel, stride, noise=False, padding='valid', mu_noise=0.4, alpha=1.3):
    y = L.Conv2D(units, kernel, stride, padding=padding)(x)
    z = tf.random.normal(tf.shape(y), mean=0.0, stddev=alpha)
#    y = y + mu_noise * z
    y = tf.nn.sigmoid(y)
    #    y = mL.Pact()(y)

    return y

def res_block(x, filt=[4, 8, 4], size=(5,5), stride=1):
    filt_1, filt_2, filt_3 = filt
    convo = L.Conv2D(filt_1, (1,1), stride)(x)
    convo = L.Conv2D(filt_2, size, 1, padding='same')(convo)
    convo = L.Conv2D(filt_3, (1,1), 1)(convo)

    cut = L.Conv2D(filt_3, (1,1), stride)(x)
    x = L.Add()([convo, cut])
    x = tf.nn.sigmoid(x)
#    x = mL.Pact()(x)
    return x

def forgetful_res_block_1d(x, filt=[64, 64, 64], size=(5,), stride=1):
    filt_1, filt_2, filt_3 = filt
    convo = tfp.layers.Convolution1DFlipout(filt_1, 1, stride)(x)
    convo = tfp.layers.Convolution1DFlipout(filt_2, size, 1, padding='same')(convo)
    convo = tfp.layers.Convolution1DFlipout(filt_3, 1, 1)(convo)

#    forget = L.Conv1D(filt_3, 1, stride, activation='sigmoid')(x)
    shortcut = tfp.layers.Convolution1DFlipout(filt_3, 1, stride)(x)
#    shortcut = shortcut * forget

    combined = L.Add()([shortcut, convo])
    return combined


def conv1d(x, filt, size, stride):
    return L.Conv1D(filt, size, stride)(x)

class GAN_DDPG:
    def __init__(self, state, action_size):
        self.state = state
        self.action_size = action_size
        self.gamma = .70
        self.tau = .005
        self.alpha = 1.0
        self.beta = 1.0
        self.theta = 0.9
        self.flipout = False

        self.buffer = deque(maxlen=1000)
        self.reward_buffer = deque(maxlen=1000)


        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.tape = tf.GradientTape(persistent=True)
        self.target_actor = self.actor_build()
        self.actor = self.actor_build()

        self.critic = self.critic_build()
        self.target_critic = self.critic_build()

        self.optimizer = tf.keras.optimizers.Adam(lr=0.000001)
        self.internal_clock = 0
        self.loss_critic = 0.0

        self.mcts = None #Pseudo_MCTS_Alpha(20, self.action_size, 2)

    def save_models(self):
        self.actor.save("actor.h5")
        self.critic.save("critic.h5")
        self.imagination_model.save("imagine.h5")

    def load_models(self):
        self.actor = tf.keras.models.load_model("actor.h5")
        self.critic = tf.keras.models.load_model("critic.h5")
        self.imagination_model = tf.keras.models.load_model("imagine.h5")

    def actor_build(self):
        state_input = L.Input(shape=self.state)
        x = state_input

        if len(self.state) == 3:
            for i in range(3):
                x = res_block(x)
            for i in range(3):
                x = conv(x, 16, (5,5), 3)
            for i in range(1):
                x = conv(x, 32, (5,5), 1)
        elif len(self.state) == 1:
            x = L.Reshape((self.state[0], 1))(x)
            x = forgetful_res_block_1d(x)
            x = forgetful_res_block_1d(x)
            x = conv1d(x, 4, 3, 1)

            # for i in range(30):
            #     if self.flipout:
            #         x = tfp.layers.DenseFlipout(8, activation='tanh')(x)
            #     else:
            #         x = L.Dense(32, activation='tanh')(x)
        else:
            raise ValueError

        x = L.Flatten()(x)

        # for i in range(10):
        #     x = L.Dense(32, activation='tanh')(x)
        #     z = tf.random.normal(tf.shape(x), mean=0.0, stddev=1.2)
        #     x = x + 0.1 * z
        control = L.Dense(self.action_size, activation='tanh')(x)
        model = tf.keras.Model(state_input, [control])
        model.summary()
        return model

    def actor_loss_obj(self, state, next_state, reward, action):
        with self.tape as tape:
            y_true = self.critic([state, self.actor(state)])
            loss = -y_true
        return loss, tape.gradient(loss, self.actor.trainable_variables)

    def curiosity(self, state_real, state_false):
        return tf.keras.losses.mse(state_real, state_false)

    def get_action(self, state):
        list_of_action_sources = []
        action_raw = self.actor.predict(state)
        list_of_action_sources.append(action_raw[0][0])
        if self.flipout:
            list_of_action_sources.append(self.actor.predict(state)[0])

        if self.mcts is not None and self.internal_clock >= 200 and self.internal_clock % 2 == 0:
            list_of_action_sources.append(self.mcts.action_selection(state, self.imagination_model))

        action = np.mean(list_of_action_sources, axis=0)

        self.internal_clock += 1
        return action, action_raw

    def get_target_action(self, state):
        return self.target_actor.predict(state)[0] + (self.theta * np.random.random((1, self.action_size)))

    def train_actor(self, states, next_states, reward, action):
        loss, grads = self.actor_loss_obj(states, next_states, reward, action)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

    def critic_build(self):
        action_input = L.Input(shape=(self.action_size,))
        state_input = L.Input(shape=self.state)
        y = action_input
        x = state_input
        if len(self.state) == 1:
            y = L.Dense(8)(y)
            x = L.Dense(32)(x)
            x = L.Dense(32)(x)
            x = L.Concatenate(-1)([y, x])
            x1 = L.Dense(1)(x)
        elif len(self.state) == 3:
            y = L.RepeatVector(self.state[1])(y)
            y = L.Flatten()(y)
            y = L.RepeatVector(self.state[0])(y)
            y = L.Reshape((self.state[0], self.state[1], 2))(y)
            x = L.Concatenate(-1)([y, x])

            for i in range(3):
                x = res_block(x)

            for i in range(3):
                x = conv(x, 16, (5,5), 3)
            for i in range(1):
                x = conv(x, 32, (5,5), 1)
            x = L.Flatten()(x)
            x1 = L.Dense(1)(x)
        else:
            raise ValueError

        model = tf.keras.Model([state_input,action_input], x1)
        model.summary()
        return model

    def critic_loss_obj(self, states, actions, target_input, y_true):
        with self.tape as tape:
            y_pred = self.critic([states, actions])
            target_Q = y_true + self.gamma * target_input
            mse = tf.keras.losses.mse(target_Q, y_pred)
        return mse, tape.gradient(mse, self.critic.trainable_variables)

    def train_critic(self, states, actions, reward, next_state):
        next_actions = self.get_target_action(next_state)
        targets = self.get_Q_target(next_state, next_actions)
        loss_vals, grads = self.critic_loss_obj(states, actions, targets, reward)
        self.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        return loss_vals

    def get_Q(self, states, actions):
        return self.critic.predict([states, actions])

    def get_Q_target(self, states, actions):
        return self.target_critic([states, actions])

    def soft_update(self):
        weights_crit_local = np.array(self.critic.get_weights())
        weights_crit_targ = np.array(self.target_critic.get_weights())
        self.critic.set_weights(self.tau * weights_crit_local + (1.-self.tau) * weights_crit_targ)

        weights_act_local = np.array(self.actor.get_weights())
        weights_act_targ = np.array(self.target_actor.get_weights())
        self.actor.set_weights(self.tau * weights_act_local + (1.-self.tau) * weights_act_targ)

    def remember(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
        self.reward_buffer.append(reward)

    def train(self):
        batch = random.sample(self.buffer, 1)
        batch_2 = random.sample(self.buffer, 1)
        batch_3 = [self.buffer[np.argmax(self.reward_buffer)]]
        for state, action, reward, next_state in batch:
            train_bool = bool(random.getrandbits(1))

            self.train_actor(state, next_state, reward, action)
            self.train_critic(state, action, reward, next_state)
            self.soft_update()

        for state, action, reward, next_state in batch_2:
            self.loss_critic = self.train_critic(state, action, reward, next_state)
            self.soft_update()

class Pseudo_MCTS_Alpha(object):
    def __init__(self, num_sims, action_size, max_branch_depth):
        self.num_sims = num_sims
        self.max_branch_depth = max_branch_depth
        self.action_size = action_size
    def action_selection(self, state, world_model):

        state_zero = state
        simulations = []
        simulations_rewards = []

        for i in range(self.num_sims):
            branch = []
            branch_rewards = []
            for j in range(self.max_branch_depth):
                action = np.random.normal(scale=2.0, size=(1, self.action_size)) - 1.0
                rewards, state = world_model.predict([state, action])
                branch.append(action)
                branch_rewards.append(rewards)
            simulations.append(branch)
            simulations_rewards.append(branch_rewards)

        best_rewards = -100000.0

        for branch, i in zip(simulations_rewards, range(len(simulations_rewards))):
            sum_reward_of_branch = np.sum(branch)
            if sum_reward_of_branch > best_rewards:
                index = i
                best_rewards = branch[0]

        best_branch = simulations[index]
        return best_branch[0]

if __name__ == '__main__':
    ddpg = DDPG((300,300,3))
    for i in range(50):

        state = np.random.random((1,300,300,3))
        state = np.float32(state)
        action = ddpg.get_action(state)
        actions = np.squeeze(action)
        print(actions)

        reward = np.array([[100.]])

        next_state = np.random.random((1,300,300,3))
        next_state = np.float32(next_state)


        ddpg.remember(state, action, reward, next_state)
        ddpg.train()
