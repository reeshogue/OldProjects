from imaginative_ddpg import GAN_DDPG
import numpy as np
from reward_model import Reward_Model as rew_mod
from collections import deque
from os import system
import time

def inverse_sigmoid(x):
    return 1. / (1 + np.exp(x))
class GodelMachine(object):
    def __init__(self):
        self.origin_pacer = GAN_DDPG((300,), 2)
        self.reward_model = rew_mod(300)
        self.run()

    def run(self):
        state = deque(maxlen=300)


        for i in range(300):
            state.append(0)

        nodems = False

        if True:
            mesa = 0
            for i in range(1000000000):

                action, action_raw = self.origin_pacer.get_action(np.expand_dims(np.float32(np.array(state)), axis=0))
                action_text = np.abs(int(action * 94)) + 32
                old_state = state

                state.append(action_text)

                reward = self.reward_model.predict(np.expand_dims(np.array(state), axis=0))
                if not nodems:
                    state_string = ''
                    for j_chrs in state:
                        state_string += chr(j_chrs)
                    system('clear')
                    print("State:", state_string)
                    human_input = '-1'
                    while True:
                        try:
                            human_input = input("Rewards: ")
                            if human_input == 'release':
                                break
                            if human_input == 'nodems':
                                nodems = True

                                break
                            human_input = float(human_input)
                            break
                        except:
                            print("Input must be a float. If you want to release the generated code, please enter 'release'.")
                    mesa += 1

                    if human_input == 'release':
                        with open("godel_machine.py", "w") as godel:
                            godel.write(state_string)
                            break
                    reward = np.array([[float(human_input)]])

                    self.reward_model.fit(np.expand_dims(np.array(state), axis=0), reward)

                if nodems:
                    reward = self.reward_model.predict(np.expand_dims(np.array(state), axis=0))


                print("Rewards given:", reward[0][0])
                reward *= self.determine_compile(state)
                self.origin_pacer.remember(np.expand_dims(np.float32(np.array(old_state)), axis=0), action_raw, reward,
                                           np.expand_dims(np.float32(np.array(state)), axis=0))

                self.origin_pacer.train()
    def determine_compile(self, state):
        state_string = ''
        for j_chrs in state:
            state_string += chr(j_chrs)

        try:
            a = compile(state_string)
            try:
                start_time = time.time()
                exec(a)
                time = time.time() - start_time
                return inverse_sigmoid(time)
            except:
                return .01
            return .1
        except:
            return .0005

godel = GodelMachine()
