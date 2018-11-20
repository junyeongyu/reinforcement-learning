import gym
import numpy as np
import random
import pandas
import math
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

episodes = 1000
batch_size = 32

if __name__ == "__main__":
    n_features = 210 * 160 * 3;
    env = gym.make('Breakout-v0')
    observation_space = env.observation_space    
    agent = DQNAgent(n_features, env.action_space.n)

    for i_episode in range(20):
        state = env.reset()
        state = np.reshape(state, [1, n_features])
        for t in range(1000):
            env.render()        
            #action = env.action_space.sample()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, n_features]);
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # print (action)
            #observation, reward, done, info = env.step(action)
            #print(observation)
            #print(observation.shape) # (210, 160, 3)
            #print(observation.shape[0]) # 210
            print(t)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
    env.close()