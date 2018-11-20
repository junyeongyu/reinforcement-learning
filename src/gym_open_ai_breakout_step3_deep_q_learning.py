from dqn_agent import DQNAgent

import gym
import numpy as np

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