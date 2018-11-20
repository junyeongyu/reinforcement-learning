import gym

env = gym.make('Breakout-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        # action = action % 2
        print (action)
        #action = 0 # start/stop program (?)
        #action = 1 # no move
        #action = 2 # right
        #action = 3 # left
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        
env.close()