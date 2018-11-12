# CarPole

## Step 1
'''
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
'''

## Step 2
'''
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        
env.close()
'''

## Step 3
import gym
import numpy
import random
import pandas

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        print(state)
        print(action)
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))] 
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values 
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]        
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

def build_state(features):    
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]


# The Q-learn algorithm
''' (Not compatible)
qlearn = QLearn(actions=range(env.action_space.n),
                alpha=0.5, gamma=0.90, epsilon=0.1)

env = gym.make('CartPole-v0')
for i_episode in range(200):
    observation = env.reset()
    state = observation
    qlearn.epsilon = qlearn.epsilon * 0.999 # added epsilon decay
    cumulated_reward = 0
    for t in range(1000):
        env.render()
        print(observation)
        #action = env.action_space.sample()
        action = qlearn.chooseAction(state)
        print(action)
        observation, reward, done, info = env.step(action)
        nextState = observation
        qlearn.learn(state, action, reward, nextState)
        state = nextState
        cumulated_reward += reward
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    print("Episode {:d} reward score: {:0.2f}".format(i_episode, cumulated_reward))     
env.close()
'''

# Q table implementation (BUG of BUG)
'''
import numpy as np
env = gym.make('CartPole-v0')
observations = env.observation_space.shape[0]
actions = env.action_space.n
Q = np.zeros([observations, actions])
G = 0
gamma = 0.618
for episode in range(1,1001):
    done = False
    G, reward, counter = 0,0,0
    state = env.reset()
    while done != True:
            print (Q)
            print(state)
            print(np.argmax(state))
            #action = np.argmax(Q[state])
            #for i in actions:
            #    Q[]
            old_maxes = np.amax(Q, axis=0)
            old_max = np.max(old_maxes)
            new_max = np.max(state)
            print(np.amax(Q, axis=0))
            print(old_max)
            print(new_max)
            #if (old_max > new_max):
            old_max_index = np.argmax(old_maxes)
            action = np.argmax(Q[:,old_max_index])
            #else:
            #    action = m
            state2, reward, done, info = env.step(action)
            #Q[3,1] = 4
            #print (Q)
            #Q[state,action] = (reward + gamma * np.max(Q[state2]))
            Q[action, np.argmax(state)] = (reward + gamma * np.max(Q[state2]))
            #Q[state,action]
            G += reward
            counter += 1
            state = state2   
    if episode % 50 == 0:
        print('Episode {} Total Reward: {} counter: {}'.format(episode,G,counter))

ex - Q (3,1)
[[0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 4.]]
'''


'''
# initialze
import numpy as np
env = gym.make('CartPole-v0')
observations = env.observation_space.shape[0]
actions = env.action_space.n
q_table = np.zeros([observations, actions])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            print (tuple)
            action = np.argmax(q_table[tuple(state)]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")
'''