#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# http://pages.cs.wisc.edu/~finton/qcontroller.html
# https://gist.github.com/n1try/af0b8476ae4106ec098fea1dfe57f578


import gym
import numpy
import random
import pandas
import math

class QLearn:
    def __init__(self, env, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions
        self.env = env

    def getQ(self, state, action):
        #print(state)
        #print(type(state))
        #print(action)
        #print(self.q)
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
        
    def discretize(self, obs): # When observation value are not discreted
        buckets=(1, 1, 6, 12,) # down-scaling feature space to discrete range
        
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)
    

def build_state(features):    
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]

env = gym.make('CartPole-v0')
qlearn = QLearn(env=env, actions=range(env.action_space.n), alpha=0.5, gamma=0.90, epsilon=0.1)

for i_episode in range(20000):
    observation = qlearn.discretize(env.reset())
    state = observation
    qlearn.epsilon = qlearn.epsilon * 0.999 # added epsilon decay
    cumulated_reward = 0
    for t in range(1000):
        env.render()
        #print(state) # [ 0.04226653  0.00547875 -0.03172034 -0.04235962]
        #action = env.action_space.sample()
        action = qlearn.chooseAction(state)
        #print(action)
        observation, reward, done, info = env.step(action)
        nextState = qlearn.discretize(observation)
        qlearn.learn(state, action, reward, nextState)
        state = nextState
        cumulated_reward += reward
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    print("Episode {:d} reward score: {:0.2f}".format(i_episode, cumulated_reward))     
env.close()
