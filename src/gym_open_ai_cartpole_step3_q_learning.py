#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        print(type(state))
        print(action)
        print(self.q)
        return self.q.get((state.tostring(), action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state.tostring(), action), None)
        if oldv is None:
            self.q[(state.tostring(), action)] = reward
        else:
            self.q[(state.tostring(), action)] = oldv + self.alpha * (value - oldv)

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
