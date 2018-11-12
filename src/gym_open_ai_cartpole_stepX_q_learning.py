# -*- coding: utf-8 -*-

# https://en.wikipedia.org/wiki/Q-learning
# http://mnemstudio.org/path-finding-q-learning-tutorial.htm
# Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
# Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
# Q(1, 5) = R(1, 5) + 0.8 * Max[Q(1, 2), Q(1, 5)] = 0 + 0.8 * Max(0, 100) = 80
# Q(1, 5) = R(1, 5) + 0.8 * Max[Q(5, 1), Q(5, 4), Q(5, 5)] = 100 + 0.8 * 0 = 100
