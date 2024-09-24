#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v td_test.py' in the terminal.
'''
#-------------------------------------------------------------------------

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project2-1
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    if isinstance(state, tuple):
        # If state is a tuple, use the first element
        state = state[0]
    # Choose action based on epsilon-greedy policy
    if random.random() < epsilon:
        action = random.choice(range(nA))
    else:
        action = np.argmax(Q[state])

    action = int(action)
    #                          #
    ############################
    return action

def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    '''20 points'''
    """On-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a. 
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """
    
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    while n_episodes > 0:
        state = env.reset()
        # print(state)
        action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
        while True:
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, env.action_space.n, epsilon)
            # below if functions are used to handle the case where state is a tuple
            if isinstance(state, tuple):
                state_ = state[0]
            else:
                state_ = state
            next_state_ = state_
            if isinstance(next_state, tuple):
                next_state_ = next_state[0]
            else:
                next_state_ = next_state
            ############################
            Q[state_][action] = Q[state_][action] + alpha * (reward + gamma * Q[next_state_][next_action] - Q[state_][action])
            state = next_state
            action = next_action
            if done or truncated:
                break
        n_episodes -= 1
        epsilon = 0.99 * epsilon
    #                          #
    ############################
    return Q

def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    '''20 points'''
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    while n_episodes > 0:
        state = env.reset()
        while True:
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            # below if functions are used to handle the case where state is a tuple
            if isinstance(state, tuple):
                state_ = state[0]
            else:
                state_ = state
            next_state_ = state_
            if isinstance(next_state, tuple):
                next_state_ = next_state[0]
            else:
                next_state_ = next_state
            ############################
            Q[state_][action] = Q[state_][action] + alpha * (reward + gamma * np.max(Q[next_state_]) - Q[state_][action])
            state = next_state
            if done or truncated:
                break
        n_episodes -= 1
        epsilon = 0.99 * epsilon
    #                          #
    ############################
    return Q
