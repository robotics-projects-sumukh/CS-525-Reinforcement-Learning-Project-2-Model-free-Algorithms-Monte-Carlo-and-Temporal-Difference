#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:11:22 2019

@author: huiminren
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
"""
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.

    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.
'''
#-------------------------------------------------------------------------


def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and hit otherwise

    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # Check if the observation is a nested tuple
    if isinstance(observation[0], tuple):
        observation = observation[0]
    
    player_score, dealer_score, usable_ace = observation

    # Policy logic: Stick if player_score >= 20, otherwise Hit
    if player_score < 20:
        action = 1  # HIT
    else:
        action = 0  # STICK
    #                          #
    ############################
    return action


def mc_prediction(policy, env, n_episodes, gamma=1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################
    # YOUR IMPLEMENTATION HERE #
    while n_episodes > 0:
        n_episodes -= 1
        state = env.reset()
        episode = []
        while True:
            action = policy(state)
            next_state, reward, done, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            if done or truncated:
                break
            state = next_state 

        G = 0
        visited_states = set()
        for state, action, reward in reversed(episode):
            if isinstance(state[0], tuple):
                state = state[0]
            G = gamma * G + reward
            if state not in visited_states:
                visited_states.add(state)
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]
    # print(len(V))
    # print("Some States:", list(V.keys())[:10])
    #                          #
    ############################

    return V


def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: 
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
    ------
    With probability (1 - epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # make action an integer
    if isinstance(state[0], tuple):
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


def mc_control_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-0.1/n_episode during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(lambda: defaultdict(float))
    returns_count = defaultdict(lambda: defaultdict(float))
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # YOUR IMPLEMENTATION HERE #
    for i in range(n_episodes):
        epsilon = epsilon - 0.1 / n_episodes
        state = tuple(env.reset()) # Ensure state is hashable (convert to tuple)
        episode = []
        while True:
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            if done or truncated:
                break
            state = tuple(next_state) # Ensure state is hashable (convert to tuple)
        G = 0
        visited_states_actions = set()
        for state, action, reward in reversed(episode):
            if isinstance(state[0], tuple):
                state = state[0]
            G = gamma * G + reward
            if (state, action) not in visited_states_actions:
                visited_states_actions.add((state, action))
                returns_sum[state][action] += G
                returns_count[state][action] += 1
                Q[state][action] = returns_sum[state][action] / returns_count[state][action]
    # print(len(Q))    
    #                          #
    ############################

    return Q
