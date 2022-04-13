#!/usr/bin/env python
'''
##
# Vanilla Policy Gradient in Jax / Flax for the Discrete Lunar Lander.
##
##############################################################
# Author:               Nicholas Ho
# Email:                
# Affiliation:        
# Date Created:               
##############################################################
# Usage:
##############################################################
# Notes: 
##############################################################
# References:
# https://github.com/pytorch/examples/blob/main/reinforcement_learning/ac.py 
##############################################################
'''

import argparse
import gym
import numpy as np
from itertools import count

from typing import Any, Callable, Sequence, Optional
import jax
import jax.numpy as jnp                # JAX NumPy
from jax import random 
from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers


import jax.numpy as jnp
import jax
from jax import jit
from jax import grad
from jax import vmap
from jax import value_and_grad



#NOTE ONLY FOR NICKS MAC CUZ IT KINDA SUCKS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


gamma = 0.99
seed = 543
render = False
log_interval = 10
eps = np.finfo(np.float32).eps.item() ## epsilon value


# env = gym.make('CartPole-v0')
# env = gym.make("LunarLanderContinuous-v2")
env = gym.make("LunarLander-v2")
state = env.reset()


class Policy(nn.Module):
    def setup(self):

        self.actor1 = nn.Dense(128)
        self.actor2 = nn.Dense(4)
        
        self.critic1 = nn.Dense(128)
        self.critic2 = nn.Dense(1)
    
    def __call__(self, inputs):
        
        x = inputs
        x = self.actor1(x)
        x = nn.relu(x)
        x = self.actor2(x)
        action_log_probs = nn.log_softmax(x) ## might be wrong ://
        
        ### critic
        x = self.critic1(inputs)
        x = nn.relu(x)
        state_values = self.critic2(x)
        
        
        return action_log_probs, state_values
    



"""
Select Action
"""

@jax.jit
def select_action(state, params, key):
    state = jnp.array(state)
    action_log_probs, value = Policy().apply(params, state) ## feeds it into the policy

    ##sampling action
    # action = jax.random.normal(key, (2,)) * std + mu ## no sigma for now
    # action = jnp.array(jax.random.bernoulli(key, p), dtype = int)
    action = jax.random.categorical(key, action_log_probs)
    
    key1, key2 = random.split(key, 2)
    return action, value, key1 ## returns the action

"""
sample trajectory

Code is based off of Pytorch's AC.py example:
https://github.com/pytorch/examples/blob/main/reinforcement_learning/AC.py 
"""

def sample_and_process(params, key):
    
    rewards_mem = []
    actions_mem = []
    states_mem = []
    value_mem = []
    
    
    ## sampling a trajectory
    state, ep_reward = env.reset(), 0 ## restatarting the states
    for t in range(0, 1000):  # Don't infinite loop while learning 10000

        states_mem.append(state)
        action, value, key = select_action(state, params, key)
        # print(action)
        state, reward, done, _ = env.step(np.array(action))
        
        rewards_mem.append(reward)
        actions_mem.append(action)
        value_mem.append(value)
        
        ep_reward += reward

        if done:
            break
    
    actions_mem = jnp.array(actions_mem)
    states_mem = jnp.array(states_mem)
    value_mem = jnp.array(value_mem)

    ## Process the rewards into returns
    R = 0
    returns = [] ## initialize the array of returns
    
    ## this is creating the R(s) = R + \gamma * R(s')
    for r in rewards_mem[::-1]: ## iterate backward through the rewards
        R = r + gamma * R ## 
        returns.insert(0, R)
        
    returns = jnp.array(returns)
    
    ## normalizes the returns ## NOTE I should do this too
    returns = (returns - returns.mean()) / (returns.std() + eps)
    
    return states_mem, actions_mem, value_mem, returns, ep_reward, key


"""
initialize NN
"""
def create_train_state(rng, learning_rate):
    """Creates initial `TrainState`."""
    model = Policy()
    x = jnp.ones([1,8])
    params = model.init(rng, x)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)


"""
Apply the update based on the sampled values
"""


@jax.jit
def policy_update(model_state, states, actions, values, returns):

    ### returns the log probs
    def AC_loss(params, states, actions, values, returns):
        advantage = returns - values.reshape(-1)
        
        logits, _ = Policy().apply(params, states) ## feeds it into the policy
        one_hot_labels = jax.nn.one_hot(actions, num_classes=4) ## 4 actions
        
        critic_loss = jnp.mean( (values.reshape(-1) - returns)**2 )
        
        policy_loss = jnp.sum((one_hot_labels * -logits) * advantage.reshape(-1, 1))
        
        return critic_loss + policy_loss
    

    grad_fn = jax.value_and_grad(AC_loss)
    loss, grads = grad_fn(model_state.params, states, actions, values, returns)
    model_state = model_state.apply_gradients(grads = grads)
    
    return model_state, loss
    




"""
Training Loop
"""


key = random.PRNGKey(0)
env.seed(seed)
model_state = create_train_state(key, 1e-2)

batch_update = 10

reward_hist = []
state_pool = []
action_pool = []
return_pool = []
value_pool = []


print("Starting...")
for i in range(10000):

    more_states, more_actions, more_val, more_returns, ep_reward, key = sample_and_process(model_state.params, key)
    reward_hist.append(ep_reward)

    state_pool += more_states
    action_pool += more_actions
    return_pool += more_returns
    value_pool += more_val

    if i % batch_update == 0:
        state_pool = jnp.array(state_pool)
        action_pool = jnp.array(action_pool)
        return_pool = jnp.array(return_pool)
        value_pool = jnp.array(value_pool)

        model_state, loss = policy_update(model_state, state_pool, action_pool, value_pool, return_pool)

        state_pool = []
        action_pool = []
        return_pool = []
        value_pool = []

        avg_reward = np.mean(reward_hist[-100:])
        batch_reward = np.mean(reward_hist[-batch_update:])
        print("episode: ", i , " batch reward: ", batch_reward, " avg: ", avg_reward)

        if(batch_reward >= env.spec.reward_threshold):
            break


for i in range(10):
    state = env.reset()
    for _ in range(1000):
        env.render()
        action, _  ,key = select_action(state, model_state.params, key)
        state, reward, done, _ = env.step(np.array(action))
        if done:
            break

env.close()
