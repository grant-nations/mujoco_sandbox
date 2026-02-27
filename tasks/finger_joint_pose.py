from myosuite.utils import gym
policy = "iterations/best_policy.pickle"

import pickle
pi = pickle.load(open(policy, 'rb'))

env = gym.make('myoElbowPose1D6MRandom-v0')
env.reset()
for _ in range(1000):
    env.mj_render()
    env.step(env.action_space.sample()) # take a random action