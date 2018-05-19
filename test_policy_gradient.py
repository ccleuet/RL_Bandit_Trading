import pandas as pd
import numpy as np
import gym
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
import pdb
import logging

log = logging.getLogger()

import policy_gradient 

# create gym
env = gym.make('trading-v0')

sess = tf.InteractiveSession()

# create policygradient
pg = policy_gradient.PolicyGradient(sess, obs_dim=5, num_actions=3, learning_rate=1e-2 )

# train model
alldf,summrzed = pg.train_model( env,episodes=401, log_freq=100)

#print alldf
pd.DataFrame(summrzed).expanding().mean().plot()
input("Press Enter to continue...")
