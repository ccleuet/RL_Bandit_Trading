import gym

import gym_trading
import pandas as pd
import numpy as np
import trading_env as te
import matplotlib.pyplot as plt

pd.set_option('display.width',400)

env = gym.make('trading-v0')

Episodes=1

obs = []

# for _ in range(Episodes):
#     observation = env.reset()
#     done = False
#     count = 0
#     while not done:
#         action = env.action_space.sample() # random
#         observation, reward, done, info = env.step(action)
#         obs = obs + [observation]
#         count += 1

# df = env.env.sim.to_df()

#buyhold=lambda x,y:2
#df = env.env.run_strat(buyhold)

#df = env.env.epsilon_greedy( )
dfall = env.env.run(10)
print dfall

plt.plot(dfall);plt.title('Episode Total Reward');plt.grid(True);plt.savefig("test.png");plt.show();

print "=================="
print "===== RESULTS ===="
print "=================="
print "===== Siemens ===="
#print df[0]
print "===== Volkswagen ===="
#print df[1]
print "===== Hugo Boss ===="
#print df[2]
#df10 = env.env.run_strats( buyhold, Episodes )
#print df10