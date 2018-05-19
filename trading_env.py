import gym
from gym import error, spaces, utils
from gym.utils import seeding
from collections import Counter

import quandl
import numpy as np
from numpy import random
import pandas as pd
import logging
import pdb

import tempfile

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.info('%s logger started.',__name__)

class QuandlEnvSrc(object):
  ''' 
  Get Trading Data Source from Quandl
  '''
  MinPercentileDays = 4700 #Min traded volume

  Name = "FSE/BOSS_X" # https://www.quandl.com/search (use 'Free' filter)

  Siemens="FSE/SIE_X"
  Volkswagen="FSE/VOW3_X"
  Continental="FSE/CON_X"
  Hugo_Boss="FSE/BOSS_X"
  Daimler="FSE/DAI_X"

  def __init__(self, days=252, name=Name, scale=True ):
    self.name = name
    self.days = days+1
    log.info('getting data for %s from Quandl...',QuandlEnvSrc.Name)
    df = quandl.get(self.name)
    log.info('got data for %s from Quandl...',QuandlEnvSrc.Name)
    
    # we calculate returns and percentiles, then kill nans
    df = df[['Close','Traded Volume','High','Low']]   
    df['Traded Volume'].replace(0,1,inplace=True) # days shouldn't have zero volume..

    return_column = (df['Close']-df['Close'].shift())/df.Close.shift() 
    df.insert(loc=2,column='Return',value=return_column) #Price evolution in Percent

    pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    df['Close Percentile rank'] = df['Close'].expanding(self.MinPercentileDays).apply(pctrank)
    df['Volume Percentile rank'] = df['Traded Volume'].expanding(self.MinPercentileDays).apply(pctrank)

    df.dropna(axis=0,inplace=True) #Drop columns with Nan elements

    self.min_values = df.min(axis=0)
    self.max_values = df.max(axis=0)
    self.data = df
    self.step = 0
    print df
    
  def reset(self):
    # we want contiguous data
    #self.idx = np.random.randint( low = 0, high=len(self.data.index)-self.days )
    self.idx=0;
    self.step = 0

  def _step(self):    
    obs = self.data.iloc[self.idx].as_matrix()
    self.idx += 1
    self.step += 1
    done = self.step >= self.days
    return obs,done

class TradingSim(object) :
  """ Implements core trading simulator for single-instrument univ """

  def __init__(self, steps):

    # invariant for object life
    self.steps            = steps

    # change every step
    self.step             = 0
    self.actions          = np.zeros(self.steps)
    self.navs             = np.ones(self.steps)
    self.mkt_nav          = np.ones(self.steps)
    self.action_reward    = np.ones(self.steps)
    self.total_reward     = np.ones(self.steps)
    self.posns            = np.zeros(self.steps)
    self.shares           = np.ones(self.steps)
    self.trades           = np.zeros(self.steps)
    self.mkt_retrns       = np.zeros(self.steps)
    
  def reset(self):
    self.step = 0
    self.actions.fill(0)
    self.navs.fill(1)
    self.mkt_nav.fill(1)
    self.action_reward.fill(0)
    self.total_reward.fill(0)
    self.posns.fill(0)
    self.shares.fill(1)
    self.trades.fill(0)
    self.mkt_retrns.fill(0)
    
  def _step(self, action, retrn ):
    """ Given an action and return for prior period, navs,
        etc and returns the reward and a  summary of the day's activity. """
    mkt_nav  = 1.0 if self.step == 0 else self.mkt_nav[self.step-1]

    self.mkt_retrns[self.step] = retrn
    self.actions[self.step] = action
    
    self.trades[self.step] = action -1   
    self.shares[self.step] = self.shares[self.step-1] + action - 1
    
    reward = self.action_reward

    self.mkt_nav[self.step] =  mkt_nav * (1 + self.mkt_retrns[self.step-1])
    self.navs[self.step] =  self.shares[self.step]* self.mkt_nav[self.step]
#   self.action_reward[self.step] = self.navs[self.step]-self.navs[self.step-1]
    self.action_reward[self.step] = self.mkt_retrns[self.step]*((action - 1) + self.shares[self.step-1])
    self.total_reward[self.step] = self.total_reward[self.step-1]+self.action_reward[self.step]
    
    info = { 'reward': reward, 'nav':self.navs[self.step]}

    self.step += 1      
    return reward, info

  def to_df(self):
    cols = ['action','trade','total_shares','nav','mkt_nav','mkt_return','action_reward','total_reward']
    df = pd.DataFrame( {
                          'action':     self.actions, # today's action (from agent)
                          'trade':  self.trades,
                          'total_shares': self.shares,
                          'nav':    self.navs,    #  Net Asset Value (NAV)
                          'mkt_nav':  self.mkt_nav, 
                          'mkt_return': self.mkt_retrns,
                          'action_reward': self.action_reward,
                          'total_reward': self.total_reward,
                       },# eod trade
                         columns=cols)
    return df

class TradingEnv(gym.Env):
  """This gym implements a simple trading environment for reinforcement learning.

  The gym provides daily observations based on real market data pulled
  from Quandl on, by default, the SPY etf. An episode is defined as 252
  contiguous days sampled from the overall dataset. Each day is one
  'step' within the gym and for each step, the algo has a choice:

  SHORT (0)
  FLAT (1)
  LONG (2)

  At the beginning of your episode, you are allocated 1 unit of
  cash. This is your starting Net Asset Value (NAV). If your NAV drops
  to 0, your episode is over and you lose. If your NAV hits 2.0, then
  you win.

  The trading env will track a buy-and-hold strategy which will act as
  the benchmark for the game.

  """
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.days = 10
    self.src = QuandlEnvSrc(days=self.days)
    self.sim = TradingSim(steps=self.days)
    self.action_space = spaces.Discrete( 3 )
    self.observation_space= spaces.Box( self.src.min_values, self.src.max_values)
    self._reset()

  def _configure(self, display=None):
    self.display = display

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _step(self, action):
    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
    observation, done = self.src._step()
    yret = observation[2]
    reward, info = self.sim._step( action, yret )

    return observation, reward, done, info
  
  def _reset(self):
    self.src.reset()
    self.sim.reset()
    return self.src._step()[0]
    
  # some convenience functions:
  
  def run_strat(self,  strategy, return_df=True):
    """run provided strategy, returns dataframe with all steps"""
    observation = self.reset()
    done = False
    while not done:
      action = strategy( observation, self ) # call strategy
      observation, reward, done, info = self.step(action)
    return self.sim.to_df() if return_df else None

  def run_strat_test(self,  strategy, return_df=True):
    """run provided strategy, returns dataframe with all steps"""
    observation = self.reset()
    done = False
    while not done:
      action = np.random.randint( low = 0, high=3)
      observation, reward, done, info = self.step(action)
    return self.sim.to_df() if return_df else None
      
  def run_strats( self, strategy, episodes=1, write_log=True, return_df=True):
    """ run provided strategy the specified # of times, possibly
        writing a log and possibly returning a dataframe summarizing activity.
    
        Note that writing the log is expensive and returning the df is moreso.  
        For training purposes, you might not want to set both.
    """
    logfile = None
    if write_log:
      logfile = tempfile.NamedTemporaryFile(delete=False)
      log.info('writing log to %s',logfile.name)
      need_df = write_log or return_df

    alldf = None
        
    for i in range(episodes):
      df = self.run_strat(strategy, return_df=need_df)
      if write_log:
        df.to_csv(logfile, mode='a')
        if return_df:
          alldf = df if alldf is None else pd.concat([alldf,df], axis=0)
            
    return alldf
