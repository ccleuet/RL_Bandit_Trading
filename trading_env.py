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

  #Siemens="FSE/SIE_X"
  #Volkswagen="FSE/VOW3_X"
  #Continental="FSE/CON_X"
  #Hugo_Boss="FSE/BOSS_X"
  #Daimler="FSE/DAI_X"

  def __init__(self, days=252, name=Name, scale=True ):

    self.days = days + 1

    print "==== Frankfurt Stock Exchange ===="
    print ""
    print "getting data for Siemens from Quandl..."
    dSiemens = quandl.get("FSE/SIE_X")
    print "got data for Siemens from Quandl"
    print ""
    print "getting data for Volkswagen from Quandl..."
   # dVolkswagen = quandl.get("FSE/VOW3_X")
    dVolkswagen = dSiemens
    print "got data for Volkswagen from Quandl"
    print ""
    print "getting data for Hugo Boss from Quandl..."
   # dHugo_Boss= quandl.get("FSE/BOSS_X")
    dHugo_Boss= dSiemens
    print "got data for Hugo Boss from Quandl"

    # we calculate returns and percentiles, then kill nans

    df1 = dSiemens[['Close','Traded Volume','High','Low']]   
    df2 = dVolkswagen[['Close','Traded Volume','High','Low']]   
    df3 = dHugo_Boss[['Close','Traded Volume','High','Low']]   

    df1['Traded Volume'].replace(0,1,inplace=True) # days shouldn't have zero volume..
    df2['Traded Volume'].replace(0,1,inplace=True) 
    df3['Traded Volume'].replace(0,1,inplace=True) 

    return_column = (df1['Close']-df1['Close'].shift())/df1.Close.shift() 
    df1.insert(loc=2,column='Return',value=return_column) #Price evolution in Percent
    return_column = (df2['Close']-df2['Close'].shift())/df2.Close.shift() 
    df2.insert(loc=2,column='Return',value=return_column) #Price evolution in Percent
    return_column = (df3['Close']-df3['Close'].shift())/df3.Close.shift() 
    df3.insert(loc=2,column='Return',value=return_column) #Price evolution in Percent

    pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]

    df1['Close Percentile rank'] = df1['Close'].expanding(self.MinPercentileDays).apply(pctrank)
    df1['Volume Percentile rank'] = df1['Traded Volume'].expanding(self.MinPercentileDays).apply(pctrank)
    df2['Close Percentile rank'] = df2['Close'].expanding(self.MinPercentileDays).apply(pctrank)
    df2['Volume Percentile rank'] = df2['Traded Volume'].expanding(self.MinPercentileDays).apply(pctrank)
    df3['Close Percentile rank'] = df3['Close'].expanding(self.MinPercentileDays).apply(pctrank)
    df3['Volume Percentile rank'] = df3['Traded Volume'].expanding(self.MinPercentileDays).apply(pctrank)

    df1.dropna(axis=0,inplace=True) #Drop columns with Nan elements
    df2.dropna(axis=0,inplace=True) 
    df3.dropna(axis=0,inplace=True) 

    self.min_values = df1.min(axis=0)
    self.max_values = df1.max(axis=0)
    self.min_values = df2.min(axis=0)
    self.max_values = df2.max(axis=0)
    self.min_values = df3.min(axis=0)
    self.max_values = df3.max(axis=0)

   #self.data = df1
    self.step = 0
    df=[df1,df2,df3]
    self.data = df

    print "------- Siemens -------"
    #print df1
    print "----- Volkswagen -----"
    #print df2
    print "----- Hugo_Boss -----"
    #print df3
  
  def reset(self):
    # we want contiguous data
    #self.idx = np.random.randint( low = 0, high=len(self.data.index)-self.days )
    self.idx=0;
    self.step = 0

  def _step(self): 
    obs=[]   
    obs.append(self.data[0].iloc[self.idx].as_matrix())
    obs.append(self.data[1].iloc[self.idx].as_matrix())
    obs.append(self.data[2].iloc[self.idx].as_matrix())

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
    self.actions          = np.zeros((self.steps,3))
    self.navs             = np.ones((self.steps,3))
    self.mkt_nav          = np.ones((self.steps,3))
    self.action_reward    = np.ones((self.steps,3))
    self.total_reward     = np.ones((self.steps,3))
    self.posns            = np.zeros((self.steps,3))
    self.shares           = np.ones((self.steps,3))
    self.trades           = np.zeros((self.steps,3))
    self.mkt_retrns       = np.zeros((self.steps,3))
    
  def reset(self):
    self.step = 0
    self.actions.fill(0)
    self.navs.fill(0)
    self.mkt_nav.fill(0)
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

    self.mkt_retrns[self.step,0] =retrn[0]
    self.mkt_retrns[self.step,1] =retrn[1]
    self.mkt_retrns[self.step,2] =retrn[2]

    if isinstance(action,int) == False:
      self.actions[self.step,0] = action[0]
      self.actions[self.step,1] = action[1]
      self.actions[self.step,2] = action[2]

    #Cannot sell shares if number of shares =0
    
    #if action == 0 and self.shares[self.step-1]==0 :
    #   action = 1

      self.trades[self.step,0] = action[0] -1  
      self.trades[self.step,1] = action[1] -1  
      self.trades[self.step,2] = action[2] -1  

      self.shares[self.step,0] = self.shares[self.step-1,0] + action[0] - 1
      self.shares[self.step,1] = self.shares[self.step-1,1] + action[1] - 1
      self.shares[self.step,2] = self.shares[self.step-1,2] + action[2] - 1

    reward = self.action_reward

    self.mkt_nav[self.step]=mkt_nav * (1 + self.mkt_retrns[self.step-1][0])

    self.navs[self.step] =  self.shares[self.step]* self.mkt_nav[self.step]
    self.action_reward[self.step] = self.mkt_retrns[self.step][0]*(action-1)
    self.total_reward[self.step] = self.total_reward[self.step-1]+self.action_reward[self.step]
    
    info = { 'reward': reward, 'nav':self.navs[self.step]}

    self.step += 1      
    return reward, info

  def to_df(self):
    cols = ['action','trade','total_shares','nav','mkt_nav','mkt_return','action_reward','total_reward']
    df = pd.DataFrame( {
                          'action':     self.actions[:,0], # today's action (from agent)
                          'trade':  self.trades[:,0],
                          'total_shares': self.shares[:,0],
                          'nav':    self.navs[:,0],    #  Net Asset Value (NAV)
                          'mkt_nav':  self.mkt_nav[:,0], 
                          'mkt_return': self.mkt_retrns[:,0],
                          'action_reward': self.action_reward[:,0],
                          'total_reward': self.total_reward[:,0],
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
#    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
    observation, done = self.src._step()

    yret = []
    yret.append(observation[0][2])
    yret.append(observation[1][2])
    yret.append(observation[2][2])

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
      action = np.random.randint(3,size = 3)
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
