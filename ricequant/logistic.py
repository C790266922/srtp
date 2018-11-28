import numpy as np
import talib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def init(context):
    context.counter = 0
    context.period = 14
    context.observation = 100
    context.reg = None
    context.s1 = 'CSI300.INDX'
    context.ma_period = 7
    context.high = 0
    
    logger.info("RunInfo: {}".format(context.run_info))
    
def standardlize(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)
    
def normalize(data):
    scaler = MinMaxScaler()
    scaler.fit_transform(data)
    return scaler.transform(data)

def train(x, y, context):
    reg = LogisticRegression()
    
    reg.fit(x, y)
    context.reg = reg
    
    return reg

def predict(x, context):
    pred = context.reg.predict(x)
    
    return pred
    
def predict_prob(x, context):
    prob = context.reg.predict_proba(x)
    return prob

def onehot_encode(arr):
    a = np.array(arr).copy()
    up_idx = np.where(a >= 0)
    down_idx = np.where(a < 0)
    
    up = np.zeros(len(a))
    up[up_idx] = 1
    down = np.zeros(len(a))
    down[down_idx] = 1
    
    return up, down
    
def handle_bar(context, bar_dict):
    closes = history_bars(context.s1, context.observation, '1d', 'close')
    high = history_bars(context.s1, context.observation, '1d', 'high')
    low = history_bars(context.s1, context.observation, '1d', 'low')
    turnover = history_bars(context.s1, context.observation, '1d', 'total_turnover')
    ma = talib.SMA(closes, context.ma_period)
    k, d = talib.STOCH(high, low, closes, fastk_period = 9, slowk_period = 3, slowk_matype = 0, slowd_period = 3)
    j = k * 3 - d * 2
    
    logret = pd.Series(np.log(closes)).diff().shift(-1)
    up, down = onehot_encode(logret)
    
    data = {
        'close': pd.Series(closes),
        'turnover': pd.Series(turnover),
        'ma': pd.Series(ma),
        'kdj_j': pd.Series(j),
        'logret': pd.Series(logret),
        'up': pd.Series(up),
        'down': pd.Series(down)
    }
    
    feat = {
        'high': pd.Series(high),
        'low': pd.Series(low),
        'close': pd.Series(closes),
        'turnover': pd.Series(turnover),
        'ma': pd.Series(ma),
        'kdj_j': pd.Series(j) 
    }
    
    target = {
        # 'close': pd.Series(closes).shift(-1)
        'up': data['up'],
        # 'down': data['down']
    }
    
    # feat = ['close', 'turnover', 'ma', 'kdj_j']
    
    x = normalize(pd.DataFrame(feat).dropna(axis = 0))
    y = normalize(pd.DataFrame(target).dropna(axis = 0))
    
    if context.counter % 7 == 0:
        x = x[-context.period:]
        y = y[-context.period:]
        train(x, y, context)
    
    context.counter += 1
    
    prob = predict_prob(x[-1], context)
    pred = predict(x[-1], context)
    up_prob = prob[0][0]
    
    # plot('prob', up_prob)
    # plot('pred', pred)
    
    if up_prob > 0.55:
        ret = order_target_percent(context.s1, 1)
        if ret != None:
            context.high = closes[-1]
    elif up_prob < 0.4:
        ret = order_target_value(context.s1, 0)  
        if ret != None:
            context.high = closes[-1]
    
    if closes[-1] >= context.high:
        context.high = closes[-1]
        
    if (context.high - closes[-1]) / context.high >= 0.1:
        ret = order_target_value(context.s1, 0)
        if ret != None:
            context.high = closes[-1]
    
    # plot('high', context.high)
    # plot('pred', pred)
    # plot('close', closes[-1])
