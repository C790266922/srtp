import numpy as np
import talib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def init(context):
    context.counter = 0
    context.period = 15
    context.observation = 100
    context.linreg = None
    context.s1 = 'CSI300.INDX'
    context.ma_period = 7
    
    logger.info("RunInfo: {}".format(context.run_info))
    
def norm(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)

def train(x, y, context):
    linreg = LinearRegression()
    
    linreg.fit(x, y)
    context.linreg = linreg
    
    return linreg

def predict(x, context):
    pred = context.linreg.predict(x)
    
    return pred


def handle_bar(context, bar_dict):
    closes = history_bars(context.s1, context.observation, '1d', 'close')
    high = history_bars(context.s1, context.observation, '1d', 'high')
    low = history_bars(context.s1, context.observation, '1d', 'low')
    turnover = history_bars(context.s1, context.observation, '1d', 'total_turnover')
    ma = talib.SMA(closes, context.ma_period)
    k, d = talib.STOCH(high, low, closes, fastk_period = 9, slowk_period = 3, slowk_matype = 0, slowd_period = 3)
    j = k * 3 - d * 2
    
    # features: [this_close, this_turnover, ma7, kdj_j]
    # target: next_close
    data = {
        'close': pd.Series(closes),
        'turnover': pd.Series(turnover),
        'ma': pd.Series(ma),
        'kdj_j': pd.Series(j)
    }
    
    target = {
        'close': pd.Series(closes).shift(-1)
    }
    
    x = norm(pd.DataFrame(data).dropna(axis = 0))
    y = norm(pd.DataFrame(target).dropna(axis = 0))
    
    if context.counter % 7 == 0:
        x = x[-context.period:]
        y = y[-context.period:]
        train(x, y, context)
    
    context.counter += 1
    
    pred = predict(x[-1], context)
    
    closes = norm(closes)
    
    # plot('pred', pred)
    # plot('close', closes[-1])
    
    change = (pred - closes[-1]) / closes[-1]
    if change > 0.05:
        order_target_percent(context.s1, 1)
    elif change < -0.03:
        order_target_value(context.s1, 0)
