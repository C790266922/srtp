import talib

# 初始化函数
def init(context):
    context.s1 = "CSI300.INDX"
    context.SHORTPERIOD = 5
    context.LONGPERIOD = 20

def handle_bar(context, bar_dict):
    prices = history_bars(context.s1, context.OBSERVATION, '1d', 'close')

    fast_ma = talib.SMA(prices, context.SHORTPERIOD)
    slow_ma = talib.SMA(prices, context.LONGPERIOD)

    if fast_ma[-1] - slow_ma[-1] > 0 and fast_ma[-2] - slow_ma[-2] < 0:
        # 满仓入股
        order_target_percent(context.s1, 1)

    if fast_ma[-1] - slow_ma[-1] < 0 and fast_ma[-2] - slow_ma[-2] > 0:
        # 获取该股票的仓位
        curPosition = context.portfolio.positions[context.s1].quantity
        # 清仓
        if curPosition > 0:
            order_target_value(context.s1, 0)

