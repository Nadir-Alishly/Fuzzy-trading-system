import numpy as np
import pandas as pd
import pandas_ta as ta
from pandas_ta.momentum.macd import macd
pd.options.mode.chained_assignment = None
import skfuzzy as fuzz
import skfuzzy.membership as mf
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt

plt.style.use('fivethirtyeight')

ticker = 'DASH-USD'
start = dt.datetime(2010, 1, 1)
end = dt.datetime(2021, 12, 1)

def FuzzySystem(rsi, macd, adx):

        #inputs-----
        input_rsi = rsi
        input_macd = macd
        input_adx = adx

        #input scaling-----
        if(input_macd>5): input_macd = 5
        if(input_macd<-5): input_macd = -5

        #indentification of (input) variables-----
        x_rsi = np.arange(0, 101, 1)
        x_macd = np.arange(-5.1, 5.2, .1)
        x_adx = np.arange(0, 101, 1)
        x_bullish = np.arange(0, 101, 1)
        x_bearish = np.arange(0, 101, 1)

        #fuzzy subset configuartion & obtain membership functions-----
        rsi_low = mf.trimf(x_rsi, [0, 0, 30])
        rsi_med = mf.trimf(x_rsi, [20, 50, 90])
        rsi_high = mf.trimf(x_rsi, [80, 100, 100])

        macd_wait = mf.trimf(x_macd, [-5, 0, 5])
        macd_short = mf.trimf(x_macd, [-5, -5, 0])
        macd_long = mf.trimf(x_macd, [0, 5, 5])

        adx_notrend = mf.trimf(x_adx, [0, 0, 30])
        adx_ontrend = mf.trimf(x_adx, [20, 30, 60])
        adx_danger = mf.trimf(x_adx, [50, 100, 100])

        bullish_weak = mf.trimf(x_bullish, [0, 0, 10])
        bullish_strong = mf.trimf(x_bullish, [0, 30, 60])
        bullish_verystrong = mf.trimf(x_bullish, [40, 100, 100])

        bearish_weak = mf.trimf(x_bearish, [0, 0, 10])
        bearish_strong = mf.trimf(x_bearish, [0, 30, 60])
        bearish_verystrong = mf.trimf(x_bearish, [40, 100, 100])

        #get membership values of input-----
        rsi_fit_low = fuzz.interp_membership(x_rsi, rsi_low, input_rsi)
        rsi_fit_med = fuzz.interp_membership(x_rsi, rsi_med, input_rsi)
        rsi_fit_high = fuzz.interp_membership(x_rsi, rsi_high, input_rsi)

        macd_fit_wait = fuzz.interp_membership(x_macd, macd_wait, input_macd)
        macd_fit_short = fuzz.interp_membership(x_macd, macd_short, input_macd)
        macd_fit_long = fuzz.interp_membership(x_macd, macd_long, input_macd)

        adx_fit_notrend = fuzz.interp_membership(x_adx, adx_notrend, input_adx)
        adx_fit_ontrend = fuzz.interp_membership(x_adx, adx_ontrend, input_adx)
        adx_fit_danger = fuzz.interp_membership(x_adx, adx_danger, input_adx)

        #fuzzy rule base configuration-----
        #bullish
        bullish_rule1 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_long), adx_fit_ontrend), bullish_weak)
        bullish_rule2 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_long), adx_fit_notrend), bullish_weak)
        bullish_rule3 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_long), adx_fit_danger), bullish_weak)
        bullish_rule4 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_long), adx_fit_ontrend), bullish_verystrong)
        bullish_rule5 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_long), adx_fit_notrend), bullish_weak)
        bullish_rule6 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_long), adx_fit_danger), bullish_strong)
        bullish_rule7 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_long), adx_fit_ontrend), bullish_verystrong)
        bullish_rule8 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_long), adx_fit_notrend), bullish_strong)
        bullish_rule9 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_long), adx_fit_danger), bullish_strong)
        bullish_rule10 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_short), adx_fit_ontrend), bullish_weak)
        bullish_rule11 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_short), adx_fit_notrend), bullish_weak)
        bullish_rule12 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_short), adx_fit_danger), bullish_weak)
        bullish_rule13 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_short), adx_fit_ontrend), bullish_weak)
        bullish_rule14 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_short), adx_fit_notrend), bullish_weak)
        bullish_rule15 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_short), adx_fit_danger), bullish_weak)
        bullish_rule16 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_short), adx_fit_ontrend), bullish_weak)
        bullish_rule17 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_short), adx_fit_notrend), bullish_weak)
        bullish_rule18 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_short), adx_fit_danger), bullish_weak)
        bullish_rule19 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_wait), adx_fit_ontrend), bullish_weak)
        bullish_rule20 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_wait), adx_fit_notrend), bullish_weak)
        bullish_rule21 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_wait), adx_fit_danger), bullish_weak)
        bullish_rule22 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_wait), adx_fit_ontrend), bullish_weak)
        bullish_rule23 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_wait), adx_fit_notrend), bullish_weak)
        bullish_rule24 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_wait), adx_fit_danger), bullish_weak)
        bullish_rule25 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_wait), adx_fit_ontrend), bullish_strong)
        bullish_rule26 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_wait), adx_fit_notrend), bullish_weak)
        bullish_rule27 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_wait), adx_fit_danger), bullish_weak)

        #bearish
        bearish_rule1 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_long), adx_fit_ontrend), bearish_weak)
        bearish_rule2 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_long), adx_fit_notrend), bearish_weak)
        bearish_rule3 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_long), adx_fit_danger), bearish_weak)
        bearish_rule4 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_long), adx_fit_ontrend), bearish_weak)
        bearish_rule5 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_long), adx_fit_notrend), bearish_weak)
        bearish_rule6 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_long), adx_fit_danger), bearish_weak)
        bearish_rule7 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_long), adx_fit_ontrend), bearish_weak)
        bearish_rule8 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_long), adx_fit_notrend), bearish_weak)
        bearish_rule9 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_long), adx_fit_danger), bearish_weak)
        bearish_rule10 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_short), adx_fit_ontrend), bearish_verystrong)
        bearish_rule11 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_short), adx_fit_notrend), bearish_strong)
        bearish_rule12 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_short), adx_fit_danger), bearish_strong)
        bearish_rule13 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_short), adx_fit_ontrend), bearish_verystrong)
        bearish_rule14 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_short), adx_fit_notrend), bearish_weak)
        bearish_rule15 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_short), adx_fit_danger), bearish_strong)
        bearish_rule16 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_short), adx_fit_ontrend), bearish_weak)
        bearish_rule17 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_short), adx_fit_notrend), bearish_weak)
        bearish_rule18 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_short), adx_fit_danger), bearish_weak)
        bearish_rule19 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_wait), adx_fit_ontrend), bearish_strong)
        bearish_rule20 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_wait), adx_fit_notrend), bearish_weak)
        bearish_rule21 = np.fmin(np.fmin(np.fmin(rsi_fit_high, macd_fit_wait), adx_fit_danger), bearish_weak)
        bearish_rule22 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_wait), adx_fit_ontrend), bearish_weak)
        bearish_rule23 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_wait), adx_fit_notrend), bearish_weak)
        bearish_rule24 = np.fmin(np.fmin(np.fmin(rsi_fit_med, macd_fit_wait), adx_fit_danger), bearish_weak)
        bearish_rule25 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_wait), adx_fit_ontrend), bearish_weak)
        bearish_rule26 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_wait), adx_fit_notrend), bearish_weak)
        bearish_rule27 = np.fmin(np.fmin(np.fmin(rsi_fit_low, macd_fit_wait), adx_fit_danger), bearish_weak)

        #identify output-----
        def fmaxRules(rules):
            temp = rules[0]
            for rule in rules:
                temp = np.fmax(rule, temp)
            return temp

        out_bullish_weak = fmaxRules([bullish_rule1, bullish_rule2, bullish_rule3, bullish_rule5, bullish_rule10, bullish_rule11, bullish_rule12, bullish_rule13, bullish_rule14, bullish_rule15, bullish_rule16, bullish_rule17, bullish_rule18, bullish_rule19, bullish_rule20, bullish_rule21, bullish_rule22, bullish_rule23, bullish_rule24, bullish_rule26, bullish_rule27])
        out_bullish_strong = fmaxRules([bullish_rule6, bullish_rule8, bullish_rule9, bullish_rule25])
        out_bullish_verystrong = fmaxRules([bullish_rule4, bullish_rule7])

        out_bearish_weak = fmaxRules([bearish_rule1, bearish_rule2, bearish_rule3, bearish_rule4, bearish_rule5, bearish_rule6, bearish_rule7, bearish_rule8, bearish_rule9, bearish_rule14, bearish_rule16, bearish_rule17, bearish_rule18, bearish_rule20, bearish_rule21, bearish_rule22, bearish_rule23, bearish_rule24, bearish_rule25, bearish_rule26, bearish_rule27])
        out_bearish_strong = fmaxRules([bearish_rule11, bearish_rule12, bearish_rule15, bearish_rule19])
        out_bearish_verystrong = fmaxRules([bearish_rule10, bearish_rule13])

        #defuzzify-----
        out_bullish = np.fmax(np.fmax(out_bullish_weak, out_bullish_strong), out_bullish_verystrong)
        defuzzified_bullish = fuzz.defuzz(x_bullish, out_bullish, 'centroid')
        #result_bullish = fuzz.interp_membership(x_bullish, out_bullish, defuzzified_bullish)

        out_bearish = np.fmax(np.fmax(out_bearish_weak, out_bearish_strong), out_bearish_verystrong)
        defuzzified_bearish = fuzz.defuzz(x_bearish, out_bearish, 'centroid')
        #result_bearish = fuzz.interp_membership(x_bearish, out_bearish, defuzzified_bearish)

        return [defuzzified_bullish, defuzzified_bearish]

#get data-----
data = yf.download(ticker, start, end)

#calculate indicators-----
rsi = data.ta.rsi()
macd_diff = data.ta.macd(close = 'Close').iloc[:, 1]
adx = data.ta.adx().iloc[:, 0]

data['RSI'] = rsi
data['MACD_difference'] = macd_diff
data['ADX'] = adx

data.dropna(inplace=True)

#implement fuzzy system on data-----
data['Bullish'] = np.nan
data['Bearish'] = np.nan
data['Buy'] = np.nan
data['Sell'] = np.nan

real_buys, real_sells = [], []

isBrought = False
for i in range(0 , len(data.index)):

    data.Bullish[i] = FuzzySystem(data.RSI[i], data.MACD_difference[i], data.ADX[i])[0]
    data.Bearish[i] = FuzzySystem(data.RSI[i], data.MACD_difference[i], data.ADX[i])[1]

    if not isBrought:
        if data.Bullish[i] >= 50:
            data.Buy[i] = data.Close[i]
            isBrought = True
            real_buys.append(i + 1)
    else:
        if data.Bearish[i] >= 50:
            data.Sell[i] = data.Close[i]
            isBrought = False
            real_sells.append(i + 1)

#calculate results
buy_prices = data.Open.iloc[real_buys]
sell_prices = data.Open.iloc[real_sells]

if len(buy_prices)==0:
    print('Not done any buy-sell operations')
    exit()

if buy_prices.index[-1] > sell_prices.index[-1]:
    buy_prices = buy_prices.drop(buy_prices.index[-1])

profits_rel = []

for i in range(len(sell_prices)):
    profits_rel.append((sell_prices[i] - buy_prices[i]) / buy_prices[i])

profits = sum(profits_rel)

profits_avr = sum(profits_rel) / len(profits_rel) 

#plot results-----
fig, ax = plt.subplots(figsize=(14,8))
ax.plot(data['Close'] , label = ticker + ' close' ,linewidth=0.5, color='blue', alpha = 0.9)
ax.plot(data['Adj Close'] , label = ticker + ' adjusted close' ,linewidth=0.25, color='black', alpha = 0.5)
ax.scatter(data.index , data['Buy'] , label = 'Buy' , marker = '^', color = 'green',alpha =1, zorder=10 )
ax.scatter(data.index , data['Sell'] , label = 'Sell' , marker = 'v', color = 'red',alpha =1, zorder=5 )
ax.set_title(ticker + " Price History with buy and sell signals",fontsize=10, backgroundcolor='blue', color='white')
ax.set_xlabel(f'{start} - {end}' ,fontsize=18)
ax.set_ylabel('Close Price INR (₨)' , fontsize=18)
legend = ax.legend()
ax.grid()

fig.text(0.5, 0.9, 'Average Profit: {0:.2f}%'.format(profits_avr * 100), fontsize = 20, horizontalalignment='center')
fig.text(0.5, 0.85, 'Total Profit: {0:.0f}%'.format(profits * 100), fontsize = 20, horizontalalignment='center')

plt.tight_layout()
plt.show()

#print(data)
