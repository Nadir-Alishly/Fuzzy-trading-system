import numpy as np
import skfuzzy as fuzz
import skfuzzy.membership as mf
import matplotlib.pyplot as plt

#inputs-----
input_rsi = 59.24
input_macd = -1.54
input_adx = 13.25

#input scaling-----
if (input_macd>5): input_macd = 5
if (input_macd<-5): input_macd = -5

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

#macd_wait_1 = mf.trimf(x_macd, [0, 0, 20])
#macd_wait_2 = mf.trimf(x_macd, [30, 50, 70])
#macd_wait_3 = mf.trimf(x_macd, [80, 100, 100])
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

#graph fuzzy subsets-----
fig1, ((ax0, ax1), (ax2, null), (ax3, ax4)) = plt.subplots(nrows = 3, ncols = 2, figsize = (6, 8))

ax0.plot(x_rsi, rsi_low, 'r', linewidth = 2, linestyle = '--', label = 'low')
ax0.plot(x_rsi, rsi_med, 'y', linewidth = 2, linestyle = '--', label = 'medium')
ax0.plot(x_rsi, rsi_high, 'g', linewidth = 2, linestyle = '--', label = 'high')
ax0.set_title('rsi')
ax0.legend()

ax1.plot(x_macd, macd_wait, 'y', linewidth = 2, linestyle = '--', label = 'wait')
ax1.plot(x_macd, macd_short, 'r', linewidth = 2, linestyle = '--', label = 'short')
ax1.plot(x_macd, macd_long, 'g', linewidth = 2, linestyle = '--', label = 'long')
ax1.set_title('macd')
ax1.legend()

ax2.plot(x_adx, adx_notrend, 'y', linewidth = 2, linestyle = '--', label = 'no trend')
ax2.plot(x_adx, adx_ontrend, 'g', linewidth = 2, linestyle = '--', label = 'on trend')
ax2.plot(x_adx, adx_danger, 'r', linewidth = 2, linestyle = '--', label = 'danger')
ax2.set_title('adx')
ax2.legend()

ax3.plot(x_bullish, bullish_weak, 'r', linewidth = 2, linestyle = '--', label = 'weak')
ax3.plot(x_bullish, bullish_strong, 'g', linewidth = 2, linestyle = '--', label = 'strong')
ax3.plot(x_bullish, bullish_verystrong, 'b', linewidth = 2, linestyle = '--', label = 'very strong')
ax3.set_title('bullish')
ax3.legend()

ax4.plot(x_bearish, bearish_weak, 'r', linewidth = 2, linestyle = '--', label = 'weak')
ax4.plot(x_bearish, bearish_strong, 'g', linewidth = 2, linestyle = '--', label = 'strong')
ax4.plot(x_bearish, bearish_verystrong, 'b', linewidth = 2, linestyle = '--', label = 'very strong')
ax4.set_title('bearish')
ax4.legend()

plt.tight_layout()
plt.autoscale(enable=False)

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

#graph results-----
bullish_bearish0 = np.zeros_like(x_bullish)

ax0.axhline(rsi_fit_low, color = 'r', linewidth = 1, linestyle = '-')
ax0.axhline(rsi_fit_med, color = 'y', linewidth = 1, linestyle = '-')
ax0.axhline(rsi_fit_high, color = 'g', linewidth = 1, linestyle = '-')
ax0.legend()

ax1.axhline(macd_fit_wait, color = 'y', linewidth = 1, linestyle = '-')
ax1.axhline(macd_fit_short, color = 'r', linewidth = 1, linestyle = '-')
ax1.axhline(macd_fit_long, color = 'g', linewidth = 1, linestyle = '-')
ax1.legend()

ax2.axhline(adx_fit_notrend, color = 'y', linewidth = 1, linestyle = '-')
ax2.axhline(adx_fit_ontrend, color = 'g', linewidth = 1, linestyle = '-')
ax2.axhline(adx_fit_danger, color = 'r', linewidth = 1, linestyle = '-')
ax2.legend()

ax3.fill_between(x_bullish, bullish_bearish0, out_bullish_weak, facecolor = 'r', alpha = 0.3)
ax3.fill_between(x_bullish, bullish_bearish0, out_bullish_strong, facecolor = 'g', alpha = 0.3)
ax3.fill_between(x_bullish, bullish_bearish0, out_bullish_verystrong, facecolor = 'b', alpha = 0.3)

ax4.fill_between(x_bearish, bullish_bearish0, out_bearish_weak, facecolor = 'r', alpha = 0.3)
ax4.fill_between(x_bearish, bullish_bearish0, out_bearish_strong, facecolor = 'g', alpha = 0.3)
ax4.fill_between(x_bearish, bullish_bearish0, out_bearish_verystrong, facecolor = 'b', alpha = 0.3)

fig1.show()

#defuzzify-----
out_bullish = np.fmax(np.fmax(out_bullish_weak, out_bullish_strong), out_bullish_verystrong)
defuzzified_bullish = fuzz.defuzz(x_bullish, out_bullish, 'centroid')
result_bullish = fuzz.interp_membership(x_bullish, out_bullish, defuzzified_bullish)

out_bearish = np.fmax(np.fmax(out_bearish_weak, out_bearish_strong), out_bearish_verystrong)
defuzzified_bearish = fuzz.defuzz(x_bearish, out_bearish, 'centroid')
result_bearish = fuzz.interp_membership(x_bearish, out_bearish, defuzzified_bearish)

#graph final results
fig2, (ax5, ax6) = plt.subplots(nrows = 2, figsize = (5,7))

ax5.fill_between(x_bullish, bullish_bearish0, out_bullish, facecolor = 'g')
ax5.axvline(defuzzified_bullish, color = 'k', label = 'defuzzified')
ax5.set_title('bullish output')

ax6.fill_between(x_bearish, bullish_bearish0, out_bearish, facecolor = 'r')
ax6.axvline(defuzzified_bearish, color = 'k', label = 'defuzzified')
ax6.set_title('bearish output')

fig2.tight_layout()
fig2.show()

print("bullish defuzzified: ")
print(defuzzified_bullish)
print("bearish defuzzified: ")
print(defuzzified_bearish)


input('Press ENTER to exit')

