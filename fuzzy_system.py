import numpy as np
import skfuzzy as fuzz
import skfuzzy.membership as mf
import matplotlib.pyplot as plt

#inputs-----
input_pedal = 20
input_speed = 90

#indentification of (input) variables-----
x_pedal = np.arange(0, 100, 1)
x_speed = np.arange(0, 100, 1)
x_brake = np.arange(0, 100, 1)

#fuzzy subset configuartion & obtain membership functions-----
pedal_low = mf.trimf(x_pedal, [0, 0, 50])
pedal_med = mf.trimf(x_pedal, [0, 50, 100])
pedal_high = mf.trimf(x_pedal, [50, 100, 100])

speed_low = mf.trimf(x_speed, [0, 0, 60])
speed_med = mf.trimf(x_speed, [20, 50, 80])
speed_high = mf.trimf(x_speed, [40, 100, 100])

brake_poor = mf.trimf(x_brake, [0, 0, 100])
brake_strong = mf.trimf(x_brake, [0, 100, 100])

#graph fuzzy subsets-----
fig, (ax0, ax1, ax2) = plt.subplots(nrows = 3, figsize = (6, 10))

ax0.plot(x_pedal, pedal_low, 'r', linewidth = 2, label = 'low')
ax0.plot(x_pedal, pedal_med, 'g', linewidth = 2, label = 'medium')
ax0.plot(x_pedal, pedal_high, 'b', linewidth = 2, label = 'high')
ax0.set_title('pedal pressure')
ax0.legend()

ax1.plot(x_speed, speed_low, 'r', linewidth = 2, label = 'low')
ax1.plot(x_speed, speed_med, 'g', linewidth = 2, label = 'medium')
ax1.plot(x_speed, speed_high, 'b', linewidth = 2, label = 'high')
ax1.set_title('speed')
ax1.legend()

ax2.plot(x_brake, brake_poor, 'r', linewidth = 2, label = 'poor')
ax2.plot(x_brake, brake_strong, 'g', linewidth = 2, label = 'strong')
ax2.set_title('brake')
ax2.legend()

#plt.tight_layout()
fig.show()

#get membership values of input-----
pedal_fit_low = fuzz.interp_membership(x_pedal, pedal_low, input_pedal)
pedal_fit_med = fuzz.interp_membership(x_pedal, pedal_med, input_pedal)
pedal_fit_high = fuzz.interp_membership(x_pedal, pedal_high, input_pedal)

speed_fit_low = fuzz.interp_membership(x_speed, speed_low, input_speed)
speed_fit_med = fuzz.interp_membership(x_speed, speed_med, input_speed)
speed_fit_high = fuzz.interp_membership(x_speed, speed_high, input_speed)

#fuzzy rule base configuration-----
rule1 = np.fmin(pedal_fit_med, brake_strong)
rule2 = np.fmin(np.fmax(pedal_fit_high, speed_fit_high), brake_strong)
rule3 = np.fmin(np.fmax(pedal_fit_low, speed_fit_low), brake_poor)
rule4 = np.fmin(pedal_fit_low, brake_poor)

#identify output-----
out_strong = np.fmax(rule1, rule2)
out_poor = np.fmax(rule3, rule4)

#graph results-----
brake0 = np.zeros_like(x_brake)

fig, ax0 = plt.subplots(figsize = (7, 4))
ax0.fill_between(x_brake, brake0, out_poor, facecolor = 'r', alpha = 0.5)
ax0.plot(x_brake, brake_poor, 'r', linestyle = '--')
ax0.fill_between(x_brake, brake0, out_strong, facecolor = 'g', alpha = 0.5)
ax0.plot(x_brake, brake_strong, 'g', linestyle = '--')
ax0.set_title = 'brake output'
fig.show()

#defuzzify-----
out_brake = np.fmax(out_poor, out_strong)
defuzzified = fuzz.defuzz(x_brake, out_brake, 'centroid')
result = fuzz.interp_membership(x_brake, out_brake, defuzzified)

fig, ax4 = plt.subplots(figsize = (7,4))
ax4.fill_between(x_brake, brake0, out_brake, facecolor = 'b')
ax4.axvline(defuzzified, color = 'y', label = 'defuzzified')
ax4.set_title = 'brake output'
fig.show()

print(result)
print(defuzzified)

input('Press ENTER to exit')
