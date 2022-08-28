# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 18:02:41 2022

@author: mayllonu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)
a0 = 5
f0 = 3
delta_f = 5.0
s = a0 * np.sin(2 * np.pi * f0 * t)
(l,) = plt.plot(t, s, lw=2)
ax.margins(x=0)

axcolor = "lightgoldenrodyellow"
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

sfreq = Slider(axfreq, "Freq", 0.1, 30.0, valinit=f0, valstep=delta_f)
samp = Slider(axamp, "Amp", 0.1, 10.0, valinit=a0)


def update(val):
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp * np.sin(2 * np.pi * freq * t))
    fig.canvas.draw_idle()


sfreq.on_changed(update)
samp.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, "Reset", color=axcolor, hovercolor="0.975")


def reset(event):
    sfreq.reset()
    samp.reset()


button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ("red", "blue", "green"), active=0)


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()


radio.on_clicked(colorfunc)

plt.show()


###### Another way #########
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button

# ax = plt.subplot(111)
# plt.subplots_adjust(left=0.25, bottom=0.25)
# img_data = np.random.rand(50,50)
# c_min = 0
# c_max = 1

# img = ax.imshow(img_data, interpolation='nearest')
# cb = plt.colorbar(img)
# axcolor = 'lightgoldenrodyellow'


# ax_cmin = plt.axes([0.25, 0.1, 0.65, 0.03])
# ax_cmax  = plt.axes([0.25, 0.15, 0.65, 0.03])

# s_cmin = Slider(ax_cmin, 'min', 0, 1, valinit=c_min)
# s_cmax = Slider(ax_cmax, 'max', 0, 1, valinit=c_max)

# def update(val, s=None):
#     _cmin = s_cmin.val
#     _cmax = s_cmax.val
#     img.set_clim([_cmin, _cmax])
#     plt.draw()

# s_cmin.on_changed(update)
# s_cmax.on_changed(update)

# resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
# def reset(event):
#     s_cmin.reset()
#     s_cmax.reset()
# button.on_clicked(reset)

# plt.show()
###############################################################


# # test layout
# fig.set_constrained_layout(True)
# fig, _ = plt.subplots()
# fig.set_constrained_layout(True)
# gs = fig.add_gridspec(2, 2, width_ratios=[0.5, 0.5],
#                       height_ratios=[1,1], top=5)

# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[1, 0])
# ax3 = fig.add_subplot(gs[:, 1])
# ax_test = fig.add_axes([0.62, 0.93, 0.3, 0.05]) #[left, bottom, width, height]
# ax_test.set_yticks([])
# ax_test.set_xticks([])
