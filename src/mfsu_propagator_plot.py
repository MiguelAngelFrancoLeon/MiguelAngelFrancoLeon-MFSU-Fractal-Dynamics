import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def eta(df, lam):
    return (df - 2)/2 + lam**2 / (32 * np.pi**2)

def G(p, df, m, lam):
    return 1 / (p**2)**(df/2 - eta(df, lam)) + m**2

p = np.logspace(-2, 2, 500)
df0 = 1.5
m0 = 1.0
lam0 = 1.0

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.35)

l, = plt.plot(p, G(p, df0, m0, lam0), lw=2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Momento $p$')
ax.set_ylabel('Propagador $G(p)$')
ax.set_title('Propagador Fractal MFSU')

ax_df = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_m = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_lam = plt.axes([0.25, 0.15, 0.65, 0.03])

s_df = Slider(ax_df, 'd_f', 1.0, 3.0, valinit=df0, valstep=0.01)
s_m = Slider(ax_m, 'm', 0.1, 5.0, valinit=m0)
s_lam = Slider(ax_lam, 'λ', 0.0, 5.0, valinit=lam0)

def update(val):
    df = s_df.val
    m = s_m.val
    lam = s_lam.val
    l.set_ydata(G(p, df, m, lam))
    fig.canvas.draw_idle()

s_df.on_changed(update)
s_m.on_changed(update)
s_lam.on_changed(update)

plt.show()
