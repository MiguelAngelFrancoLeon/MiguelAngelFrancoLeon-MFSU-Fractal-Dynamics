import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def Tc(df, C, delta):
    return C / np.sqrt(df - 1) * np.tanh(np.sqrt(df - 1) / delta)

df_vals = np.linspace(1.01, 2.0, 500)
C0 = 350
delta0 = 0.01

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.35)

l, = plt.plot(df_vals, Tc(df_vals, C0, delta0), lw=2)
ax.set_xlabel("Dimensión fractal $d_f$")
ax.set_ylabel("Temperatura crítica $T_c$ (K)")
ax.set_title("Temperatura Crítica vs Dimensión Fractal")
ax.grid(True)

ax_C = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_delta = plt.axes([0.25, 0.2, 0.65, 0.03])

s_C = Slider(ax_C, 'C (K)', 100, 500, valinit=C0)
s_delta = Slider(ax_delta, 'δ', 0.001, 0.1, valinit=delta0)

def update(val):
    C = s_C.val
    delta = s_delta.val
    l.set_ydata(Tc(df_vals, C, delta))
    fig.canvas.draw_idle()

s_C.on_changed(update)
s_delta.on_changed(update)

plt.show()
