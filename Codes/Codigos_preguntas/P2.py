import numpy as np
import matplotlib.pyplot as plt
from parametros import BCF, B_tri, y_2


u_TRI = np.linspace(-100, 100, 1000)
u_BI = (y_2 + B_tri * u_TRI) / BCF
#tramo factible
u_TRI_f = np.linspace(0, 1, 200)
u_BI_f = (y_2 + B_tri * u_TRI_f) / BCF

mask = (u_BI_f >= 0) & (u_BI_f <= 1)


fig, ax = plt.subplots(figsize=(10, 6))


ax.fill_between([0, 1], 0, 1, color="tab:blue", alpha=0.08, label=r"$u \in [0,1]$")
ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="tab:blue", lw=1)

ax.plot(u_TRI, u_BI, "k--", lw=1, label="Recta de equilibrio")
ax.plot(u_TRI_f[mask], u_BI_f[mask], "r", lw=3, label="Soluciones factibles")



ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel(r"$u_{TRI}$")
ax.set_ylabel(r"$u_{BI}$")
ax.set_title("Soluciones de equilibrio del codo en q_0")
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.legend(loc="upper left")

plt.tight_layout()
plt.show()