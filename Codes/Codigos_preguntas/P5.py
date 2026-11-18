
import sys
import os
sys.path.append('../')

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import matplotlib.pyplot as plt
from Codes.IEE2613_arm_utils import tono_equilibrio, cargar_modelo, simular
import numpy as np
import control as ct


from parametros import (BCF,JC,t_a,BC,KC)

pm = np.polymul

A = [t_a, 1]
B = [JC, BC, KC]

mult = pm(A,B)

num = [BCF]
den = [mult[0], mult[1], mult[2], mult[3]]

Gc= ct.tf(num, den)

q0 = np.deg2rad([0.0, 90.0])
tau_eq = np.array([2.904, 2.904])



u0 = tono_equilibrio(cargar_modelo('xml/brazo_2dof_musculos.xml'), q0)[0]  # tono [DA, DP, BI, TRI]

t_escalon = 0.5


fig, axs = plt.subplots(len([0.02, 0.05, 0.1]), 1, figsize=(8, 3 * len([0.02, 0.05, 0.1])), sharex=True)

for ax, delta_u in zip(axs, [0.02, 0.05, 0.1]):
    ley = lambda t, y, r: u0 + np.array([0, 0, delta_u if t >= t_escalon else 0.0, 0])
    t, q, u = simular(ley, T=0.002, q_ref=q0, duracion=3.0, salida='excitacion')

    ts = t - t_escalon
    dq_mj = q[:, 1] - q0[1] 
    u_step = np.zeros_like(t)
    u_step[t >= t_escalon] = delta_u

    mask = t >= 0
    dq_lineal = np.zeros_like(t)
    _, y = ct.step_response(Gc, T=ts[mask])
    dq_lineal[mask] = delta_u * y

    ax.plot(t, np.rad2deg(dq_mj), label='MuJoCo')
    ax.plot(t, np.rad2deg(dq_lineal), '--', label='$G_c(s)$')
    ax.set_title(f'Escalon delta_u = {delta_u}')
    ax.set_ylabel('delta codo')
    ax.grid(True)
    ax.legend()

    # normalizar
    # ax_n.plot(t, np.rad2deg(dq_mj) / delta_u, label=f'MuJoCo delta_u={delta_u}')

    # error
    err = np.sqrt(np.mean((dq_mj - dq_lineal)**2)) / abs(dq_lineal[-1]) * 100 # rms
    print(f'delta_u={delta_u}: valor final MuJoCo={np.rad2deg(dq_mj[-1]):.2f}°, '
          f'lineal={np.rad2deg(dq_lineal[-1]):.2f}°, error RMS={err:.1f}% del valor final lineal')


fig.tight_layout()
plt.show()