import numpy as np
import matplotlib.pyplot as plt

from IEE2613_arm_utils import simular
from controlador_rl import ControladorRL

Q0 = np.deg2rad([0.0, 90.0])
TS = 0.002                      # paso de MuJoCo, simular entrega t, q, u cada 2 ms


def correr_rl(q_ref, duracion, **kw):
    ctrl = ControladorRL('controlador_rl.pt')
    return simular(ctrl, T=ctrl.T, q_ref=q_ref, duracion=duracion, salida='excitacion', **kw)


def esfuerzo(u):
    return float(np.sum(u ** 2) * TS)          # J_u = int sum_i u_i^2 dt


if __name__ == '__main__':
    ref = lambda t: Q0 + (np.deg2rad([0.0, 20.0]) if t >= 0.5 else np.zeros(2))
    t, q, u = correr_rl(ref, 4.0)
    print(f'Escalon codo +20 deg: angulo final {np.rad2deg(q[-1, 1]):.2f} deg, J_u = {esfuerzo(u):.3f}')

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(t, np.rad2deg(q[:, 1]), label='codo'); ax[0].plot(t, [np.rad2deg(ref(x)[1]) for x in t], 'k--', label='referencia')
    ax[0].set_xlabel('t [s]'); ax[0].set_ylabel('angulo [deg]'); ax[0].legend(); ax[0].grid(alpha=.3)
    for i, nom in enumerate(['DA', 'DP', 'BI', 'TRI']):
        ax[1].plot(t, u[:, i], label=nom)
    ax[1].set_xlabel('t [s]'); ax[1].set_ylabel('excitacion'); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()

    plt.figure(figsize=(6, 3.5))
    for nom, kw in [('nominal', {}), ('carga 1 kg', dict(carga=1.0)), ('sarcopenia', dict(sarcopenia=0.5)),
                    ('activacion fisiologica', dict(activacion_fisio=True))]:
        t, q, u = correr_rl(ref, 4.0, **kw)
        print(f'{nom:24s}: angulo final {np.rad2deg(q[-1, 1]):.2f} deg, J_u = {esfuerzo(u):.3f}')
        plt.plot(t, np.rad2deg(q[:, 1]), label=nom)
    plt.xlabel('t [s]'); plt.ylabel('codo [deg]'); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    
    t, q, u = correr_rl(Q0, 4.0, pulso=3.0, t_pulso=2.0)
    print(f'Pulso +3 N m: desviacion maxima del codo {np.rad2deg(np.abs(q[:, 1] - Q0[1]).max()):.2f} deg')

    plt.show()
