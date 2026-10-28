from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from IEE2613_arm_utils import simular, tono_equilibrio, cargar_modelo, Q0

L1, LM = 0.30, 0.32                                   # largo del brazo y del antebrazo [m]
mano = lambda q: np.stack([L1 * np.sin(q[:, 0]) + LM * np.sin(q[:, 0] + q[:, 1]),
                           -L1 * np.cos(q[:, 0]) - LM * np.cos(q[:, 0] + q[:, 1])], axis=1)

u0 = tono_equilibrio(cargar_modelo(str(Path(__file__).parent / 'xml' / 'brazo_2dof_musculos.xml')), Q0)[0]
ley = lambda t, y, r: u0                              # open loop

t, q, u = simular(ley, T=0.010, q_ref=Q0, duracion=6.0, salida='excitacion',
                  pulso=3.0, t_pulso=2.0, gui=True)  

p = (mano(q) - mano(q)[0]) * 100                      # trayectoria de la mano 
plt.plot(p[:, 0], p[:, 1]); plt.plot(0, 0, 'ko')
plt.axis('equal'); plt.grid(alpha=.3)
plt.xlabel('x de la mano [cm]'); plt.ylabel('z de la mano [cm]'); plt.title('Trayectoria de la mano')
plt.show()
