# Oscilador de dos neuronas Morris-Lecar

import numpy as np


ML = dict(C=20.0, gL=2.0, VL=-60.0, gCa=4.4, VCa=120.0, gK=8.0, VK=-84.0,
          V1=-1.2, V2=18.0, V3=2.0, V4=30.0, phi=0.005)


class CPGMorrisLecar:

    def __init__(self, I_app=110.0, g_syn=8.0, E_syn=-80.0, V_th=-20.0, V_s=2.0,
                 g_fb=0.0, E_fb=-80.0, tempo=0.28, n_sub=6, prm=ML):
        self.p = dict(prm)
        self.I_app = I_app
        self.g_syn = g_syn  # inhibicion mutua
        self.E_syn = E_syn
        self.V_th, self.V_s = V_th, V_s
        self.g_fb = g_fb  # ganancia de la inhibicion sensorial (huso)
        self.E_fb = E_fb
        self.tempo = tempo  # ms neurales por ms reales (<1 = mas lento)
        self.n_sub = n_sub
        self.reset()

    def reset(self):
        self.V = np.array([10.0, -50.0])
        self.w = np.array([0.30, 0.05])
        self.I_fb = np.zeros(2)

    # funciones de estado estacionario
    def _derivadas(self, V, w, f_sens):
        p = self.p
        minf = 0.5 * (1 + np.tanh((V - p['V1']) / p['V2']))
        winf = 0.5 * (1 + np.tanh((V - p['V3']) / p['V4']))
        tw   = 1.0 / np.cosh((V - p['V3']) / (2 * p['V4']))
        s    = 0.5 * (1 + np.tanh((V - self.V_th) / self.V_s))  # sinapsis
        I_syn = self.g_syn * s[::-1] * (V - self.E_syn)  # inhibicion mutua
        I_fb = self.g_fb * f_sens[::-1] * (V - self.E_fb)
        dV = (self.I_app - p['gL'] * (V - p['VL']) - p['gCa'] * minf * (V - p['VCa'])
              - p['gK'] * w * (V - p['VK']) - I_syn - I_fb) / p['C']
        dw = p['phi'] * (winf - w) / tw
        return dV, dw, I_fb

    def paso(self, Ts, f_sens=(0.0, 0.0)):
        f = np.maximum(np.asarray(f_sens, dtype=float), 0.0)
        h = 1000.0 * Ts * self.tempo / self.n_sub  # paso neural [ms]
        for _ in range(self.n_sub):
            dV, dw, self.I_fb = self._derivadas(self.V, self.w, f)
            self.V = self.V + h * dV
            self.w = self.w + h * dw
        return self.V

    @staticmethod
    def actividad(V, V_mid=-15.0, ancho=12.0):
        return 0.5 * (1 + np.tanh((V - V_mid) / ancho))

    def salidas(self):
        return self.actividad(self.V)


def estiramiento_normalizado(l, lam, L0):
    return np.maximum((np.asarray(l) - lam) / L0, 0.0)
