import numpy as np
from pathlib import Path

Q0 = np.deg2rad([0.0, 90.0])                          # postura nominal [hombro, codo] [rad]
U0_NOM = np.array([0.1112, 0.0500, 0.1236, 0.0500])   # tono que sostiene Q0 [DA, DP, BI, TRI]
RETARDO = 0.030                                       # retardo neural [s]

ESC_E, ESC_P, ESC_V, ESC_I = 0.35, 0.7, 2.0, 0.10
I_MAX = 0.30          
E_INT_ON = 0.15       


class Observador: # vector de entrada de la red
    def __init__(self, T=0.010, retardo=RETARDO, n_hist=4, n_act=None):
        self.T, self.n_hist = float(T), int(n_hist)
        self.n_act = int(n_act) if n_act is not None else int(np.ceil(retardo / T - 1e-9)) + 1
        self.dim = 2 + 2 + 2 + 2 * (self.n_hist - 1) + 2 + 4 * self.n_act
        self.reset()

    def reset(self):
        self.hist = None
        self.acts = [U0_NOM.copy() for _ in range(self.n_act)] 
        self.eint = np.zeros(2)

    def observe(self, y, r):
        y, r = np.asarray(y, float).reshape(2), np.asarray(r, float).reshape(2)
        self.hist = [y.copy()] * self.n_hist if self.hist is None else [y.copy()] + self.hist[:-1]
        e = r - y
        self.eint = np.clip(self.eint + np.where(np.abs(e) < E_INT_ON, e * self.T, 0.0), -I_MAX, I_MAX)
        v = np.concatenate([(self.hist[i] - self.hist[i + 1]) / self.T for i in range(self.n_hist - 1)])
        x = np.concatenate([e / ESC_E, (r - Q0) / ESC_P, (y - Q0) / ESC_P, v / ESC_V,
                            self.eint / ESC_I, 2.0 * np.concatenate(self.acts) - 1.0])
        return np.clip(x, -5.0, 5.0).astype(np.float32)

    def register_action(self, u):
        self.acts = [np.asarray(u, float).copy()] + self.acts[:-1]


class ControladorRL: 
    def __init__(self, ruta=None):
        import torch
        ruta = Path(ruta) if ruta is not None else Path(__file__).with_name('controlador_rl.pt')
        ck = torch.load(ruta, map_location='cpu', weights_only=True)
        sd = ck['actor']
        capas = sorted({int(k.split('.')[0]) for k in sd})
        self.W = [sd[f'{i}.weight'].double().numpy() for i in capas]
        self.b = [sd[f'{i}.bias'].double().numpy() for i in capas]
        self.lo, self.hi = ck['u_low'].double().numpy(), ck['u_high'].double().numpy()
        self.T = float(ck['T'])
        self.obs = Observador(self.T, float(ck['retardo']), int(ck['n_hist']), int(ck['n_act']))
        if self.obs.dim != self.W[0].shape[1]:
            raise ValueError('El archivo .pt no corresponde a esta version de controlador_rl.py')
        self._t_prev = None

    def _red(self, x):
        h = x
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.tanh(W @ h + b)
        z = self.W[-1] @ h + self.b[-1]
        return self.lo + 0.5 * (np.tanh(z) + 1.0) * (self.hi - self.lo)

    def __call__(self, t, y, r):
        if self._t_prev is None or t <= self._t_prev + 1e-12:   # simulacion nueva (t vuelve a 0)
            self.obs.reset()
        elif abs((t - self._t_prev) - self.T) > 1e-6:
            raise ValueError(f'Este controlador se entreno con T = {self.T*1e3:.0f} ms; use simular(..., T=ctrl.T).')
        self._t_prev = t
        u = self._red(self.obs.observe(y, r))
        self.obs.register_action(u)
        return u
