import numpy as np
import mujoco

from IEE2613_arm_utils import NOMBRES_CORTOS

ANCHO_TENDON_MIN = 0.007
ANCHO_TENDON_MAX = 0.024

COLORES_MUSCULO = np.array([[0.95, 0.35, 0.25],  # DA
                            [0.25, 0.65, 0.95],  # DP
                            [0.98, 0.75, 0.20],  # BI
                            [0.45, 0.85, 0.45]])  # TRI


def _color_activacion(a_vis):
    return [0.55 + 0.45 * a_vis, 0.12 + 0.18 * a_vis + 0.25 * a_vis ** 2, 0.12 * (1 - a_vis), 1.0]


def colorear_musculos(model, data):
    for k in range(model.ntendon):
        a = float(np.clip(data.act[k], 0.0, 1.0))
        a_vis = min(1.0, 3.0 * a)
        c = _color_activacion(a_vis)
        model.tendon_rgba[k] = [0.5 * c[0] + 0.35, 0.5 * c[1] + 0.30, 0.5 * c[2] + 0.28, 1.0]
        model.tendon_width[k] = ANCHO_TENDON_MIN * 0.7


def agregar_vientres(scene, model, data, radio_min=0.007, radio_max=0.017):
    W = data.wrap_xpos.reshape(-1, 3)
    for k in range(model.ntendon):
        adr, n = data.ten_wrapadr[k], data.ten_wrapnum[k]
        pts = W[adr:adr + n]
        if len(pts) < 2:
            continue
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        j = int(np.argmax(seg))
        p0, p1 = pts[j], pts[j + 1]
        L = seg[j]
        if L < 1e-4 or scene.ngeom >= scene.maxgeom:
            continue
        a_vis = min(1.0, 3.0 * float(np.clip(data.act[k], 0.0, 1.0)))
        r = radio_min + (radio_max - radio_min) * a_vis
        z = (p1 - p0) / L
        x = np.cross(z, [0.0, 1.0, 0.0])
        if np.linalg.norm(x) < 1e-6:
            x = np.cross(z, [1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        mat = np.stack([x, y, z], axis=1).flatten()
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ELLIPSOID, np.array([r, r, 0.42 * L]),
                            0.5 * (p0 + p1), mat, np.asarray(_color_activacion(a_vis), dtype=np.float32))
        scene.ngeom += 1


def _capsula(scene, p0, p1, radio, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radio,
                         np.asarray(p0, dtype=float), np.asarray(p1, dtype=float))
    scene.ngeom += 1


def _esfera(scene, p, radio, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radio, 0, 0]), np.asarray(p, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def agregar_fantasma(scene, model, data_vis, q, rgba):
    data_vis.qpos[:] = q
    mujoco.mj_kinematics(model, data_vis)
    hombro = data_vis.body('brazo').xpos.copy()
    codo   = data_vis.body('antebrazo').xpos.copy()
    mano   = data_vis.site('mano').xpos.copy()
    _capsula(scene, hombro, codo, 0.015, rgba)
    _capsula(scene, codo, mano, 0.012, rgba)
    _esfera(scene, mano, 0.022, rgba)


class GraficosVivo:

    def __init__(self, T_ventana=4.0, n_max=400):
        self.T = T_ventana
        self.n_max = n_max
        self.t = []
        self.series = {}
        self.figs = [mujoco.MjvFigure() for _ in range(3)]
        for f in self.figs:
            mujoco.mjv_defaultFigure(f)
            f.figurergba[:] = [0.06, 0.07, 0.09, 0.82]
            f.panergba[:]   = [0.10, 0.11, 0.14, 0.9]
            f.gridrgb[:]    = [0.35, 0.35, 0.40]
            f.textrgb[:]    = [0.92, 0.92, 0.95]
            f.legendrgba[:] = [0.0, 0.0, 0.0, 0.0]
            f.gridsize[:]   = [5, 5]
            f.flg_legend    = 1
            f.flg_extend    = 0
            f.xformat       = '%.0f'
            f.yformat       = '%.0f'
            f.range[0][:]   = [-self.T, 0.0]
            f.linewidth     = 2.0

    def reset(self):
        self.t = []
        self.series = {}

    def agregar(self, t, **valores):
        if self.t and t < self.t[-1]:  # se reinicio la simulacion
            self.reset()
        self.t.append(t)
        for k, v in valores.items():
            self.series.setdefault(k, []).append(np.atleast_1d(np.asarray(v, float)).copy())
        # descartar lo que salio de la ventana
        while len(self.t) > 2 and (self.t[-1] - self.t[0] > self.T or len(self.t) > self.n_max):
            self.t.pop(0)
            for k in self.series:
                self.series[k].pop(0)

    def _cargar(self, fig, lineas, titulo, yrango, ylabel, yformat='%.0f', ngrid=5):
        fig.title = titulo
        fig.gridsize[1] = ngrid
        fig.xlabel = 't - t_actual [s]' + ylabel
        fig.yformat = yformat
        fig.range[1][:] = yrango
        t = np.array(self.t)
        x = t - t[-1] if len(t) else t
        for i in range(len(fig.linepnt)):
            fig.linepnt[i] = 0
        for i, (nombre, y, rgb) in enumerate(lineas):
            n = len(y)
            fig.linename[i] = nombre
            fig.linergb[i] = rgb
            fig.linepnt[i] = n
            fig.linedata[i, 0:2 * n:2] = x[:n]
            fig.linedata[i, 1:2 * n:2] = y

    def dibujar(self, viewport, context, modo, mostrar_neuronas):
        if len(self.t) < 2:
            return
        S = {k: np.array(v) for k, v in self.series.items()}
        if mostrar_neuronas:
            l1 = [('N1 flex', S['V'][:, 0], [0.95, 0.35, 0.25]),
                  ('N2 ext',  S['V'][:, 1], [0.25, 0.65, 0.95])]
            self._cargar(self.figs[0], l1, 'Neuronas CPG (Morris-Lecar) V [mV]',
                         [-100, 100], '')
        else:
            l1 = [(NOMBRES_CORTOS[i], 100 * S['u'][:, i], COLORES_MUSCULO[i]) for i in range(4)]
            umax = 100 * float(np.max(S['u']))
            self._cargar(self.figs[0], l1, 'Excitacion neural u [%]',
                         [0, 20 * np.ceil(max(umax, 15) / 20 * 1.1)], '')
        # Grafico 2: fuerza muscular
        l2 = [(NOMBRES_CORTOS[i], S['fuerza'][:, i], COLORES_MUSCULO[i]) for i in range(4)]
        fmax = float(np.max(S['fuerza']))
        self._cargar(self.figs[1], l2, 'Fuerza muscular (Golgi) [N]',
                     [0, 100 * np.ceil(max(fmax, 100) / 100 * 1.05)], '')
        # Grafico 3: angulos y referencias
        l3 = [('hombro', S['q'][:, 0], [0.95, 0.95, 0.95]),
              ('codo',   S['q'][:, 1], [1.0, 0.75, 0.2])]
        if modo != 'manual':
            l3.append(('ref codo', S['q_ref'][:, 1], [0.6, 0.45, 0.15]))
            if modo == 'postura':
                l3.append(('ref hombro', S['q_ref'][:, 0], [0.55, 0.55, 0.55]))
        self._cargar(self.figs[2], l3, 'Angulos articulares [deg]', [-40, 200], '', ngrid=7)

        W, H = viewport.width, viewport.height
        w = int(0.34 * W)
        h = int(H / 3)
        for k, f in enumerate(self.figs):
            rect = mujoco.MjrRect(W - w, H - (k + 1) * h, w, h)
            mujoco.mjr_figure(rect, f, context)


NOMBRE_MODO = {'auto': 'AUTO (mano: IK + control)',
               'rl': 'RL (politica PPO)',
               'manual': 'MANUAL (lazo abierto)',
               'postura': 'POSTURA (tono + reflejo)',
               'ritmo': 'RITMO (CPG neural + reflejo)'}


def texto_estado(data, ui):
    q = np.rad2deg(data.qpos)
    izq = ["Modo", "t", "q hombro / codo", "u  DA DP BI TRI"]
    der = [NOMBRE_MODO[ui.control_mode] + ("   [PAUSA]" if ui.pausa else ""),
           f"{data.time:6.2f} s",
           f"{q[0]:6.1f} / {q[1]:6.1f} deg",
           "  ".join(f"{u:.2f}" for u in data.ctrl)]
    if ui.control_mode == 'auto' and ui.p_ref is not None:
        izq += ["p_ref mano x / z"]
        der += [f"{100*ui.p_ref[0]:6.1f} / {100*ui.p_ref[1]:6.1f} cm"]
    elif ui.control_mode != 'manual':
        izq += ["q_ref hombro / codo", "reflejo K_l / retardo"]
        der += [f"{ui.q_ref_deg[0]:6.1f} / {ui.q_ref_deg[1]:6.1f} deg",
                (f"{ui.K_l:.2f} / {ui.retardo*1e3:.0f} ms" if ui.reflejo_on else "OFF")]
    if ui.control_mode == 'ritmo':
        izq += ["CPG tempo / amplitud", "realim. sensorial g_fb"]
        der += [f"{ui.cpg_tempo:.3f} / {ui.cpg_amp:.2f}",
                (f"{ui.cpg_g_fb:.1f}" if ui.cpg_fb_on else "OFF")]
    izq.append("H = ayuda")
    der.append("")
    return "\n".join(izq), "\n".join(der)


AYUDA_IZQ = "\n".join(["M / P / O / Q / B", "a/z s/x d/c f/v", "1/2  3/4", "5/6", "K/L  Y/U  R",
                       "7/8  9/0", "I  J/T", "E / N", "ESPACIO G W", "BACKSPACE / ESC"])
AYUDA_DER = "\n".join(["manual / postura / ritmo / auto (mano) / RL",
                       "excitacion DA, DP, BI, TRI  -/+",
                       "ref. hombro -/+, ref. codo -/+",
                       "co-contraccion -/+",
                       "ganancia refleja, retardo, reflejo on/off",
                       "tempo CPG -/+, amplitud CPG -/+",
                       "realim. sensorial on/off, g_fb -/+",
                       "pulso de torque en el codo +/-",
                       "pausa, graficos, fantasma",
                       "reiniciar / salir y guardar"])
