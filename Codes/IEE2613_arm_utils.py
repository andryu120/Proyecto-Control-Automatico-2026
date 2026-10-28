import os
import numpy as np
import mujoco
from collections import deque


# Carga el modelo de MuJoCo desde el XML
def cargar_modelo(xml_path):
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(
            f"No se encontro el modelo: {xml_path}\n"
            "Revisa que la carpeta 'xml' (con brazo_2dof_musculos.xml) este "
            "junto a los archivos .py.")
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml = f.read()
    cb = mujoco.get_mjcb_control()  # la compilacion simula: sin controlador
    mujoco.set_mjcb_control(None)
    try:
        return mujoco.MjModel.from_xml_string(xml)
    finally:
        mujoco.set_mjcb_control(cb)


NOMBRES = ['deltoide_ant', 'deltoide_post', 'biceps', 'triceps']
NOMBRES_CORTOS = ['DA', 'DP', 'BI', 'TRI']
PARES = [(0, 1), (2, 3)]  # (flexor, extensor) de hombro y codo
JUNTA = np.array([0, 0, 1, 1])
SIGNO = np.array([+1, -1, +1, -1])  # +1 flexor, -1 extensor


# mj_forward sin ejecutar el controlador registrado
def forward_sin_callback(model, data):
    cb = mujoco.get_mjcb_control()
    mujoco.set_mjcb_control(None)
    try:
        mujoco.mj_forward(model, data)
    finally:
        mujoco.set_mjcb_control(cb)


# Matriz de brazos de momento R(q) = dl/dq [m]
def matriz_momentos(model, data):
    R = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(R, data.actuator_moment, data.moment_rownnz,
                            data.moment_rowadr, data.moment_colind)
    return R


# Brazo de momento de cada musculo [m]
def brazos_momento(model, data):
    R = matriz_momentos(model, data)
    return np.abs(R[np.arange(model.nu), JUNTA])


# F0, L0, articulacion y signo de cada musculo
def parametros_musculos(model):
    F0  = model.actuator_gainprm[:, 2].copy()  # fuerza isometrica max [N]
    lr  = model.actuator_lengthrange.copy()  # rango de largos [m]
    rng = model.actuator_gainprm[:, 0:2].copy()  # rango operativo en L/L0
    L0  = (lr[:, 1] - lr[:, 0]) / (rng[:, 1] - rng[:, 0])  # largo optimo [m]
    return dict(F0=F0, L0=L0, junta=JUNTA.copy(), signo=SIGNO.copy())


# Fuerza por unidad de activacion F0*F_L y fuerza pasiva [N]
def ganancia_estatica(model, data, vel=None):
    G  = np.zeros(model.nu)
    Fp = np.zeros(model.nu)
    for i in range(model.nu):
        v = 0.0 if vel is None else vel[i]
        g = mujoco.mju_muscleGain(data.actuator_length[i], v,
                                  model.actuator_lengthrange[i],
                                  model.actuator_acc0[i],
                                  model.actuator_gainprm[i, :9])
        b = mujoco.mju_muscleBias(data.actuator_length[i],
                                  model.actuator_lengthrange[i],
                                  model.actuator_acc0[i],
                                  model.actuator_biasprm[i, :9])
        G[i]  = -g  # MuJoCo usa fuerza negativa = traccion
        Fp[i] = -b
    return G, Fp


B_NOMINAL = np.array([40.00, 30.88, 33.07, 23.66])  # [DA, DP, BI, TRI] en q0 [N m]


# Reparte el torque deseado [hombro, codo] entre los musculos -> u = [DA, DP, BI, TRI]
def asignacion_muscular(tau_des, B=B_NOMINAL, cocontraccion=0.05):
    kappa = np.broadcast_to(np.asarray(cocontraccion, dtype=float), (2,))
    u = np.zeros(4)
    for j, (f, e) in enumerate(PARES):
        T_cc = kappa[j] * min(B[f], B[e])
        u[f] = (T_cc + max(tau_des[j], 0.0)) / B[f]
        u[e] = (T_cc + max(-tau_des[j], 0.0)) / B[e]
    return np.clip(u, 0.0, 1.0)


# Torque por unidad de activacion B = r*F0*F_L en la postura de data [N m]
def ganancias_musculares(model, data):
    G, _ = ganancia_estatica(model, data)
    return brazos_momento(model, data) * G


# Tono que sostiene la postura q_ref (modelo completo) -> u0, largos lambda, gravedad
def tono_equilibrio(model, q_ref, prm=None, cocontraccion=0.05, data_tmp=None):
    if data_tmp is None:
        data_tmp = mujoco.MjData(model)
    data_tmp.qpos[:] = q_ref
    data_tmp.qvel[:] = 0.0
    data_tmp.act[:]  = 0.0  # activacion nula -> solo fuerza pasiva
    data_tmp.ctrl[:] = 0.0
    forward_sin_callback(model, data_tmp)
    tau_grav = data_tmp.qfrc_bias.copy()  # torque para sostener la gravedad
    tau_pas  = data_tmp.qfrc_actuator.copy()  # torque de las fuerzas pasivas
    B = ganancias_musculares(model, data_tmp)  # B en la postura q_ref (modelo completo)
    u0 = asignacion_muscular(tau_grav - tau_pas, B, cocontraccion)
    lam = data_tmp.actuator_length.copy()
    return u0, lam, tau_grav


def _datos_aux(model, data_tmp):
    return mujoco.MjData(model) if data_tmp is None else data_tmp


# Torque de los musculos sobre [hombro, codo] [N m]
def torque_muscular(model, q, dq, a, data_tmp=None):
    d = _datos_aux(model, data_tmp)
    d.qpos[:] = q; d.qvel[:] = dq; d.act[:] = a; d.ctrl[:] = a
    forward_sin_callback(model, d)
    return d.qfrc_actuator.copy()


# Modelo completo dx/dt = f(x, u), x = [q, dq, a]
def dinamica(model, x, u, data_tmp=None):
    d = _datos_aux(model, data_tmp)
    d.qpos[:] = x[0:2]; d.qvel[:] = x[2:4]; d.act[:] = x[4:8]; d.ctrl[:] = u
    forward_sin_callback(model, d)
    return np.concatenate([d.qvel, d.qacc, d.act_dot])


_NOMINAL = {}  # propiedades originales, para revertir


def _guardar_nominal(model):
    k = id(model)
    if k not in _NOMINAL:
        b = model.body('antebrazo').id
        _NOMINAL[k] = dict(F0=model.actuator_gainprm[:, 2].copy(), masa=model.body_mass[b].copy(),
                           ipos=model.body_ipos[b].copy(), inertia=model.body_inertia[b].copy(),
                           iquat=model.body_iquat[b].copy())
    return _NOMINAL[k]


# Agrega una masa [kg] en la mano; masa = 0 la quita
def agregar_carga(model, masa=1.0):
    nom = _guardar_nominal(model)
    b = model.body('antebrazo').id
    p = model.site('mano').pos.copy()  # posicion de la mano en el marco del antebrazo
    m1, c1 = float(nom['masa']), nom['ipos']
    R = np.zeros(9); mujoco.mju_quat2Mat(R, nom['iquat']); R = R.reshape(3, 3)
    I1 = R @ np.diag(nom['inertia']) @ R.T  # inercia del antebrazo en su centro de masa
    # nuevo centro de masa e inercia del antebrazo (Steiner)
    mt = m1 + masa
    c = (m1 * c1 + masa * p) / mt
    def steiner(mm, dvec):
        return mm * (dvec @ dvec * np.eye(3) - np.outer(dvec, dvec))
    I = I1 + steiner(m1, c1 - c) + steiner(masa, p - c)
    w, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    q = np.zeros(4); mujoco.mju_mat2Quat(q, V.flatten())
    model.body_mass[b], model.body_ipos[b], model.body_inertia[b], model.body_iquat[b] = mt, c, w, q
    cb = mujoco.get_mjcb_control()
    mujoco.set_mjcb_control(None)
    try:
        mujoco.mj_setConst(model, mujoco.MjData(model))
    finally:
        mujoco.set_mjcb_control(cb)


# Fuerza muscular = factor * F0 nominal; factor = 1 vuelve al musculo sano
def aplicar_sarcopenia(model, factor=0.5):
    nom = _guardar_nominal(model)
    model.actuator_gainprm[:, 2] = factor * nom['F0']
    model.actuator_biasprm[:, 2] = factor * nom['F0']


TAU_ACTIVACION = 0.03  # [s]


# Activacion asimetrica real (activar=True) o lineal de 30 ms (activar=False)
def activacion_fisiologica(model, activar=True, tau_act=0.01, tau_deact=0.04):
    if activar:
        model.actuator_dyntype[:] = mujoco.mjtDyn.mjDYN_MUSCLE
        model.actuator_dynprm[:, 0:3] = [tau_act, tau_deact, 0.0]
    else:
        model.actuator_dyntype[:] = mujoco.mjtDyn.mjDYN_FILTER
        model.actuator_dynprm[:, 0:3] = [TAU_ACTIVACION, 0.0, 0.0]


# Entrega mediciones con un retardo dado
class BufferRetardo:

    def __init__(self, Ts, retardo_max=0.3):
        self.Ts = Ts
        self.buf = deque(maxlen=int(np.ceil(retardo_max / Ts)) + 2)

    def push(self, x):
        self.buf.append(np.array(x, dtype=float))

    def get(self, retardo):
        n = int(round(retardo / self.Ts))
        if len(self.buf) == 0:
            return None
        n = min(n, len(self.buf) - 1)
        return self.buf[-1 - n]

    def reset(self):
        self.buf.clear()


# Excitacion del reflejo de estiramiento a partir de largos y velocidades retardados
def reflejo_estiramiento(l_ret, v_ret, lam, prm, K_l, K_v):
    L0 = prm['L0']
    return K_l * (l_ret - lam) / L0 + K_v * v_ret / L0


# Rigidez y amortiguamiento articular equivalentes del reflejo
def rigidez_articular_refleja(G, r, prm, K_l, K_v):
    Kp = np.zeros(2)
    Kd = np.zeros(2)
    for i in range(len(G)):
        j = prm['junta'][i]
        Kp[j] += r[i] ** 2 * G[i] * K_l / prm['L0'][i]
        Kd[j] += r[i] ** 2 * G[i] * K_v / prm['L0'][i]
    return Kp, Kd


_MODELO = {}  # modelo que usa simular() si no se le entrega uno
Q0 = np.deg2rad([0.0, 90.0])  # postura nominal [rad]


def _modelo_por_defecto():
    if 'm' not in _MODELO:
        _MODELO['m'] = cargar_modelo(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xml', 'brazo_2dof_musculos.xml'))
    return _MODELO['m']


# Simula el brazo real (MuJoCo) en lazo cerrado con su ley de control -> t, q, u
def simular(ley, T=0.002, q_ref=Q0, duracion=3.0, salida='torque', retardo=0.030,
            pulso=0.0, t_pulso=1.0, ruido_sensor=0.0, ruido_excitacion=0.0,
            carga=0.0, sarcopenia=1.0, activacion_fisio=False, q_inicial=Q0, semilla=0, model=None,
            gui=False, velocidad=1.0):
    #   ley(t, y, r): y = angulos medidos con retardo [rad], r = referencia [rad] (ambos [hombro, codo]).
    #     salida='torque'     -> devuelve el torque deseado [hombro, codo] [N m] (se usa asignacion_muscular)
    #     salida='excitacion' -> devuelve las excitaciones [DA, DP, BI, TRI]
    #   T: periodo del controlador [s] (multiplo de 2 ms); u se mantiene constante entre actualizaciones (ZOH)
    #   q_ref: referencia constante o funcion q_ref(t) [rad]
    #   pulso: torque en el codo [N m] durante 50 ms desde t_pulso
    #   ruido_sensor [rad] y ruido_excitacion: amplitud de un ruido uniforme de media cero
    #   gui=True: abre una ventana y muestra la simulacion en tiempo real (brazo, referencia, graficos);
    #             velocidad = 1 tiempo real, 0.5 camara lenta, 0 lo mas rapido posible. ESPACIO pausa, ESC cierra,
    #             E / N perturban el codo con un pulso de +3 / -3 N m (50 ms);
    #             los pulsos aplicados quedan en simular.perturbaciones = [(t, N m), ...]
    model = _modelo_por_defecto() if model is None else model
    Ts = model.opt.timestep
    paso_c, paso_r = T / Ts, retardo / Ts
    if abs(paso_c - round(paso_c)) > 1e-9 or abs(paso_r - round(paso_r)) > 1e-9:
        raise ValueError(f'T y el retardo deben ser multiplos de Ts = {Ts*1e3:.0f} ms')
    paso_c, paso_r = int(round(paso_c)), int(round(paso_r))
    ref = q_ref if callable(q_ref) else (lambda t, r0=np.array(q_ref, float): r0)
    rng = np.random.default_rng(semilla)
    data = mujoco.MjData(model)
    visor = None
    cb = mujoco.get_mjcb_control()  # sin el controlador del simulador
    mujoco.set_mjcb_control(None)
    try:
        u = tono_equilibrio(model, q_inicial, data_tmp=mujoco.MjData(model))[0]  # brazo en reposo con su tono
        agregar_carga(model, carga); aplicar_sarcopenia(model, sarcopenia); activacion_fisiologica(model, activacion_fisio)
        if gui:
            from IEE2613_viewer import Visor
            visor = Visor(model, velocidad)
        data.qpos[:] = q_inicial; data.act[:] = u; data.ctrl[:] = u; mujoco.mj_forward(model, data)
        n = int(round(duracion / Ts))
        hist = [np.array(q_inicial, float)] * (paso_r + 1)  # mediciones pasadas (retardo)
        t_out, q_out, u_out = np.arange(n) * Ts, np.zeros((n, 2)), np.zeros((n, 4))
        n_ok = n
        pulsos = [(t_pulso, pulso)] if pulso != 0.0 else []  # (instante, torque) de cada pulso en el codo
        for i in range(n):
            t = i * Ts
            if i % paso_c == 0:  # actualizacion del controlador
                y = hist[-1 - paso_r] + rng.uniform(-ruido_sensor, ruido_sensor, 2)
                v = np.asarray(ley(t, y, np.asarray(ref(t), float)), float).reshape(-1)
                if v.size != (2 if salida == 'torque' else 4):
                    raise ValueError(f"ley debe devolver {2 if salida == 'torque' else 4} valores con salida='{salida}', devolvio {v.size}")
                u = asignacion_muscular(v) if salida == 'torque' else v
                u = np.clip(u + rng.uniform(-ruido_excitacion, ruido_excitacion, 4), 0.0, 1.0)
            q_out[i], u_out[i] = data.qpos, u
            if visor is not None and i % visor.cada == 0:
                recientes = [a for tp, a in pulsos if tp <= t < tp + 1.5]
                if not visor.dibujar(t, data, np.asarray(ref(t), float), u, recientes[-1] if recientes else None):
                    n_ok = i  # se cerro la ventana: termina la simulacion
                    break
                if visor.pulso_tecla:  # tecla E / N: pulso de +-3 N m en el codo
                    pulsos.append((t, 3.0 * visor.pulso_tecla)); visor.pulso_tecla = 0
            data.ctrl[:] = u
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[1] = sum(a for tp, a in pulsos if tp <= t < tp + 0.05)
            mujoco.mj_step(model, data)
            hist.append(data.qpos.copy()); hist = hist[-(paso_r + 2):]
        simular.perturbaciones = list(pulsos)  # [(t, N m), ...] pulsos aplicados (programados y con E / N)
    finally:
        agregar_carga(model, 0.0); aplicar_sarcopenia(model, 1.0); activacion_fisiologica(model, False)
        mujoco.set_mjcb_control(cb)
        if visor is not None:
            visor.cerrar()
    return t_out[:n_ok], q_out[:n_ok], u_out[:n_ok]
