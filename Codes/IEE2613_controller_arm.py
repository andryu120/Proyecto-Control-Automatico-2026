import numpy as np
import mujoco

from IEE2613_mujoco_arm_model_data import *
from IEE2613_gui import *
from IEE2613_arm_utils import (parametros_musculos, tono_equilibrio,
                               BufferRetardo, reflejo_estiramiento)
from IEE2613_neural import CPGMorrisLecar, estiramiento_normalizado

prm = parametros_musculos(model)
nu  = model.nu
data_tmp = mujoco.MjData(model)  # datos auxiliares para calcular el tono

buffer_huso = BufferRetardo(Ts, retardo_max=0.3)
cpg = CPGMorrisLecar(tempo=uiVars.cpg_tempo, g_fb=uiVars.cpg_g_fb)
u0  = np.zeros(nu)
lam = np.zeros(nu)

senales = {'V': cpg.V.copy(), 'f_sens': np.zeros(2), 'pert': 0.0}

MODOS = ['manual', 'postura', 'ritmo', 'auto']  # 'auto': modo de la pregunta 15 (tecla Q)
hist = {m: {} for m in MODOS}
for modo in hist:
    for k in ['t', 'q', 'dq', 'u', 'act', 'fuerza', 'mano', 'q_ref', 'K_l', 'retardo',
              'pert', 'V', 'f_sens']:
        hist[modo][k] = []


def _registrar(modo, q_ref, pert):
    h = hist[modo]
    h['t'].append(data.time)
    h['q'].append(data.qpos.copy())
    h['dq'].append(data.qvel.copy())
    h['u'].append(data.ctrl.copy())
    h['act'].append(data.act.copy())
    h['fuerza'].append(-data.actuator_force.copy())  # >= 0 = traccion
    h['mano'].append(data.site('mano').xpos.copy())
    h['q_ref'].append(q_ref.copy())
    h['K_l'].append(uiVars.K_l)
    h['retardo'].append(uiVars.retardo)
    h['pert'].append(pert)
    h['V'].append(senales['V'].copy())
    h['f_sens'].append(senales['f_sens'].copy())


def reiniciar_controlador():
    buffer_huso.reset()
    cpg.reset()
    uiVars.ref_cambiada = True


def controller(model, data):
    global u0, lam

    buffer_huso.push(np.concatenate([data.actuator_length, data.actuator_velocity]))

    pert = uiVars.pert_signo * uiVars.pert_torque if data.time < uiVars.pert_t_fin else 0.0
    data.qfrc_applied[1] = pert
    senales['pert'] = pert

    q_ref = np.deg2rad(uiVars.q_ref_deg)
    modo = uiVars.control_mode

    if modo == 'manual':
        data.ctrl[:] = uiVars.u_manual

    elif modo in ('postura', 'ritmo'):
        if uiVars.ref_cambiada:
            q_tono = q_ref.copy()
            if modo == 'ritmo':
                q_tono[0] = 0.0  # el hombro oscila en torno a la vertical
            u0, lam, _ = tono_equilibrio(model, q_tono, prm,
                                         tuple(uiVars.cocontraccion), data_tmp)
            uiVars.ref_cambiada = False

        y = buffer_huso.get(uiVars.retardo)  # husos con retardo neural
        l_ret, v_ret = y[:nu], y[nu:]
        u = u0.copy()
        if uiVars.reflejo_on:
            u += reflejo_estiramiento(l_ret, v_ret, lam, prm,
                                      uiVars.K_l, uiVars.beta * uiVars.K_l)

        if modo == 'ritmo':
            cpg.tempo = uiVars.cpg_tempo
            cpg.g_fb  = uiVars.cpg_g_fb if uiVars.cpg_fb_on else 0.0
            f = estiramiento_normalizado(l_ret[:2], lam[:2], prm['L0'][:2])
            cpg.paso(Ts, f)
            a = cpg.salidas()
            u[0] = u0[0] + uiVars.cpg_amp * a[0]  # neurona 1 -> deltoide anterior
            u[1] = u0[1] + uiVars.cpg_amp * a[1]  # neurona 2 -> deltoide posterior
            senales['f_sens'] = f

        data.ctrl[:] = np.clip(u, 0.0, 1.0)  # los musculos solo tiran: 0 <= u <= 1

    # elif modo == 'auto':  su controlador (pregunta 15, tecla Q); la referencia de la mano es uiVars.p_ref
    #     se ejecuta cada Ts = 2 ms; para periodo T, actualice data.ctrl cada T/Ts pasos

    senales['V'] = cpg.V.copy()
    _registrar(modo, q_ref, pert)
    return None


mujoco.set_mjcb_control(controller)
