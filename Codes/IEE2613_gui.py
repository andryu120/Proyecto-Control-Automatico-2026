from mujoco.glfw import glfw  # MuJoCo incluye glfw
import OpenGL.GL as gl
import mujoco
import numpy as np

from IEE2613_mujoco_arm_model_data import *  # model, data, cam, scene, q_inicial


class uiVars:  # User Interface Variables
    control_mode = 'postura'  # 'manual', 'postura', 'ritmo' o 'auto'
    control_mode_color = {'manual': [0.2, 0.9, 0.9, 0.25],
                          'postura': [1.0, 0.85, 0.2, 0.25],
                          'ritmo': [0.6, 0.4, 1.0, 0.25]}

    # Modo manual (lazo abierto)
    u_manual = np.zeros(4)  # excitaciones [0..1] de los 4 musculos
    delta_u  = 0.02

    # Modo postura (lazo cerrado)
    q_ref_deg     = np.rad2deg(q_inicial).copy()  # [hombro, codo] en grados
    delta_q       = 10.0
    cocontraccion = np.array([0.05, 0.05])
    delta_cc      = 0.02
    reflejo_on    = True
    K_l           = 3.0  # ganancia refleja de largo
    beta          = 0.05  # K_v = beta*K_l  [s]
    retardo       = 0.030  # retardo neural [s]
    delta_ret     = 0.005

    # Modo ritmo (CPG)
    cpg_tempo     = 0.28  # ms neurales por ms reales (periodo ~1.15 s)
    cpg_amp       = 0.05  # excitacion maxima que agrega el CPG
    cpg_g_fb      = 20.0  # inhibicion desde el huso contrario
    cpg_fb_on     = True

    # Perturbacion externa
    pert_torque   = 3.0  # [Nm] en el codo
    pert_duracion = 0.05  # [s]
    pert_t_fin    = -1.0  # instante en que termina el pulso activo
    pert_signo    = 0.0

    # Visualizacion
    pausa         = False
    ver_graficos  = True
    ver_ayuda     = False
    ver_fantasma  = True

    p_ref         = np.array([0.32, -0.30])  # referencia de la mano (x, z) [m], modo 'auto'
    delta_p       = 0.02  # [m]
    ref_cambiada  = True  # avisa al controlador que recalcule el tono
    reset_hooks   = []  # funciones a llamar al reiniciar (controlador)


# Manejadores (handlers) de callbacks de GLFW
mouse_button_left   = False
mouse_button_middle = False
mouse_button_right  = False
mouse_lastx = 0
mouse_lasty = 0


def keyboard(window, key, scancode, act, mods):

    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mujoco.mj_resetData(model, data)
        data.qpos[:] = q_inicial
        mujoco.mj_forward(model, data)
        uiVars.q_ref_deg = np.rad2deg(q_inicial).copy()
        uiVars.ref_cambiada = True
        for f in uiVars.reset_hooks:
            f()
        print('Reset')

    if act == glfw.PRESS:
        if key == glfw.KEY_ESCAPE:
            print('Bye!')
            glfw.set_window_should_close(window, True)

        # cambio de modo
        if key == glfw.KEY_M:
            uiVars.control_mode = 'manual'
            uiVars.u_manual = np.array(data.ctrl).copy()  # transferencia sin salto
            print('Modo MANUAL (lazo abierto) u =', uiVars.u_manual.round(3))
        if key == glfw.KEY_P:
            uiVars.control_mode = 'postura'
            uiVars.ref_cambiada = True
            print('Modo POSTURA (tono + reflejo)')
        if key == glfw.KEY_O:
            uiVars.control_mode = 'ritmo'
            uiVars.ref_cambiada = True
            print('Modo RITMO (CPG Morris-Lecar en el hombro + reflejo en el codo)')

        # visualizacion
        if key == glfw.KEY_SPACE:
            uiVars.pausa = not uiVars.pausa
        if key == glfw.KEY_G:
            uiVars.ver_graficos = not uiVars.ver_graficos
        if key == glfw.KEY_H:
            uiVars.ver_ayuda = not uiVars.ver_ayuda
        if key == glfw.KEY_W:
            uiVars.ver_fantasma = not uiVars.ver_fantasma

        # modo ritmo (CPG)
        if key == glfw.KEY_7:
            uiVars.cpg_tempo /= 1.1
            print(f'tempo CPG = {uiVars.cpg_tempo:.3f}')
        if key == glfw.KEY_8:
            uiVars.cpg_tempo *= 1.1
            print(f'tempo CPG = {uiVars.cpg_tempo:.3f}')
        if key in (glfw.KEY_9, glfw.KEY_0):
            sgn = -1 if key == glfw.KEY_9 else +1
            uiVars.cpg_amp = float(np.clip(uiVars.cpg_amp + sgn * 0.01, 0.0, 0.3))
            print(f'amplitud CPG = {uiVars.cpg_amp:.2f}')
        if key == glfw.KEY_I:
            uiVars.cpg_fb_on = not uiVars.cpg_fb_on
            print('Realimentacion sensorial al CPG', 'ON' if uiVars.cpg_fb_on else 'OFF')
        if key == glfw.KEY_J:
            uiVars.cpg_g_fb /= 1.5
            print(f'g_fb = {uiVars.cpg_g_fb:.1f}')
        if key == glfw.KEY_T:
            uiVars.cpg_g_fb *= 1.5
            print(f'g_fb = {uiVars.cpg_g_fb:.1f}')

        # modo postura
        cambios = {glfw.KEY_1: (0, -1), glfw.KEY_2: (0, +1),
                   glfw.KEY_3: (1, -1), glfw.KEY_4: (1, +1)}
        if key in cambios:
            j, sgn = cambios[key]
            uiVars.q_ref_deg[j] += sgn * uiVars.delta_q
            lo, hi = np.rad2deg(model.jnt_range[j])
            uiVars.q_ref_deg[j] = np.clip(uiVars.q_ref_deg[j], lo, hi)
            uiVars.ref_cambiada = True
            print(f'q_ref = hombro {uiVars.q_ref_deg[0]:.0f} deg, codo {uiVars.q_ref_deg[1]:.0f} deg')
        if key in (glfw.KEY_5, glfw.KEY_6):
            sgn = -1 if key == glfw.KEY_5 else +1
            uiVars.cocontraccion = np.clip(uiVars.cocontraccion + sgn * uiVars.delta_cc, 0.0, 0.5)
            uiVars.ref_cambiada = True
            print('co-contraccion =', uiVars.cocontraccion.round(2))
        if key == glfw.KEY_K:
            uiVars.K_l /= 1.25
            print(f'K_l = {uiVars.K_l:.2f}')
        if key == glfw.KEY_L:
            uiVars.K_l *= 1.25
            print(f'K_l = {uiVars.K_l:.2f}')
        if key == glfw.KEY_Y:
            uiVars.retardo = max(0.0, uiVars.retardo - uiVars.delta_ret)
            print(f'retardo = {uiVars.retardo*1e3:.0f} ms')
        if key == glfw.KEY_U:
            uiVars.retardo = min(0.25, uiVars.retardo + uiVars.delta_ret)
            print(f'retardo = {uiVars.retardo*1e3:.0f} ms')
        if key == glfw.KEY_R:
            uiVars.reflejo_on = not uiVars.reflejo_on
            print('Reflejo', 'ACTIVADO' if uiVars.reflejo_on else 'DESACTIVADO (solo tono)')

        # perturbacion
        if key in (glfw.KEY_E, glfw.KEY_N):
            uiVars.pert_signo = +1.0 if key == glfw.KEY_E else -1.0
            uiVars.pert_t_fin = data.time + uiVars.pert_duracion
            print('Perturbacion', '+' if key == glfw.KEY_E else '-')

    # modo manual (admite mantener presionado)
    if act == glfw.REPEAT or act == glfw.PRESS:
        teclas_man = {glfw.KEY_A: (0, -1), glfw.KEY_Z: (0, +1),  # deltoide anterior
                      glfw.KEY_S: (1, -1), glfw.KEY_X: (1, +1),  # deltoide posterior
                      glfw.KEY_D: (2, -1), glfw.KEY_C: (2, +1),  # biceps
                      glfw.KEY_F: (3, -1), glfw.KEY_V: (3, +1)}  # triceps
        if key in teclas_man:
            i, sgn = teclas_man[key]
            uiVars.u_manual[i] = np.clip(uiVars.u_manual[i] + sgn * uiVars.delta_u, 0.0, 1.0)
            if uiVars.control_mode == 'manual':
                print('u_manual =', uiVars.u_manual.round(3))


def mouse_button(window, button, act, mods):
    global mouse_button_left, mouse_button_middle, mouse_button_right
    mouse_button_left   = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)   == glfw.PRESS)
    mouse_button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    mouse_button_right  = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT)  == glfw.PRESS)


def mouse_scroll(window, xoffset, yoffset):
    action = mujoco.mjtMouse.mjMOUSE_ZOOM
    mujoco.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)


def mouse_move(window, xpos, ypos):
    global mouse_lastx, mouse_lasty
    dx = xpos - mouse_lastx
    dy = ypos - mouse_lasty
    mouse_lastx = xpos
    mouse_lasty = ypos
    if (not mouse_button_left) and (not mouse_button_middle) and (not mouse_button_right):
        return
    width, height = glfw.get_window_size(window)
    mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
    if mouse_button_right:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
    elif mouse_button_left:
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM
    mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)


# Inicializacion de la ventana GLFW
def glfw_init():
    width, height = 1280, 720
    window_name = 'IEE2613 - Brazo neuromecanico (MuJoCo + reflejo + CPG)'
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    if not window:
        glfw.terminate()
        print("Could not initialize GLFW window")
        exit(1)
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # v-sync
    glfw.set_key_callback(window, keyboard)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_scroll_callback(window, mouse_scroll)
    return window
