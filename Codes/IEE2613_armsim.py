import numpy as np

from IEE2613_mujoco_arm_model_data import *
from IEE2613_gui import *
from IEE2613_controller_arm import *
from IEE2613_mujoco_sensors import *
from IEE2613_visual import (colorear_musculos, agregar_vientres, agregar_fantasma, GraficosVivo,
                            texto_estado, AYUDA_IZQ, AYUDA_DER)

FPS = 60.0
PASOS_POR_MUESTRA = 3  # muestra para graficos cada 3*Ts = 6 ms

uiVars.reset_hooks.append(reiniciar_controlador)


def main():
    window = glfw_init()
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    data_vis = mujoco.MjData(model)  # para dibujar el brazo fantasma
    graficos = GraficosVivo(T_ventana=4.0, n_max=700)
    uiVars.reset_hooks.append(graficos.reset)
    k_paso = 0

    while not glfw.window_should_close(window):
        glfw.poll_events()

        if not uiVars.pausa:
            t_objetivo = data.time + 1.0 / FPS
            while data.time < t_objetivo:
                mujoco.mj_step(model, data)
                k_paso += 1
                if k_paso % PASOS_POR_MUESTRA == 0:
                    graficos.agregar(data.time, V=senales['V'], u=data.ctrl,
                                     fuerza=-data.actuator_force,
                                     q=np.rad2deg(data.qpos), q_ref=uiVars.q_ref_deg)

        colorear_musculos(model, data)
        mujoco.mjv_updateScene(model, data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        agregar_vientres(scene, model, data)
        if uiVars.ver_fantasma and uiVars.control_mode == 'postura':
            agregar_fantasma(scene, model, data_vis, np.deg2rad(uiVars.q_ref_deg),
                             uiVars.control_mode_color['postura'])

        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjr_render(viewport, scene, context)
        if uiVars.ver_graficos:
            graficos.dibujar(viewport, context, uiVars.control_mode,
                             mostrar_neuronas=(uiVars.control_mode == 'ritmo'))
        izq, der = texto_estado(data, uiVars)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           viewport, izq, der, context)
        if uiVars.ver_ayuda:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                               viewport, AYUDA_IZQ, AYUDA_DER, context)
        glfw.swap_buffers(window)

    # Guardar historiales
    for modo, hh in hist.items():
        if len(hh['t']) > 0:
            np.savez(f'datos_{modo}.npz', **{k: np.array(v) for k, v in hh.items()})
            print(f"Datos guardados en datos_{modo}.npz ({len(hh['t'])} muestras)")
        else:
            print(f"datos_{modo}.npz NO sobreescrito (no se uso el modo '{modo}')")
    glfw.terminate()


if __name__ == '__main__':
    main()
