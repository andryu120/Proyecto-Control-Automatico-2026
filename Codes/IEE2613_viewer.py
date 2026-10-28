# Visor en tiempo real para simular(..., gui=True): dibuja el brazo (con la referencia como brazo "fantasma"),
# los graficos en vivo y los textos de estado (igual que la ventana de IEE2613_armsim.py).
# Teclas (se explican en el README): ESPACIO pausa, ESC cierra, E / N pulso en el codo.

import time
import numpy as np
import mujoco
from mujoco.glfw import glfw

from IEE2613_visual import colorear_musculos, agregar_vientres, agregar_fantasma, GraficosVivo

COLOR_REF = [1.0, 0.85, 0.2, 0.25]


class Visor:
    def __init__(self, model, velocidad=1.0, fps=60.0):
        self.model, self.vel, self.pausa, self.t_wall0, self.pulso_tecla = model, float(velocidad), False, None, 0
        self.cada = max(1, int(round(1.0 / fps / model.opt.timestep)))     # pasos de MuJoCo entre cuadros
        if not glfw.init():
            raise RuntimeError('No se pudo inicializar GLFW (OpenGL)')
        self.window = glfw.create_window(1280, 720, 'IEE2613 - Brazo neuromecanico (simular)', None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('No se pudo crear la ventana')
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_key_callback(self.window, self._tecla)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.opt)
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
        self.cam = mujoco.MjvCamera()
        self.cam.azimuth, self.cam.elevation, self.cam.distance = -100.0, -4.0, 1.7
        self.cam.lookat = np.array([-0.20, -0.10, 1.12])
        self.data_vis = mujoco.MjData(model)
        self.graficos = GraficosVivo(T_ventana=4.0, n_max=700)

    def _tecla(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        if act == glfw.PRESS and key == glfw.KEY_SPACE:
            self.pausa = not self.pausa
        if act == glfw.PRESS and key in (glfw.KEY_E, glfw.KEY_N):
            self.pulso_tecla = +1 if key == glfw.KEY_E else -1

    def dibujar(self, t, data, q_ref, u, aviso=None):
        """Dibuja un cuadro y espera para ir en tiempo real. Devuelve False si se cerro la ventana."""
        glfw.poll_events()
        if self.t_wall0 is None:
            self.t_wall0 = time.perf_counter() - (t / self.vel if self.vel > 0 else 0.0)
        while self.pausa and not glfw.window_should_close(self.window):
            glfw.poll_events(); time.sleep(0.03)
            self.t_wall0 = time.perf_counter() - (t / self.vel if self.vel > 0 else 0.0)   # no "recuperar" lo pausado
        if glfw.window_should_close(self.window):
            return False
        if self.vel > 0:                                                    # velocidad = 0: lo mas rapido posible
            espera = self.t_wall0 + t / self.vel - time.perf_counter()
            if espera > 0:
                time.sleep(espera)

        m = self.model
        self.graficos.agregar(t, u=u, fuerza=-data.actuator_force, q=np.rad2deg(data.qpos), q_ref=np.rad2deg(q_ref))
        colorear_musculos(m, data)
        mujoco.mjv_updateScene(m, data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        agregar_vientres(self.scene, m, data)
        agregar_fantasma(self.scene, m, self.data_vis, q_ref, COLOR_REF)
        w, h = glfw.get_framebuffer_size(self.window)
        vp = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjr_render(vp, self.scene, self.context)
        self.graficos.dibujar(vp, self.context, 'postura', False)           # 'postura' dibuja tambien las referencias
        q, qr = np.rad2deg(data.qpos), np.rad2deg(q_ref)
        izq = ["Modo", "t", "q hombro / codo", "u  DA DP BI TRI", "q_ref hombro / codo"]
        der = ["SIMULAR (su ley de control)" + ("   [PAUSA]" if self.pausa else ""), f"{t:6.2f} s",
               f"{q[0]:6.1f} / {q[1]:6.1f} deg", "  ".join(f"{x:.2f}" for x in u), f"{qr[0]:6.1f} / {qr[1]:6.1f} deg"]
        if aviso is not None:
            izq.append(">>> PERTURBACION"); der.append(f"{aviso:+.1f} N m en el codo")
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, vp,
                           "\n".join(izq), "\n".join(der), self.context)
        glfw.swap_buffers(self.window)
        return True

    def cerrar(self):
        try:
            self.context.free()
        except Exception:
            pass
        glfw.terminate()
