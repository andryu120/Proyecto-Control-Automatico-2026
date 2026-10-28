import os
import mujoco
import numpy as np
from IEE2613_arm_utils import cargar_modelo

# Definicion del modelo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(BASE_DIR, "xml", "brazo_2dof_musculos.xml")


model = cargar_modelo(xml_path)  # MuJoCo model
data  = mujoco.MjData(model)  # MuJoCo data
cam   = mujoco.MjvCamera()  # Abstract camera
opt   = mujoco.MjvOption()  # Visualization options

Ts = model.opt.timestep

q_inicial = np.deg2rad([0.0, 90.0])
data.qpos[:] = q_inicial

cam.azimuth   = -100.0
cam.elevation = -4.0
cam.distance  = 1.7
cam.lookat    = np.array([-0.20, -0.10, 1.12])

# Estructuras de visualizacion
mujoco.mjv_defaultOption(opt)
opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
scene = mujoco.MjvScene(model, maxgeom=10000)
scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False  # sin piso no hay sombras que mostrar

mujoco.mj_forward(model, data)


if __name__ == '__main__':
    print("Modelo de MuJoCo cargado correctamente como script principal.")
    print("nq =", model.nq, " nu =", model.nu, " na =", model.na, " Ts =", Ts)
    for i in range(model.nu):
        print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
