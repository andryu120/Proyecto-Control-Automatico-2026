from IEE2613_mujoco_arm_model_data import *  # model, data
import numpy as np


MUSCULOS = ['delt_ant', 'delt_post', 'biceps', 'triceps']


def leer_sensores(model=model, data=data):
    s = {}
    s['q']       = np.array([data.sensor('q_hombro').data[0], data.sensor('q_codo').data[0]])
    s['dq']      = np.array([data.sensor('dq_hombro').data[0], data.sensor('dq_codo').data[0]])
    s['huso_l']  = np.array([data.sensor('huso_l_' + m).data[0] for m in MUSCULOS])
    s['huso_v']  = np.array([data.sensor('huso_v_' + m).data[0] for m in MUSCULOS])
    s['golgi']   = -np.array([data.sensor('golgi_' + m).data[0] for m in MUSCULOS])  # >= 0
    p = data.sensor('pos_mano').data - data.body('brazo').xpos
    s['mano']    = np.array([p[0], p[2]])  # (x, z) con el hombro en el origen [m]
    s['act']     = data.act.copy()  # activacion muscular (estado interno)
    return s


def imprimir_sensores(model=model, data=data):
    s = leer_sensores(model, data)
    print('-- Estado del brazo --')
    print('q  [deg]      :', np.rad2deg(s['q']).round(2))
    print('dq [deg/s]    :', np.rad2deg(s['dq']).round(2))
    print('huso largo [m]:', s['huso_l'].round(4))
    print('huso vel [m/s]:', s['huso_v'].round(4))
    print('Golgi F [N]   :', s['golgi'].round(1))
    print('activacion    :', s['act'].round(3))
    print('mano [m]      :', s['mano'].round(3))
    return s


if __name__ == '__main__':
    mujoco.mj_forward(model, data)
    imprimir_sensores()
