import numpy as np
import control as ct
import matplotlib.pyplot as plt
from parametros import (BCF,JC,t_a,BC,KC)

pm = np.polymul

A = [t_a, 1]
B = [JC, BC, KC]

mult = pm(A,B)

num = [BCF]
den = [mult[0], mult[1], mult[2], mult[3]]

Gc= ct.tf(num, den)

Mec = ct.tf([BCF], B)

print( ct.bandwidth(Gc) )
ct.bode(Gc, dB=True)
ct.bode(Mec, dB=True)

plt.show()


