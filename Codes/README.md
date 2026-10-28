# Simulador del brazo neuromecánico (IEE2613)

Brazo de dos articulaciones (hombro y codo) movido por cuatro músculos, simulado con MuJoCo.
Hace el papel del **brazo real** del proyecto, donde sus resultados se comparan con los del modelo del enunciado.

## Instalación

```
pip install mujoco control glfw pyopengl
```

Probado con Python 3.12 y MuJoCo 3.16.

## Dos formas de usarlo

1. **`simular` (la que usará en el proyecto):** escribe su ley de control como una función y `simular` corre el brazo real. Devuelve los resultados para graficarlos, y con `gui=True` además los muestra en una ventana en tiempo real (ver más abajo).
2. **La ventana del simulador** (`python IEE2613_armsim.py`): para explorar el brazo con el teclado (modos manual, postura con reflejo y ritmo). Los controladores del proyecto **no se programan aquí**.

## La ventana del simulador

```
python IEE2613_armsim.py
```

Abre la ventana. Al cerrarla (tecla `ESC`), guarda los datos de cada modo usado en `datos_<modo>.npz`.

| Tecla | Acción |
|---|---|
| `m` | Modo manual (lazo abierto) |
| `a`/`z`, `s`/`x`, `d`/`c`, `f`/`v` | Excitación −/+ del deltoide anterior, deltoide posterior, bíceps y tríceps (modo manual) |
| `p` | Modo postura: tono + reflejo de estiramiento (controlador de referencia) |
| `1`/`2`, `3`/`4` | Referencia −/+ del hombro y del codo (modo postura) |
| `5`/`6` | Co-contracción −/+ |
| `k`/`l`, `y`/`u`, `r` | Ganancia del reflejo −/+, retardo neural −/+, reflejo encendido/apagado (modo postura) |
| `e`/`n` | Pulso de perturbación en el codo: +3 o −3 N·m durante 50 ms |
| `o` | Modo ritmo (demostración, no se evalúa) |
| `ESPACIO`, `g`, `w`, `h` | Pausa, gráficos en vivo, brazo de referencia, ayuda en pantalla |
| `BACKSPACE`, `ESC` | Reiniciar; salir y guardar datos |

## Archivos

| Archivo | Contenido |
|---|---|
| `IEE2613_armsim.py` | Programa principal (ventana y bucle de simulación) |
| `IEE2613_controller_arm.py` | Controlador (`mjcb_control`) de la ventana, con los modos manual, postura y ritmo |
| `IEE2613_gui.py` | Teclado y variables de la interfaz (`uiVars`) |
| `IEE2613_arm_utils.py` | Funciones del modelo, incluida `simular` (ver abajo) |
| `IEE2613_viewer.py` | Ventana en tiempo real de `simular(..., gui=True)` |
| `IEE2613_mujoco_sensors.py` | `leer_sensores()`: ángulos, velocidades, husos, Golgi y posición de la mano (x, z), con el hombro en el origen como en el enunciado |
| `xml/brazo_2dof_musculos.xml` | Modelo del brazo para MuJoCo |
| Otros `IEE2613_*.py` | Carga del modelo, dibujo y oscilador del modo ritmo |
| `demo_capsula.py` | Demo de la cápsula: `simular` en lazo abierto, vista en tiempo real (`gui=True`) y trayectoria de la mano |
| `controlador_rl.py`, `controlador_rl.pt`, `evaluar_rl.py` | Controlador entrenado con RL y ejemplos de uso (bonus, pregunta 16) |

## Funciones útiles (`IEE2613_arm_utils.py`)

| Función | Qué hace |
|---|---|
| `simular(ley, T, q_ref, duracion, ...)` | **Simula el brazo real con su ley de control** (ver abajo) |
| `cargar_modelo(ruta_xml)` | Carga el modelo de MuJoCo |
| `asignacion_muscular(tau, B=B_NOMINAL, cocontraccion=0.05)` | Reparte el torque deseado `[hombro, codo]` entre los cuatro músculos (ecuación de asignación del enunciado). Devuelve `u = [DA, DP, BI, TRI]` |
| `B_NOMINAL` | Los torques por unidad de activación `[DA, DP, BI, TRI]` de la Tabla 1 |
| `tono_equilibrio(model, q)` | Tono que sostiene la postura `q` según el **modelo completo** de MuJoCo (pregunta 2c). Devuelve `(u0, largos, torque_gravedad)`; el tono es `tono_equilibrio(model, q)[0]` |
| `activacion_fisiologica(model)` | Cambia a la activación asimétrica real; `activacion_fisiologica(model, False)` vuelve a la nominal |
| `agregar_carga(model, masa)` | Agrega una masa [kg] en la mano; `agregar_carga(model, 0)` la quita |
| `aplicar_sarcopenia(model, factor)` | Fuerza muscular a `factor` veces la nominal; `aplicar_sarcopenia(model, 1)` vuelve al músculo sano |
| `BufferRetardo` | Entrega una medición con el retardo que se le indique |

Con `simular` no use estas funciones: las condiciones se piden con sus argumentos `carga`, `sarcopenia` y `activacion_fisio`, que se aplican solo durante esa simulación. Las funciones `agregar_carga`, `aplicar_sarcopenia` y `activacion_fisiologica` sirven si trabaja con su propio `model` de MuJoCo; en ese caso, llame a `mujoco.mj_forward(model, data)` después.

## Simular el brazo con su controlador: `simular`

Para todas las preguntas no necesita programar MuJoCo: escriba su ley de control y use `simular`.

```python
import numpy as np
from IEE2613_arm_utils import simular

q0 = np.deg2rad([0.0, 90.0])
tau_eq = np.array([2.904, 2.904])                  # torque que sostiene q0 [N m]

def ley(t, y, r):                                  # y: angulos medidos con retardo; r: referencia [rad]
    return tau_eq + 5.0 * (r - y)                  # torque deseado [hombro, codo] (aqui, un proporcional)

t, q, u = simular(ley, T=0.010, q_ref=q0 + np.deg2rad([0, 5]), duracion=3.0)
print(np.rad2deg(q[-1]))                           # angulos al final [grados]
```

`simular` hace lo que haría el brazo real: entrega las mediciones con 30 ms de retardo, ejecuta su ley cada `T` segundos y mantiene `u` constante entre actualizaciones (retención de orden cero). Devuelve el tiempo `t`, los ángulos `q` (en radianes) y las excitaciones `u`. El brazo parte en reposo en `q_inicial`, sostenido por su tono.

| Argumento | Qué es |
|---|---|
| `ley(t, y, r)` | Su controlador. Devuelve el torque deseado `[hombro, codo]` en N·m, o las excitaciones `[DA, DP, BI, TRI]` si `salida='excitacion'` |
| `T` | Período del controlador [s], múltiplo de 2 ms |
| `q_ref` | Referencia `[hombro, codo]` en radianes: un valor fijo o una función `q_ref(t)` |
| `duracion` | Tiempo simulado [s] |
| `pulso`, `t_pulso` | Torque en el codo [N·m] durante 50 ms desde `t_pulso` |
| `ruido_sensor`, `ruido_excitacion` | Amplitud de un ruido uniforme de media cero (en rad y en excitación) |
| `carga`, `sarcopenia`, `activacion_fisio` | Masa en la mano [kg], factor de fuerza muscular, activación asimétrica (`True`) |
| `retardo` | Retardo neural [s] (por defecto, 0.030) |
| `gui`, `velocidad` | `gui=True` muestra la simulación en una ventana en tiempo real; `velocidad` es 1 en tiempo real, 0.5 en cámara lenta, 0 lo más rápido posible |

Para controlar directamente las excitaciones (preguntas 5 y 8), use `salida='excitacion'` y devuelva las cuatro `[DA, DP, BI, TRI]`:

```python
from IEE2613_arm_utils import tono_equilibrio, cargar_modelo
u0 = tono_equilibrio(cargar_modelo('xml/brazo_2dof_musculos.xml'), q0)[0]   # tono [DA, DP, BI, TRI]
ley = lambda t, y, r: u0 + np.array([0, 0, 0.5 * (r[1] - y[1]), 0])        # proporcional sobre el bíceps
t, q, u = simular(ley, T=0.002, q_ref=q0 + np.deg2rad([0, 5]), duracion=3.0, salida='excitacion')
```

Si su controlador tiene memoria (por ejemplo, un integrador), guárdela fuera de la función o use una clase, y cree un controlador nuevo para cada simulación.

### Ver su controlador en tiempo real: `gui=True`

Agregue `gui=True` a `simular` y se abre una ventana con el brazo, la referencia (brazo translúcido) y gráficos en vivo de excitaciones, fuerzas y ángulos:

```python
t, q, u = simular(ley, T=0.010, q_ref=q_ref, duracion=6.0, gui=True)                    # tiempo real
t, q, u = simular(ley, T=0.010, q_ref=q_ref, duracion=6.0, gui=True, velocidad=0.5)     # cámara lenta
```

| Tecla | Acción |
|---|---|
| `ESPACIO` | Pausa |
| `e` / `n` | Perturba el codo con un pulso de +3 / −3 N·m (50 ms), sin programar `pulso` |
| `ESC` o cerrar la ventana | Termina la simulación y devuelve lo calculado hasta ese instante |

Los resultados son los mismos con y sin `gui`. Los pulsos aplicados quedan en `simular.perturbaciones` como una lista de `(t, N·m)`, y los que se aplican con el teclado no son reproducibles, así que para resultados del informe programe `pulso` y `t_pulso`.

### Referencia que cambia en el tiempo (pregunta 15)

`q_ref` también puede ser una función del tiempo. Por ejemplo, para cambiar la referencia cada 3 s:

```python
refs = [q_a, q_b, q_c]                                   # referencias articulares [hombro, codo] en rad
q_ref = lambda t: refs[min(int(t / 3.0), len(refs) - 1)]
t, q, u = simular(ley, T=0.010, q_ref=q_ref, duracion=9.0)
```

`simular` entrega los ángulos `q`; la posición de la mano se calcula con la ecuación de cinemática del enunciado.

## Controlador RL (bonus, pregunta 16)

Se entrega un controlador entrenado con aprendizaje reforzado (`controlador_rl.py` y `controlador_rl.pt`). Entrega directamente las excitaciones, con las mismas mediciones retardadas que su controlador y la restricción `u ∈ [0, 1]`. Requiere `pip install torch` (basta la versión CPU, solo para leer el archivo `.pt`) y funciona con `T = 10 ms`.

```python
from controlador_rl import ControladorRL

ctrl = ControladorRL('controlador_rl.pt')          # se crea una vez; simular lo llama en cada paso de control
t, q, u = simular(ctrl, T=ctrl.T, q_ref=q_ref, duracion=6.0, salida='excitacion')
```

Para otra simulación, cree otro `ControladorRL` (así parte con la memoria en cero). `evaluar_rl.py` trae ejemplos (escalón, condiciones de la pregunta 12 y pulso). Las comparaciones con su controlador las hace usted.