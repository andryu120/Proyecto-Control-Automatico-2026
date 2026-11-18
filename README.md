Consideraciones:

# 1. Requisitos del Entorno (MuJoCo 3.16)

Tienen que tener installado MuJoCo 3.16, ademas de numpy, matplotlib:
```bash
pip install mujoco==3.16.0
```


# 2. Trabajo Local (Sin Git / GitHub)

Si no utilizas github, puedes descarga el proyecto en un zip y trabajar localmente

# 3. Estructura y Creación de Archivos

La idea, si es que van a utilizar github, crear los archivos en Codigos_preguntas (Para las preguntas)


# 4. Importación de Módulos Superiores (sys.path)


Para evitar errores de importación, añade esta línea al comienzo de tus archivos de código:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
```



# 5. Manejo de Rutas de Archivos

Para que no tengan problemas al abrir archivos, deben usar rutas absolutas o basadas en el script, para ello se puede usar pathlib de la siguiente manera:

```python
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
# Ejemplo de uso con archivos locales
archivo_datos = BASE_DIR / "mis_datos.npz"
```