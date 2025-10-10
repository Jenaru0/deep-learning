# 🚀 Comandos de Referencia Rápida - RTX 2050 Turbo

## 📋 Comandos más usados

### 🔧 Setup Inicial (solo primera vez)

```bash
# Navegar al proyecto:
cd /mnt/c/Users/jonna/OneDrive/Escritorio/DEEP\ LEARNING/investigacion_fisuras

# Instalar todo automáticamente:
bash scripts/utils/instalar_gpu_wsl2.sh
```

---

## ✅ Verificación

### Verificar GPU y drivers:

```bash
# Ver info GPU:
nvidia-smi

# Ver uso en tiempo real:
watch -n 1 nvidia-smi

# Test completo:
bash scripts/utils/test_gpu_completo.sh

# Solo verificar TensorFlow:
python3 scripts/utils/verificar_gpu.py
```

### Verificar configuración optimizada:

```bash
# Test configuración GPU:
python3 scripts/utils/configurar_gpu.py

# Verificar Mixed Precision:
python3 -c "from tensorflow.keras import mixed_precision; print(mixed_precision.global_policy())"
# Debe mostrar: <Policy "mixed_float16">

# Verificar TensorFlow + GPU:
python3 -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

---

## 🎓 Entrenamiento

### Preparar datos (solo primera vez):

```bash
# Dividir SDNET2018:
python3 scripts/preprocesamiento/dividir_sdnet2018.py

# Validar splits:
python3 scripts/preprocesamiento/validar_splits.py
```

### Entrenar modelo optimizado:

```bash
# RECOMENDADO: Script turbo (30-40 min):
python3 scripts/entrenamiento/entrenar_deteccion_turbo.py

# Alternativa: Script original modificado:
python3 scripts/entrenamiento/entrenar_deteccion.py
```

### Monitoreo durante entrenamiento:

```bash
# Terminal 1: Entrenar
python3 scripts/entrenamiento/entrenar_deteccion_turbo.py

# Terminal 2: Monitor GPU básico
watch -n 1 nvidia-smi

# Terminal 2 alternativa: Monitor detallado
nvidia-smi dmon -s pucvmet

# Terminal 3: TensorBoard (opcional)
tensorboard --logdir modelos/deteccion/logs
# Abrir: http://localhost:6006
```

---

## 📊 Evaluación

```bash
# Evaluar modelo entrenado:
python3 scripts/evaluacion/evaluar_deteccion.py

# Generar visualizaciones:
python3 scripts/evaluacion/generar_visualizaciones.py
```

---

## 🔍 Información del Sistema

### Ver versiones instaladas:

```bash
# Python:
python3 --version

# TensorFlow:
python3 -c "import tensorflow as tf; print(tf.__version__)"

# CUDA:
nvcc --version

# GPU info detallada:
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
```

### Ver configuración del proyecto:

```bash
# Ver config.py:
cat config.py | grep -A 3 "BATCH_SIZE\|EPOCHS\|GPU_CONFIG"

# Ver estructura:
tree -L 2 -I '__pycache__|*.pyc'
```

---

## 🛠️ Troubleshooting

### GPU no detectada:

```bash
# Verificar driver:
nvidia-smi

# Reinstalar TensorFlow:
pip3 uninstall tensorflow tensorflow-gpu
pip3 install tensorflow[and-cuda] --upgrade

# Verificar instalación:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### OOM Error (Out of Memory):

```python
# Editar config.py:
# Reducir batch size:
BATCH_SIZE = 56  # Era 64
# O más conservador:
BATCH_SIZE = 48
```

### Entrenamiento lento:

```bash
# Verificar Mixed Precision:
python3 -c "from tensorflow.keras import mixed_precision; print(mixed_precision.global_policy())"

# Verificar que uses script turbo:
python3 scripts/entrenamiento/entrenar_deteccion_turbo.py

# Verificar GPU utilization:
nvidia-smi
# Debe mostrar 85-95% GPU utilization
```

### Limpiar memoria GPU:

```bash
# Matar procesos Python:
pkill -9 python3

# Ver procesos usando GPU:
nvidia-smi pmon

# Reiniciar WSL (desde PowerShell Windows):
wsl --shutdown
```

---

## 📦 Gestión de Dependencias

### Actualizar paquetes:

```bash
# Actualizar pip:
python3 -m pip install --upgrade pip

# Actualizar TensorFlow:
pip3 install tensorflow[and-cuda] --upgrade

# Instalar desde requirements.txt:
pip3 install -r requirements.txt

# Ver paquetes instalados:
pip3 list | grep -i "tensor\|numpy\|cuda"
```

---

## 🎯 Workflows Comunes

### Workflow 1: Primera vez (setup completo)

```bash
# 1. Instalar:
bash scripts/utils/instalar_gpu_wsl2.sh

# 2. Verificar:
bash scripts/utils/test_gpu_completo.sh

# 3. Preparar datos:
python3 scripts/preprocesamiento/dividir_sdnet2018.py

# 4. Entrenar:
python3 scripts/entrenamiento/entrenar_deteccion_turbo.py
```

### Workflow 2: Desarrollo diario

```bash
# 1. Verificar GPU:
nvidia-smi

# 2. Entrenar:
python3 scripts/entrenamiento/entrenar_deteccion_turbo.py

# 3. Evaluar:
python3 scripts/evaluacion/evaluar_deteccion.py
```

### Workflow 3: Experimentación

```bash
# 1. Modificar hiperparámetros en config.py

# 2. Entrenar con nuevos parámetros:
python3 scripts/entrenamiento/entrenar_deteccion_turbo.py

# 3. Comparar resultados:
python3 scripts/evaluacion/comparar_modelos.py
```

---

## 📂 Rutas Importantes

```bash
# Proyecto:
PROJECT=/mnt/c/Users/jonna/OneDrive/Escritorio/DEEP\ LEARNING/investigacion_fisuras

# Datos:
DATOS=$PROJECT/datos/procesados/deteccion

# Modelos:
MODELOS=$PROJECT/modelos/deteccion

# Resultados:
RESULTADOS=$PROJECT/resultados

# Logs:
LOGS=$PROJECT/docs/logs
```

---

## 🎨 Alias Útiles (opcional)

Agrega a `~/.bashrc`:

```bash
# Alias para proyecto:
alias cdproj='cd /mnt/c/Users/jonna/OneDrive/Escritorio/DEEP\ LEARNING/investigacion_fisuras'
alias gpu='watch -n 1 nvidia-smi'
alias gpuinfo='nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv'
alias tfver='python3 -c "import tensorflow as tf; print(tf.__version__)"'
alias testgpu='bash scripts/utils/test_gpu_completo.sh'
alias entrenar='python3 scripts/entrenamiento/entrenar_deteccion_turbo.py'

# Aplicar cambios:
source ~/.bashrc
```

Ahora puedes usar:

```bash
cdproj    # Ir al proyecto
gpu       # Monitor GPU
testgpu   # Test completo
entrenar  # Entrenar modelo
```

---

## 📊 Métricas Esperadas

### Durante entrenamiento:

```
GPU Utilization:  85-95% ✅
VRAM Usage:       3000-3500 MB / 4096 MB
Tiempo/epoch:     1.0-1.2 min
Temperatura:      65-75°C
Power Draw:       35-40W
```

### Modelo final:

```
Val Accuracy:  93-95%
Val Precision: 88-92%
Val Recall:    85-90%
Val AUC:       0.96-0.98
Tiempo total:  30-40 min ⚡
```

---

## 📚 Documentación

```bash
# Ver guía completa:
cat docs/guias/SETUP_FINAL_RTX2050.md

# Ver optimizaciones:
cat docs/guias/OPTIMIZACIONES_GPU_RESUMEN.md

# Ver guía detallada:
cat docs/guias/GUIA_ENTRENAMIENTO_TURBO_GPU.md

# Ver este archivo:
cat docs/guias/COMANDOS_REFERENCIA_RAPIDA.md
```

---

## 🆘 Ayuda Rápida

```bash
# Proyecto no funciona:
bash scripts/utils/test_gpu_completo.sh

# GPU no detectada:
nvidia-smi
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Entrenamiento lento:
python3 -c "from tensorflow.keras import mixed_precision; print(mixed_precision.global_policy())"

# OOM Error:
# Editar config.py → BATCH_SIZE = 48
```

---

**Última actualización:** Octubre 2025  
**Hardware:** RTX 2050 (4GB VRAM)  
**Optimización:** 2.5-3.0x speed-up ⚡
