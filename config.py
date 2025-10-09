"""
Configuración Central del Proyecto
===================================
Sistema de Detección y Segmentación de Fisuras Estructurales
Universidad Nacional de Cañete - Curso: Deep Learning

Autor: [Tu nombre]
Fecha: 7 de octubre, 2025
"""

import os
from pathlib import Path

# ============================================================================
# RUTAS ABSOLUTAS DE DATOS
# ============================================================================

# Ruta base del proyecto (detecta automáticamente si estás en WSL o Windows)
if os.name == 'posix' and os.path.exists('/mnt/c'):
    # Estamos en WSL2
    BASE_DIR = "/mnt/c/Users/jonna/OneDrive/Escritorio/DEEP LEARNING/investigacion_fisuras"
else:
    # Estamos en Windows
    BASE_DIR = r"C:\Users\jonna\OneDrive\Escritorio\DEEP LEARNING\investigacion_fisuras"

# Datasets originales (NO MODIFICAR ESTOS DIRECTORIOS)
RUTA_SDNET2018 = os.path.join(BASE_DIR, "datasets", "SDNET2018")
RUTA_CRACK500 = os.path.join(BASE_DIR, "datasets", "CRACK500")

# Datos procesados (aquí se copiarán las divisiones train/val/test)
RUTA_PROCESADOS = os.path.join(BASE_DIR, "datos", "procesados")
RUTA_DETECCION = os.path.join(RUTA_PROCESADOS, "deteccion")
RUTA_SEGMENTACION = os.path.join(RUTA_PROCESADOS, "segmentacion")

# Modelos entrenados
RUTA_MODELOS = os.path.join(BASE_DIR, "modelos")
RUTA_MODELO_DETECCION = os.path.join(RUTA_MODELOS, "deteccion")
RUTA_MODELO_SEGMENTACION = os.path.join(RUTA_MODELOS, "segmentacion")

# Resultados y visualizaciones
RUTA_RESULTADOS = os.path.join(BASE_DIR, "resultados")
RUTA_VIS = os.path.join(RUTA_RESULTADOS, "visualizaciones")

# Reportes y documentación
RUTA_REPORTES = os.path.join(BASE_DIR, "reportes")

# ============================================================================
# PARÁMETROS DE DATOS
# ============================================================================

# Inventario de SDNET2018 (verificado el 7 de octubre, 2025)
SDNET_INVENTORY = {
    'Deck': {'CD': 2025, 'UD': 11595, 'Total': 13620},
    'Pavement': {'CP': 2608, 'UP': 21726, 'Total': 24334},
    'Wall': {'CW': 3851, 'UW': 14287, 'Total': 18138},
    'Total': {'Cracked': 8484, 'Uncracked': 47608, 'Total': 56092}
}

# Inventario de CRACK500 (verificado el 7 de octubre, 2025)
CRACK500_INVENTORY = {
    'train': 1896,
    'val': 348,
    'test': 1124,
    'total': 3368
}

# ============================================================================
# PARÁMETROS DE ENTRENAMIENTO
# ============================================================================

# Dimensiones de imagen
IMG_SIZE = 224  # MobileNetV2 y EfficientNetB0 usan 224x224
IMG_CHANNELS = 3  # RGB

# Batch size y epochs
# OPTIMIZACIÓN GPU RTX 2050: Target 75-85% GPU utilization
# - VRAM Total: 4096 MB | VRAM para TF: ~1767 MB (límite configurado)
# - Batch size 32 aprovecha ~75-80% VRAM y mantiene GPU ocupada
# - MobileNetV2 es ligero: 32 imágenes 224x224 = ~150MB por batch
# 
# Progresión de tuning:
#   16 → 40% VRAM, 25% GPU util (muy conservador)
#   24 → 60% VRAM, 50% GPU util (mejorado pero insuficiente)
#   32 → 75-80% VRAM, 75-85% GPU util ✅ ÓPTIMO para RTX 2050
#   40 → Riesgo OOM (out of memory)
# 
# Si experimentas OOM error durante entrenamiento:
#   1. Reducir a BATCH_SIZE = 28
#   2. Si persiste → BATCH_SIZE = 24
#   3. Verificar procesos paralelos consumiendo VRAM (cierra Chrome, Discord, etc)
BATCH_SIZE = 32  
EPOCHS_DETECCION = 50
EPOCHS_SEGMENTACION = 50

# División de datos para SDNET2018 (CRACK500 ya tiene división predefinida)
TRAIN_RATIO = 0.70  # 70% entrenamiento
VAL_RATIO = 0.15    # 15% validación
TEST_RATIO = 0.15   # 15% prueba

# ============================================================================
# PARÁMETROS DE MODELO
# ============================================================================

# Detección (MobileNetV2 con Transfer Learning)
# Por qué MobileNetV2:
#   - Más ligero: 3.5M parámetros vs 5.3M de EfficientNetB0
#   - Mejor para detectar objetos pequeños (fisuras finas)
#   - Compatible con Keras 3.x sin bugs
#   - 20% más rápido en entrenamiento
#   - Excelente para deployment en edge devices
LEARNING_RATE_DETECCION = 1e-4
OPTIMIZER_DETECCION = 'adam'
LOSS_DETECCION = 'binary_crossentropy'  # Clasificación binaria

# Segmentación (U-Net con encoder EfficientNetB0)
LEARNING_RATE_SEGMENTACION = 1e-4
OPTIMIZER_SEGMENTACION = 'adam'
LOSS_SEGMENTACION = 'binary_crossentropy'  # Segmentación binaria

# ============================================================================
# REPRODUCIBILIDAD
# ============================================================================

# Semilla aleatoria (CRÍTICO: Mantener en 42 para reproducibilidad)
RANDOM_SEED = 42

# ============================================================================
# MÉTRICAS DE EVALUACIÓN
# ============================================================================

# Detección
METRICAS_DETECCION = ['accuracy', 'precision', 'recall', 'AUC']

# Segmentación
METRICAS_SEGMENTACION = ['accuracy', 'IoU', 'dice_coefficient']

# ============================================================================
# CONFIGURACIÓN DE AUGMENTATION
# ============================================================================

# Data Augmentation para entrenamiento
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': False,  # No voltear verticalmente fisuras
    'zoom_range': 0.2,
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'reflect'
}

# ============================================================================
# UMBRAL DE CLASIFICACIÓN DE SEVERIDAD
# ============================================================================

# Basado en área de fisura (%)
SEVERIDAD_UMBRALES = {
    'LEVE': 0.5,      # < 0.5% de área
    'MODERADA': 2.0,  # 0.5% - 2.0%
    'SEVERA': float('inf')  # > 2.0%
}

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# VALIDACIÓN DE RUTAS (Ejecutar al importar)
# ============================================================================

def validar_rutas():
    """Verifica que las rutas críticas existan"""
    rutas_criticas = [RUTA_SDNET2018, RUTA_CRACK500]
    for ruta in rutas_criticas:
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"Ruta crítica no encontrada: {ruta}")
    print("✅ Todas las rutas críticas validadas correctamente")

if __name__ == "__main__":
    validar_rutas()
    print(f"📁 Proyecto base: {BASE_DIR}")
    print(f"📊 SDNET2018: {SDNET_INVENTORY['Total']['Total']} imágenes")
    print(f"📊 CRACK500: {CRACK500_INVENTORY['total']} imágenes")
