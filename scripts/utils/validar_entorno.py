#!/usr/bin/env python3
"""
Script de Validación de Entorno Pre-Entrenamiento
=================================================

Este script valida que el entorno de ejecución cumple con todos los
requisitos necesarios para entrenar modelos de Deep Learning sin errores.

Validaciones:
    1. Detecta si se está ejecutando en WSL (no Windows nativo)
    2. Verifica que TensorFlow detecta la GPU correctamente
    3. Valida que hay suficiente VRAM disponible (>= 3.5 GB)
    4. Comprueba espacio en disco disponible (>= 10 GB)
    5. Verifica que los directorios de datos existen
    6. Valida versiones de TensorFlow y Keras
    7. Comprueba que CUDA está disponible

Uso:
    python3 scripts/utils/validar_entorno.py

Autor: Jesus Naranjo (bajo supervisión de Claude 4.5)
Fecha: 9 de octubre de 2025
"""

import os
import sys
import platform
import shutil
from pathlib import Path

# Colores para terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Imprime encabezado destacado"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}\n")

def print_success(text):
    """Imprime mensaje de éxito"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text):
    """Imprime mensaje de error"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text):
    """Imprime mensaje de advertencia"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text):
    """Imprime mensaje informativo"""
    print(f"   {text}")

# =============================================================================
# VALIDACIÓN 1: Detectar WSL vs Windows
# =============================================================================

def validar_wsl():
    """
    Verifica que el script se está ejecutando en WSL, no en Windows nativo.
    
    TensorFlow GPU solo funciona correctamente en WSL2 con este setup.
    """
    print_header("VALIDACIÓN 1: Entorno de Ejecución")
    
    # Detectar WSL
    es_wsl = False
    
    # Método 1: Verificar /proc/version
    if os.path.exists('/proc/version'):
        with open('/proc/version', 'r') as f:
            version_info = f.read().lower()
            if 'microsoft' in version_info or 'wsl' in version_info:
                es_wsl = True
    
    # Método 2: Verificar variable de entorno
    if 'WSL_DISTRO_NAME' in os.environ or 'WSL_INTEROP' in os.environ:
        es_wsl = True
    
    # Método 3: Verificar si /mnt/c existe (típico de WSL)
    if os.path.exists('/mnt/c'):
        es_wsl = True
    
    if es_wsl:
        print_success("Ejecutando en WSL2 (correcto)")
        print_info(f"Distribución: {os.environ.get('WSL_DISTRO_NAME', 'Unknown')}")
        print_info(f"Sistema: {platform.system()} {platform.release()}")
        return True
    else:
        print_error("NO se está ejecutando en WSL2")
        print_warning("TensorFlow GPU solo funciona en WSL2 en este setup")
        print_info("Solución: Abre una terminal WSL y ejecuta desde ahí")
        return False

# =============================================================================
# VALIDACIÓN 2: TensorFlow y GPU
# =============================================================================

def validar_tensorflow_gpu():
    """
    Verifica que TensorFlow está instalado y detecta la GPU correctamente.
    """
    print_header("VALIDACIÓN 2: TensorFlow y GPU")
    
    try:
        import tensorflow as tf
        print_success(f"TensorFlow {tf.__version__} importado correctamente")
        
        # Verificar versión de Keras
        print_info(f"Keras version: {tf.keras.__version__}")
        
        # Verificar si está compilado con CUDA
        cuda_build = tf.test.is_built_with_cuda()
        if cuda_build:
            print_success("TensorFlow compilado con soporte CUDA")
        else:
            print_error("TensorFlow NO compilado con CUDA")
            print_warning("Se ejecutará en CPU (100x más lento)")
            return False
        
        # Detectar GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        if len(gpus) == 0:
            print_error("No se detectaron GPUs disponibles")
            print_warning("El entrenamiento se ejecutará en CPU")
            print_info("Verifica que los drivers NVIDIA están instalados en Windows")
            return False
        else:
            print_success(f"Detectadas {len(gpus)} GPU(s):")
            for gpu in gpus:
                print_info(f"  • {gpu.name}")
            
            # Intentar obtener detalles de la GPU
            try:
                # Crear un tensor en GPU para forzar inicialización
                with tf.device('/GPU:0'):
                    _ = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                
                # Obtener propiedades de la GPU
                from tensorflow.python.client import device_lib
                local_devices = device_lib.list_local_devices()
                
                for device in local_devices:
                    if device.device_type == 'GPU':
                        print_info(f"  • Memoria total: {device.memory_limit / (1024**3):.2f} GB")
                
            except Exception as e:
                print_warning(f"No se pudo obtener información detallada de GPU: {e}")
            
            return True
            
    except ImportError:
        print_error("TensorFlow no está instalado")
        print_info("Instala con: pip install tensorflow==2.17.0")
        return False
    except Exception as e:
        print_error(f"Error al verificar TensorFlow: {e}")
        return False

# =============================================================================
# VALIDACIÓN 3: Memoria VRAM
# =============================================================================

def validar_vram():
    """
    Verifica que hay suficiente VRAM disponible (>= 3.5 GB).
    
    El modelo MobileNetV2 con batch_size=32 requiere ~3.2-3.5 GB VRAM.
    """
    print_header("VALIDACIÓN 3: Memoria GPU (VRAM)")
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) == 0:
            print_warning("No hay GPU para verificar VRAM")
            return False
        
        # Crear un tensor pequeño para inicializar GPU
        with tf.device('/GPU:0'):
            _ = tf.constant([[1.0]])
        
        # Intentar obtener información de memoria
        try:
            from tensorflow.python.client import device_lib
            local_devices = device_lib.list_local_devices()
            
            for device in local_devices:
                if device.device_type == 'GPU':
                    memory_gb = device.memory_limit / (1024**3)
                    
                    # Verificar que hay al menos 3.5 GB
                    if memory_gb >= 3.5:
                        print_success(f"VRAM disponible: {memory_gb:.2f} GB (suficiente)")
                        return True
                    else:
                        print_warning(f"VRAM disponible: {memory_gb:.2f} GB")
                        print_warning("Se recomienda >= 3.5 GB para batch_size=32")
                        print_info("Considera reducir batch_size en config.py")
                        return True  # No bloqueante, solo warning
            
            print_warning("No se pudo determinar VRAM disponible")
            return True  # No bloqueante
            
        except Exception as e:
            print_warning(f"No se pudo verificar VRAM: {e}")
            return True  # No bloqueante
            
    except Exception as e:
        print_error(f"Error al verificar VRAM: {e}")
        return False

# =============================================================================
# VALIDACIÓN 4: Espacio en Disco
# =============================================================================

def validar_espacio_disco():
    """
    Verifica que hay suficiente espacio en disco (>= 10 GB).
    
    Necesario para guardar modelos, checkpoints, logs y resultados.
    """
    print_header("VALIDACIÓN 4: Espacio en Disco")
    
    try:
        # Obtener información del disco donde está el proyecto
        proyecto_path = Path(__file__).parent.parent.parent.absolute()
        
        disk_usage = shutil.disk_usage(proyecto_path)
        free_gb = disk_usage.free / (1024**3)
        
        if free_gb >= 10:
            print_success(f"Espacio disponible: {free_gb:.2f} GB (suficiente)")
            return True
        else:
            print_error(f"Espacio disponible: {free_gb:.2f} GB")
            print_warning("Se recomienda >= 10 GB para modelos y resultados")
            print_info("Libera espacio antes de continuar")
            return False
            
    except Exception as e:
        print_warning(f"No se pudo verificar espacio en disco: {e}")
        return True  # No bloqueante

# =============================================================================
# VALIDACIÓN 5: Directorios de Datos
# =============================================================================

def validar_directorios():
    """
    Verifica que los directorios de datos procesados existen.
    """
    print_header("VALIDACIÓN 5: Directorios de Datos")
    
    # Determinar ruta base del proyecto
    proyecto_path = Path(__file__).parent.parent.parent.absolute()
    
    # Directorios críticos para entrenamiento de detección
    directorios_criticos = [
        proyecto_path / "datos" / "procesados" / "deteccion" / "train" / "cracked",
        proyecto_path / "datos" / "procesados" / "deteccion" / "train" / "uncracked",
        proyecto_path / "datos" / "procesados" / "deteccion" / "val" / "cracked",
        proyecto_path / "datos" / "procesados" / "deteccion" / "val" / "uncracked",
        proyecto_path / "datos" / "procesados" / "deteccion" / "test" / "cracked",
        proyecto_path / "datos" / "procesados" / "deteccion" / "test" / "uncracked",
        proyecto_path / "modelos" / "deteccion",
        proyecto_path / "resultados" / "visualizaciones",
    ]
    
    todos_existen = True
    
    for directorio in directorios_criticos:
        if directorio.exists():
            # Contar archivos si es directorio de datos
            if "train" in str(directorio) or "val" in str(directorio) or "test" in str(directorio):
                num_archivos = len(list(directorio.glob("*.jpg")))
                print_success(f"{directorio.name}: {num_archivos} imágenes")
        else:
            print_error(f"Directorio faltante: {directorio}")
            todos_existen = False
    
    if todos_existen:
        print_success("Todos los directorios críticos existen")
        return True
    else:
        print_error("Faltan directorios necesarios")
        print_info("Ejecuta los scripts de preprocesamiento primero")
        return False

# =============================================================================
# VALIDACIÓN 6: Versiones de Paquetes
# =============================================================================

def validar_versiones():
    """
    Verifica que las versiones de paquetes críticos son compatibles.
    """
    print_header("VALIDACIÓN 6: Versiones de Paquetes")
    
    paquetes_criticos = {
        'tensorflow': '2.17.0',
        'keras': '3.11.3',
        'numpy': '1.26.4',
        'pandas': '2.3.3',
        'scikit-learn': '1.7.2',
        'matplotlib': '3.10.7',
        'seaborn': '0.13.2'
    }
    
    todas_correctas = True
    
    for paquete, version_esperada in paquetes_criticos.items():
        try:
            modulo = __import__(paquete)
            version_actual = modulo.__version__
            
            if version_actual == version_esperada:
                print_success(f"{paquete}: {version_actual}")
            else:
                print_warning(f"{paquete}: {version_actual} (esperada: {version_esperada})")
                # No bloqueante, solo warning
                
        except ImportError:
            print_error(f"{paquete}: NO INSTALADO")
            todas_correctas = False
        except AttributeError:
            print_warning(f"{paquete}: instalado pero sin __version__")
    
    return todas_correctas

# =============================================================================
# VALIDACIÓN 7: Configuración del Proyecto
# =============================================================================

def validar_config():
    """
    Verifica que el archivo config.py es válido y accesible.
    """
    print_header("VALIDACIÓN 7: Archivo config.py")
    
    try:
        # Añadir ruta del proyecto al path
        proyecto_path = Path(__file__).parent.parent.parent.absolute()
        sys.path.insert(0, str(proyecto_path))
        
        import config
        
        # Verificar parámetros críticos
        parametros = {
            'IMG_SIZE': config.IMG_SIZE,
            'BATCH_SIZE': config.BATCH_SIZE,
            'RANDOM_SEED': config.RANDOM_SEED,
            'RUTA_DETECCION': config.RUTA_DETECCION,
            'RUTA_MODELO_DETECCION': config.RUTA_MODELO_DETECCION,
        }
        
        print_success("config.py cargado correctamente")
        for param, valor in parametros.items():
            print_info(f"  • {param}: {valor}")
        
        return True
        
    except ImportError as e:
        print_error(f"No se pudo importar config.py: {e}")
        return False
    except AttributeError as e:
        print_error(f"Falta parámetro en config.py: {e}")
        return False

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Ejecuta todas las validaciones en secuencia.
    
    Returns:
        bool: True si todas las validaciones críticas pasan, False si alguna falla
    """
    print_header("VALIDACIÓN DE ENTORNO PARA ENTRENAMIENTO")
    print(f"Fecha: {os.popen('date').read().strip()}")
    print(f"Usuario: {os.environ.get('USER', 'Unknown')}")
    print(f"Directorio actual: {os.getcwd()}")
    
    # Ejecutar validaciones
    resultados = {
        'WSL': validar_wsl(),
        'TensorFlow y GPU': validar_tensorflow_gpu(),
        'VRAM': validar_vram(),
        'Espacio en Disco': validar_espacio_disco(),
        'Directorios': validar_directorios(),
        'Versiones': validar_versiones(),
        'Config': validar_config(),
    }
    
    # Resumen final
    print_header("RESUMEN DE VALIDACIONES")
    
    validaciones_criticas = ['WSL', 'TensorFlow y GPU', 'Directorios', 'Config']
    validaciones_opcionales = ['VRAM', 'Espacio en Disco', 'Versiones']
    
    print(f"\n{Colors.BOLD}Validaciones Críticas:{Colors.END}")
    criticas_ok = True
    for validacion in validaciones_criticas:
        status = "✅ PASS" if resultados[validacion] else "❌ FAIL"
        color = Colors.GREEN if resultados[validacion] else Colors.RED
        print(f"  {color}{status}{Colors.END} - {validacion}")
        if not resultados[validacion]:
            criticas_ok = False
    
    print(f"\n{Colors.BOLD}Validaciones Opcionales:{Colors.END}")
    for validacion in validaciones_opcionales:
        status = "✅ PASS" if resultados[validacion] else "⚠️  WARN"
        color = Colors.GREEN if resultados[validacion] else Colors.YELLOW
        print(f"  {color}{status}{Colors.END} - {validacion}")
    
    # Veredicto final
    print_header("VEREDICTO FINAL")
    
    if criticas_ok:
        print_success("ENTORNO VALIDADO - LISTO PARA ENTRENAR")
        print_info("Puedes proceder con el entrenamiento ejecutando:")
        print_info("  python3 scripts/entrenamiento/entrenar_deteccion.py")
        return True
    else:
        print_error("ENTORNO NO VÁLIDO - CORRIGE LOS ERRORES ANTES DE ENTRENAR")
        print_info("Revisa los errores marcados arriba y corrígelos")
        return False

if __name__ == "__main__":
    exito = main()
    sys.exit(0 if exito else 1)
