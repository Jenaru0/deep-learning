"""
Script de Verificación de GPU para TensorFlow
==============================================

Verifica que TensorFlow esté usando la NVIDIA RTX 2050 correctamente
y proporciona diagnóstico completo del entorno GPU.

Uso:
    python3 scripts/utils/verificar_gpu.py

Autor: Jesus Naranjo
Fecha: Octubre 2025
"""

import sys
import subprocess
import tensorflow as tf
import numpy as np

print("=" * 80)
print("🔍 DIAGNÓSTICO COMPLETO DE GPU")
print("=" * 80)

# 1. Información de TensorFlow
print(f"\n📦 TensorFlow Version: {tf.__version__}")
print(f"🔧 Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"🔧 CUDA Version: {tf.sysconfig.get_build_info().get('cuda_version', 'N/A')}")
print(f"🔧 cuDNN Version: {tf.sysconfig.get_build_info().get('cudnn_version', 'N/A')}")

# 2. Listar TODAS las GPUs físicas
print("\n" + "=" * 80)
print("🎮 GPUs FÍSICAS DETECTADAS POR TENSORFLOW")
print("=" * 80)

all_physical_gpus = tf.config.list_physical_devices('GPU')
if all_physical_gpus:
    for idx, gpu in enumerate(all_physical_gpus):
        print(f"\n   GPU {idx}:")
        print(f"      Nombre: {gpu.name}")
        print(f"      Tipo: {gpu.device_type}")
        
        # Intentar obtener detalles
        try:
            details = tf.config.experimental.get_device_details(gpu)
            if details:
                print(f"      Detalles: {details}")
        except:
            pass
else:
    print("   ❌ No se detectaron GPUs físicas")
    print("   Verifica:")
    print("      - Drivers NVIDIA instalados (nvidia-smi)")
    print("      - CUDA Toolkit instalado")
    print("      - cuDNN instalado")
    sys.exit(1)

# 3. Listar GPUs VISIBLES (las que TensorFlow usará)
print("\n" + "=" * 80)
print("👁️ GPUs VISIBLES (QUE TENSORFLOW PUEDE USAR)")
print("=" * 80)

visible_gpus = tf.config.get_visible_devices('GPU')
if visible_gpus:
    for idx, gpu in enumerate(visible_gpus):
        print(f"\n   GPU Visible {idx}:")
        print(f"      Nombre: {gpu.name}")
        print(f"      Tipo: {gpu.device_type}")
        
        # Verificar si es NVIDIA
        is_nvidia = any(keyword in gpu.name.lower() for keyword in ['nvidia', 'geforce', 'rtx'])
        if is_nvidia:
            print(f"      ✅ NVIDIA detectada - GPU correcta")
        else:
            print(f"      ⚠️ No parece ser NVIDIA - posible iGPU Intel")
else:
    print("   ⚠️ No hay GPUs visibles configuradas")
    print("   TensorFlow usará CPU por defecto")

# 4. Test de computación en GPU
print("\n" + "=" * 80)
print("🧪 TEST DE COMPUTACIÓN EN GPU")
print("=" * 80)

try:
    # Crear tensores de prueba
    print("\n   Creando tensores de prueba (2000x2000)...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([2000, 2000])
        b = tf.random.normal([2000, 2000])
        
        # Multiplicación de matrices (operación intensiva)
        print("   Ejecutando multiplicación de matrices en GPU...")
        c = tf.matmul(a, b)
        
        # Forzar ejecución
        result = c.numpy()
        
    print(f"   ✅ Operación completada exitosamente en GPU")
    print(f"   Resultado shape: {result.shape}")
    print(f"   Device usado: GPU:0")
    
except Exception as e:
    print(f"   ❌ Error ejecutando en GPU: {e}")
    print("   Posibles causas:")
    print("      - GPU no disponible")
    print("      - Drivers NVIDIA no instalados")
    print("      - CUDA/cuDNN incompatibles")

# 5. Información de nvidia-smi
print("\n" + "=" * 80)
print("🖥️ INFORMACIÓN DE NVIDIA-SMI")
print("=" * 80)

try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True, timeout=5)
    
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for idx, line in enumerate(lines):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 7:
                print(f"\n   GPU {idx}:")
                print(f"      Nombre: {parts[0]}")
                print(f"      Driver: {parts[1]}")
                print(f"      VRAM Total: {parts[2]} MB")
                print(f"      VRAM Libre: {parts[3]} MB")
                print(f"      VRAM Usada: {parts[4]} MB ({float(parts[4])/float(parts[2])*100:.1f}%)")
                print(f"      GPU Utilization: {parts[5]}%")
                print(f"      Temperatura: {parts[6]}°C")
                
                # Verificar si es RTX 2050
                if 'rtx 2050' in parts[0].lower() or '2050' in parts[0].lower():
                    print(f"      ✅ RTX 2050 CONFIRMADA")
    else:
        print(f"   ⚠️ nvidia-smi retornó error: {result.stderr}")
        
except FileNotFoundError:
    print("   ❌ nvidia-smi no encontrado")
    print("   Instala drivers NVIDIA: https://www.nvidia.com/Download/index.aspx")
except subprocess.TimeoutExpired:
    print("   ⚠️ nvidia-smi timeout (5s)")
except Exception as e:
    print(f"   ⚠️ Error ejecutando nvidia-smi: {e}")

# 6. Recomendaciones finales
print("\n" + "=" * 80)
print("💡 RECOMENDACIONES")
print("=" * 80)

if visible_gpus and len(all_physical_gpus) > len(visible_gpus):
    print("\n   ✅ CONFIGURACIÓN CORRECTA:")
    print(f"      - Se detectaron {len(all_physical_gpus)} GPUs físicas (Intel + NVIDIA)")
    print(f"      - TensorFlow usará SOLO {len(visible_gpus)} GPU(s) (filtrado correcto)")
    print("      - Esto evita usar la iGPU Intel por error")
    
elif visible_gpus:
    print("\n   ℹ️ CONFIGURACIÓN ESTÁNDAR:")
    print(f"      - TensorFlow usará {len(visible_gpus)} GPU(s)")
    if len(all_physical_gpus) == 1:
        print("      - Solo hay 1 GPU en el sistema")
    
else:
    print("\n   ⚠️ PROBLEMA DETECTADO:")
    print("      - No hay GPUs visibles para TensorFlow")
    print("      - El entrenamiento usará CPU (muy lento)")
    print("\n   Soluciones:")
    print("      1. Verifica drivers NVIDIA: nvidia-smi")
    print("      2. Reinstala TensorFlow GPU: pip install tensorflow[and-cuda]")
    print("      3. Verifica compatibilidad CUDA/cuDNN")

print("\n" + "=" * 80)
print("✅ DIAGNÓSTICO COMPLETO")
print("=" * 80)
