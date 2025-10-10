#!/bin/bash

# ============================================================================
# Script de Instalación Automática - GPU Setup para WSL2
# ============================================================================
#
# Instala y configura todo lo necesario para usar RTX 2050 con TensorFlow
# en WSL2 (Ubuntu).
#
# REQUISITOS PREVIOS (en Windows):
#   1. WSL2 instalado y actualizado
#   2. Drivers NVIDIA instalados (versión Game Ready o Studio)
#   3. nvidia-smi funcional en Windows
#
# USO:
#   cd /mnt/c/Users/jonna/OneDrive/Escritorio/DEEP\ LEARNING/investigacion_fisuras
#   bash scripts/utils/instalar_gpu_wsl2.sh
#
# Autor: Jesus Naranjo
# Fecha: Octubre 2025
# ============================================================================

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🚀 INSTALACIÓN AUTOMÁTICA - GPU SETUP PARA WSL2"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Este script instalará:"
echo "  • CUDA Toolkit 12.5"
echo "  • cuDNN"
echo "  • TensorFlow con soporte GPU"
echo "  • Dependencias del proyecto"
echo ""
read -p "¿Continuar? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Instalación cancelada."
    exit 1
fi

echo ""

# ============================================================================
# 1. ACTUALIZAR SISTEMA
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  Actualizando sistema..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

sudo apt update
sudo apt upgrade -y

echo "✅ Sistema actualizado"
echo ""

# ============================================================================
# 2. INSTALAR DEPENDENCIAS BASE
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  Instalando dependencias base..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

sudo apt install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    software-properties-common

echo "✅ Dependencias base instaladas"
echo ""

# ============================================================================
# 3. VERIFICAR NVIDIA DRIVER
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  Verificando drivers NVIDIA..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi encontrado"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo ""
else
    echo "❌ ERROR: nvidia-smi NO encontrado"
    echo ""
    echo "SOLUCIÓN:"
    echo "  1. Instala drivers NVIDIA en Windows:"
    echo "     https://www.nvidia.com/Download/index.aspx"
    echo "  2. Reinicia Windows"
    echo "  3. Verifica con: nvidia-smi (en PowerShell)"
    echo "  4. Ejecuta este script nuevamente"
    exit 1
fi

# ============================================================================
# 4. INSTALAR CUDA TOOLKIT
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️⃣  Instalando CUDA Toolkit 12.5..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verificar si ya está instalado
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA Toolkit ya instalado:"
    nvcc --version | grep "release"
    echo ""
else
    echo "📥 Descargando e instalando CUDA Toolkit..."
    
    # Descargar keyring
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    
    # Actualizar e instalar
    sudo apt update
    sudo apt install -y cuda-toolkit-12-5
    
    # Limpiar
    rm cuda-keyring_1.1-1_all.deb
    
    echo "✅ CUDA Toolkit instalado"
    echo ""
fi

# ============================================================================
# 5. CONFIGURAR PATH
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5️⃣  Configurando PATH..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Agregar CUDA al PATH (solo si no existe)
if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA Path" >> ~/.bashrc
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo "✅ PATH agregado a ~/.bashrc"
else
    echo "✅ PATH ya configurado"
fi

# Aplicar cambios
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo ""

# ============================================================================
# 6. INSTALAR TENSORFLOW
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6️⃣  Instalando TensorFlow con soporte GPU..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Actualizar pip
python3 -m pip install --upgrade pip

# Instalar TensorFlow con GPU
pip3 install tensorflow[and-cuda] --upgrade

echo "✅ TensorFlow instalado"
echo ""

# ============================================================================
# 7. INSTALAR DEPENDENCIAS DEL PROYECTO
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7️⃣  Instalando dependencias del proyecto..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PROJECT_DIR="/mnt/c/Users/jonna/OneDrive/Escritorio/DEEP LEARNING/investigacion_fisuras"

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip3 install -r "$PROJECT_DIR/requirements.txt"
    echo "✅ Dependencias del proyecto instaladas"
else
    echo "⚠️  requirements.txt no encontrado"
    echo "   Instalando paquetes básicos..."
    pip3 install numpy pandas matplotlib seaborn scikit-learn pillow opencv-python
fi

echo ""

# ============================================================================
# 8. VERIFICAR INSTALACIÓN
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8️⃣  Verificando instalación..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 << 'PYTHON_VERIFY'
import sys

print("\n🔍 Verificando TensorFlow y GPU...\n")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__}")
    print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ {len(gpus)} GPU(s) detectada(s):")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        
        # Test rápido
        print("\n🧪 Test de computación GPU...")
        with tf.device('/GPU:0'):
            import time
            start = time.time()
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            result = c.numpy()
            elapsed = time.time() - start
        
        print(f"✅ GPU funcional ({elapsed*1000:.2f}ms)")
        
    else:
        print("❌ No se detectaron GPUs")
        print("   Verifica drivers NVIDIA")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    
print("\n" + "="*80)
print("✅ INSTALACIÓN COMPLETADA EXITOSAMENTE")
print("="*80)
PYTHON_VERIFY

echo ""

# ============================================================================
# 9. RESUMEN Y PRÓXIMOS PASOS
# ============================================================================

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🎉 INSTALACIÓN COMPLETADA"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ TODO LISTO PARA ENTRENAMIENTO ULTRA-RÁPIDO"
echo ""
echo "📊 Configuración instalada:"
echo "   • NVIDIA Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo "   • CUDA: $(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')"
echo "   • TensorFlow: $(python3 -c 'import tensorflow as tf; print(tf.__version__)')"
echo ""
echo "🚀 PRÓXIMOS PASOS:"
echo ""
echo "   1. Ejecutar test completo:"
echo "      bash scripts/utils/test_gpu_completo.sh"
echo ""
echo "   2. Entrenar modelo optimizado:"
echo "      python3 scripts/entrenamiento/entrenar_deteccion_turbo.py"
echo ""
echo "   3. Leer guía completa:"
echo "      cat docs/guias/GUIA_ENTRENAMIENTO_TURBO_GPU.md"
echo ""
echo "💡 IMPORTANTE:"
echo "   • Reinicia tu terminal para aplicar cambios en PATH"
echo "   • O ejecuta: source ~/.bashrc"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
