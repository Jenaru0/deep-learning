#!/bin/bash
# Script para detener entrenamiento actual y reiniciar con optimizaciones
# Uso: bash scripts/utils/restart_training.sh

echo "================================================"
echo "🛑 DETENIENDO ENTRENAMIENTO ACTUAL"
echo "================================================"

# Encontrar proceso de Python ejecutando entrenar_deteccion.py
PID=$(ps aux | grep "entrenar_deteccion.py" | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "⚠️  No se encontró proceso de entrenamiento activo"
else
    echo "🔍 Encontrado proceso: PID $PID"
    echo "⏳ Deteniendo proceso..."
    kill -15 $PID
    sleep 2
    
    # Verificar si se detuvo
    if ps -p $PID > /dev/null; then
        echo "⚠️  Proceso no se detuvo, forzando terminación..."
        kill -9 $PID
    fi
    
    echo "✅ Proceso detenido"
fi

echo ""
echo "================================================"
echo "🔄 LIMPIANDO CACHE DE TensorFlow"
echo "================================================"

# Limpiar cache de TensorFlow/Keras
rm -rf ~/.keras/models/*
echo "✅ Cache limpiado"

echo ""
echo "================================================"
echo "📊 ESTADO DE GPU ANTES DE REINICIAR"
echo "================================================"

nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "================================================"
echo "⏳ Esperando 5 segundos para liberar VRAM..."
echo "================================================"

sleep 5

echo ""
echo "================================================"
echo "🚀 REINICIANDO ENTRENAMIENTO CON OPTIMIZACIONES"
echo "================================================"
echo ""
echo "Cambios aplicados:"
echo "  ✅ BATCH_SIZE: 16 → 24 (mejor saturación GPU)"
echo "  ✅ PREFETCH_BUFFER: AUTOTUNE → 3 batches (agresivo)"
echo "  ✅ NUM_IO_THREADS: 6 threads para I/O paralelo"
echo "  ✅ Pipeline optimizado para GPU rápida"
echo ""
echo "📝 RECOMENDACIÓN: Ejecutar en terminal separada:"
echo "     python3 scripts/utils/monitor_gpu_wsl.py"
echo ""
echo "Iniciando en 3 segundos..."
sleep 3

# Reiniciar entrenamiento
cd /mnt/c/Users/jonna/OneDrive/Escritorio/DEEP\ LEARNING/investigacion_fisuras
python3 scripts/entrenamiento/entrenar_deteccion.py
