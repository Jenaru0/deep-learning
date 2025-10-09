#!/bin/bash
################################################################################
# Script: Copiar Datos a Sistema de Archivos Nativo de WSL
################################################################################
# 
# PROBLEMA: 
#   - Leer datos desde OneDrive en WSL2 es LENTO (298ms/step)
#   - Overhead: WSL → Windows → OneDrive sync → Lectura
#
# SOLUCIÓN:
#   - Copiar datos procesados a /home/jesus/investigacion_fisuras_data/
#   - Sistema de archivos nativo ext4 (5-10x más rápido)
#   - Speedup esperado: 298ms/step → 150-180ms/step
#
# USO:
#   bash scripts/utils/copiar_datos_wsl.sh
#
# Autor: Jesus Naranjo
# Fecha: Octubre 2025
################################################################################

set -e  # Salir si hay error

echo "================================================================================"
echo "🚀 OPTIMIZACIÓN: Copiar datos a filesystem nativo de WSL"
echo "================================================================================"

# Rutas
ORIGEN="/mnt/c/Users/jonna/OneDrive/Escritorio/DEEP LEARNING/investigacion_fisuras/datos/procesados/deteccion"
DESTINO="$HOME/investigacion_fisuras_data/deteccion"

# Verificar que origen existe
if [ ! -d "$ORIGEN" ]; then
    echo "❌ ERROR: Directorio origen no existe: $ORIGEN"
    exit 1
fi

# Mostrar información
echo ""
echo "📁 Origen (OneDrive/WSL - LENTO):"
echo "   $ORIGEN"
echo ""
echo "📁 Destino (WSL nativo ext4 - RÁPIDO):"
echo "   $DESTINO"
echo ""

# Calcular tamaño total
echo "📊 Calculando tamaño de datos..."
TAMANO=$(du -sh "$ORIGEN" | cut -f1)
echo "   Tamaño total: $TAMANO"
echo ""

# Contar archivos
NUM_FILES=$(find "$ORIGEN" -type f | wc -l)
echo "   Total archivos: $NUM_FILES"
echo ""

# Confirmar con usuario
echo "⚠️  Esta operación copiará ~$TAMANO de datos"
echo "   Tiempo estimado: 3-5 minutos (depende del tamaño)"
echo ""
read -p "¿Continuar? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Operación cancelada por el usuario"
    exit 0
fi

echo ""
echo "================================================================================"
echo "🔄 Copiando datos..."
echo "================================================================================"

# Crear directorio destino
mkdir -p "$DESTINO"

# Copiar con rsync (muestra progreso)
if command -v rsync &> /dev/null; then
    echo "   Usando rsync (con progreso)..."
    rsync -ah --progress "$ORIGEN/" "$DESTINO/"
else
    echo "   Usando cp (sin progreso)..."
    cp -r "$ORIGEN/"* "$DESTINO/"
fi

echo ""
echo "================================================================================"
echo "✅ COPIA COMPLETADA"
echo "================================================================================"

# Verificar estructura
echo ""
echo "📂 Estructura de directorios:"
ls -lh "$DESTINO"

echo ""
echo "📊 Estadísticas:"
echo "   Train: $(find "$DESTINO/train" -type f 2>/dev/null | wc -l) archivos"
echo "   Val:   $(find "$DESTINO/val" -type f 2>/dev/null | wc -l) archivos"
echo "   Test:  $(find "$DESTINO/test" -type f 2>/dev/null | wc -l) archivos"

echo ""
echo "================================================================================"
echo "🎯 PRÓXIMO PASO: Actualizar config.py"
echo "================================================================================"
echo ""
echo "Edita config.py y cambia RUTA_DETECCION a:"
echo ""
echo "   RUTA_DETECCION = \"$DESTINO\""
echo ""
echo "Luego reinicia el entrenamiento:"
echo "   python3 scripts/entrenamiento/entrenar_deteccion.py"
echo ""
echo "⚡ Speedup esperado: 298ms/step → 150-180ms/step (2x más rápido)"
echo "================================================================================"
