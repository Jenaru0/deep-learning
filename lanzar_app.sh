#!/bin/bash

# Script para lanzar la aplicación web de detección de fisuras
# Uso: ./lanzar_app.sh

echo "🏗️  Sistema de Detección de Fisuras"
echo "===================================="
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -f "config.py" ]; then
    echo "❌ Error: Ejecuta este script desde la raíz del proyecto"
    exit 1
fi

# Verificar que existe el modelo
if [ ! -d "modelos/deteccion" ]; then
    echo "❌ Error: No se encontró el directorio de modelos"
    echo "Por favor, entrena un modelo primero con:"
    echo "  python3 scripts/entrenamiento/entrenar_deteccion.py"
    exit 1
fi

# Buscar modelos
modelo_final=$(find modelos/deteccion -name "modelo_deteccion_final.keras" -o -name "modelo_deteccion_final.h5" 2>/dev/null | head -1)
modelo_best=$(find modelos/deteccion -name "best_model_*.keras" -o -name "best_model_*.h5" 2>/dev/null | head -1)

if [ -z "$modelo_final" ] && [ -z "$modelo_best" ]; then
    echo "⚠️  Advertencia: No se encontró ningún modelo entrenado"
    echo "La aplicación intentará cargar el modelo, pero puede fallar."
    echo ""
    read -p "¿Deseas continuar de todos modos? (s/n): " continuar
    if [ "$continuar" != "s" ] && [ "$continuar" != "S" ]; then
        exit 0
    fi
else
    echo "✅ Modelo encontrado"
fi

echo ""
echo "📦 Verificando entorno virtual..."

# Activar entorno virtual si existe
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Entorno virtual activado"
else
    echo "⚠️  No se encontró entorno virtual"
    echo "Usando Python del sistema"
fi

echo ""
echo "🔍 Verificando Streamlit..."

# Verificar que streamlit está instalado
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit no está instalado"
    echo ""
    read -p "¿Deseas instalar Streamlit ahora? (s/n): " instalar
    if [ "$instalar" = "s" ] || [ "$instalar" = "S" ]; then
        echo "📥 Instalando Streamlit..."
        pip install streamlit==1.31.0
    else
        echo "Abortando..."
        exit 1
    fi
else
    echo "✅ Streamlit está instalado"
fi

echo ""
echo "🚀 Lanzando aplicación web..."
echo ""
echo "La aplicación se abrirá en: http://localhost:8501"
echo "Presiona Ctrl+C para detener la aplicación"
echo ""
echo "----------------------------------------"
echo ""

# Lanzar aplicación
streamlit run app_web/app.py
