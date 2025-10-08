# 🏗️ Investigación: Detección de Fisuras en Edificaciones con Deep Learning

## 📋 Descripción del Proyecto

Sistema de visión computacional basado en Deep Learning para detectar, clasificar y evaluar fisuras y grietas en edificaciones.

## 📂 Estructura del Proyecto

```
investigacion_fisuras/
│
├── datasets/              # Datasets de imágenes
│   ├── CRACK500/         # Dataset CRACK500
│   └── SDNET2018/        # Dataset SDNET2018 (D, P, W)
│
├── analisis/             # Scripts y resultados de análisis
│
└── modelos/              # Modelos entrenados
```

## 🎯 Objetivos

### Objetivo General
Desarrollar un sistema de visión computacional basado en Deep Learning que permita identificar, clasificar y evaluar fisuras y grietas en edificaciones.

### Objetivos Específicos
1. Recolectar y preparar datasets de imágenes de fisuras
2. Seleccionar arquitectura de red neuronal convolucional apropiada
3. Clasificar fisuras según tipo y características geométricas
4. Evaluar confiabilidad del sistema

## 📊 Datasets Disponibles

### CRACK500
- Imágenes de fisuras en pavimento
- Incluye máscaras de segmentación
- Subdivisión: train/val/test

### SDNET2018
- Categorías: D (Deck), P (Pavement), W (Wall)
- Clasificación binaria: Con fisura (C) / Sin fisura (U)
- Gran volumen de imágenes

## 🚀 Comenzar

1. Ejecutar análisis inicial de datasets:
```bash
python analizar_datasets.py
```

## 📝 Estado Actual

- ✅ Datasets descargados y organizados
- ⏳ Análisis exploratorio pendiente
- ⏳ Preprocesamiento de datos
- ⏳ Desarrollo del modelo

---
**Fecha de inicio:** Octubre 2025  
**Investigador:** [Tu nombre]
