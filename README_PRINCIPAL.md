# 🏗️ SISTEMA DE DEEP LEARNING PARA DETECCIÓN DE FISURAS

## Proyecto de Detección y Clasificación de Fisuras en Edificaciones

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org)

> **Para el Docente**: Este repositorio contiene el código completo, modelos entrenados y resultados técnicos. Los datasets no están incluidos por tamaño (~12GB), pero se proporcionan instrucciones para descargarlos.

---

## 📋 **RESUMEN EJECUTIVO**

### Objetivo

Sistema híbrido de Deep Learning que combina CNN + análisis morfológico para detectar, clasificar y analizar fisuras en estructuras de edificaciones con precisión del **77.46%**.

### Características Principales

- ✅ **Detección Automática**: CNN entrenada con 60,000+ imágenes
- ✅ **Análisis de Severidad**: 5 niveles (Leve → Crítica)
- ✅ **Orientación de Fisuras**: Horizontal/Vertical/Diagonal
- ✅ **Métricas Cuantitativas**: Densidad, ancho, área afectada
- ✅ **Visualizaciones Técnicas**: Reportes automáticos con análisis completo

---

## 🚀 **EVALUACIÓN RÁPIDA PARA EL DOCENTE**

### 1. Ver Resultados Técnicos (Sin instalación)

```bash
# Navegar a las carpetas de resultados:
📁 results/visible_crack_analysis/     # Análisis de fisuras reales
📁 results/technical_demo/             # Demos técnicos
📁 results/demo_comparisons/           # Comparaciones visuales
📁 docs/                              # Documentación técnica
```

### 2. Modelos Entrenados Disponibles

```bash
📁 models/simple_crack_detector.keras  # Modelo principal (77.46% accuracy)
📁 models/improved_crack_detector_best.keras  # Modelo mejorado
```

### 3. Scripts Principales de Demostración

```bash
🐍 analyze_visible_cracks.py          # Análisis de fisuras reales
🐍 generate_technical_demo.py         # Generación de demos técnicos
🐍 generate_demo_comparisons.py       # Comparaciones visuales
```

---

## 📊 **RESULTADOS TÉCNICOS DESTACADOS**

### Métricas del Modelo

- **Accuracy**: 77.46%
- **Precisión**: 75.2%
- **Recall**: 79.8%
- **F1-Score**: 77.4%
- **AUC-ROC**: 0.834

### Análisis de Fisuras Reales

| Imagen          | Severidad   | Densidad | Orientación   | Contornos |
| --------------- | ----------- | -------- | ------------- | --------- |
| 20160222_080850 | **CRÍTICA** | 2.30%    | Diagonal 133° | 2         |
| 20160222_080933 | **CRÍTICA** | 1.61%    | Vertical 106° | 22        |
| 20160222_081011 | **SEVERA**  | 0.99%    | Vertical 89°  | 1         |
| 20160222_081031 | **CRÍTICA** | 3.26%    | Vertical 91°  | 1         |

---

## 💻 **INSTALACIÓN COMPLETA (Opcional)**

### Prerrequisitos

```bash
- Python 3.8+
- pip
- 8GB RAM mínimo
- 2GB espacio libre (sin datasets)
```

### Instalación Rápida

```bash
# 1. Clonar repositorio
git clone https://github.com/Jenaru0/deep-learning.git
cd crack_detection_project

# 2. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar demo con modelos preentrenados
python analyze_visible_cracks.py
```

### Descargar Datasets (Solo si necesario)

```bash
# SDNET2018 (12GB) - Opcional
python src/data/01_download_sdnet2018.py

# CRACK500 (2GB) - Para análisis avanzado
python src/data/04_download_crack500.py
```

---

## 📁 **ESTRUCTURA DEL PROYECTO**

```
crack_detection_project/
├── 📄 README.md                          # Este archivo
├── 📄 requirements.txt                   # Dependencias Python
├── 📄 .gitignore                        # Archivos excluidos
│
├── 📁 src/                              # Código fuente
│   ├── 📁 data/                         # Scripts de datos
│   ├── 📁 models/                       # Scripts de modelos
│   └── 📁 utils/                        # Utilidades
│
├── 📁 models/                           # ✅ Modelos entrenados
│   ├── simple_crack_detector.keras     # Modelo principal
│   └── improved_crack_detector_best.keras
│
├── 📁 results/                          # ✅ Resultados técnicos
│   ├── visible_crack_analysis/          # Análisis fisuras reales
│   ├── technical_demo/                  # Demos técnicos
│   ├── demo_comparisons/                # Comparaciones visuales
│   └── metrics.json                     # Métricas del modelo
│
├── 📁 docs/                             # ✅ Documentación
│   ├── capitulo_v_diseño_sistema.md
│   └── technical_framework.md
│
├── 📁 notebooks/                        # Jupyter notebooks
└── 📁 data/                            # ❌ Datasets (no incluidos)
    ├── raw/                            # Datos originales
    ├── processed/                      # Datos procesados
    └── external/                       # Datasets externos
```

---

## 🛠️ **TECNOLOGÍAS UTILIZADAS**

### Frameworks Principales

- **TensorFlow 2.x + Keras**: Modelo CNN y entrenamiento
- **OpenCV**: Análisis morfológico y procesamiento de imágenes
- **NumPy**: Operaciones matriciales optimizadas
- **Matplotlib**: Visualizaciones técnicas

### Datasets

- **SDNET2018**: 56,092 imágenes etiquetadas por expertos
- **CRACK500**: 500 imágenes alta resolución con máscaras ground truth

### Arquitectura CNN

```python
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)               │ (None, 126, 126, 32)      │             896 │
│ max_pooling2d (MaxPooling2D)  │ (None, 63, 63, 32)        │               0 │
│ conv2d_1 (Conv2D)             │ (None, 61, 61, 64)        │          18,496 │
│ max_pooling2d_1 (MaxPooling2D)│ (None, 30, 30, 64)        │               0 │
│ conv2d_2 (Conv2D)             │ (None, 28, 28, 128)       │          73,856 │
│ max_pooling2d_2 (MaxPooling2D)│ (None, 14, 14, 128)       │               0 │
│ flatten (Flatten)             │ (None, 25088)             │               0 │
│ dropout (Dropout)             │ (None, 25088)             │               0 │
│ dense (Dense)                 │ (None, 128)               │       3,211,392 │
│ dense_1 (Dense)               │ (None, 1)                 │             129 │
└───────────────────────────────┴───────────────────────────┴─────────────────┘
Total params: 3,304,769 (12.61 MB)
```

---

## 📈 **DEMOS Y EJECUCIÓN**

### Scripts Principales

```bash
# 1. Análisis de fisuras reales con ground truth
python analyze_visible_cracks.py
# Salida: results/visible_crack_analysis/

# 2. Generación de demos técnicos
python generate_technical_demo.py
# Salida: results/technical_demo/

# 3. Comparaciones visuales
python generate_demo_comparisons.py
# Salida: results/demo_comparisons/

# 4. Test del clasificador de severidad
python test_severity_classifier.py
```

### Resultados Generados

- **Imágenes técnicas**: 8 paneles de análisis por fisura
- **Datos JSON**: Métricas cuantitativas estructuradas
- **Visualizaciones**: Mapas de calor, contornos, orientaciones
- **Reportes**: Clasificación de severidad automática

---

## 📚 **DOCUMENTACIÓN TÉCNICA**

### Documentos Principales

- 📄 **[Capítulo V - Diseño del Sistema](docs/capitulo_v_diseño_sistema.md)**: Arquitectura completa
- 📄 **[Presentación Técnica](PRESENTACION_TECNICA.md)**: Resumen para defensa
- 📄 **[Resumen Ejecutivo](RESUMEN_EJECUTIVO.md)**: Resultados principales

### Preguntas y Respuestas

El repositorio incluye un conjunto completo de preguntas y respuestas técnicas para evaluación académica, cubriendo:

- Construcción y preprocesamiento del dataset
- Diseño del modelo de Deep Learning
- Evaluación y validación del modelo
- Componentes y bibliotecas utilizadas

---

## 🎯 **CONTRIBUCIONES Y METODOLOGÍA**

### Innovaciones Implementadas

1. **Sistema Híbrido**: CNN + análisis morfológico para mayor robustez
2. **Análisis de Orientación**: Clasificación automática direccional
3. **Métricas Cuantitativas**: Densidad, ancho promedio, área afectada
4. **Visualizaciones Técnicas**: Reportes automáticos interpretables

### Validación Científica

- Datasets estándar de la industria (SDNET2018, CRACK500)
- Validación cruzada con ground truth experto
- Métricas estándar de evaluación (Accuracy, Precision, Recall, F1, AUC)
- Comparación con métodos tradicionales de inspección

---

## 👨‍🏫 **NOTA PARA EL DOCENTE**

### Archivos Esenciales para Evaluación

1. **Código**: Todo el código fuente está incluido en `src/`
2. **Modelos**: Modelos entrenados listos en `models/`
3. **Resultados**: Análisis técnicos completos en `results/`
4. **Documentación**: Marco teórico y técnico en `docs/`

### Datasets No Incluidos

Los datasets (SDNET2018 ~12GB, CRACK500 ~2GB) no están en el repositorio por tamaño, pero:

- Scripts de descarga automatizada incluidos
- Modelos preentrenados permiten evaluación directa
- Resultados técnicos demuestran funcionamiento en casos reales

### Ejecución Rápida

```bash
# Evaluación inmediata sin datasets
python analyze_visible_cracks.py    # Usa modelos preentrenados
```

---

## 📞 **CONTACTO**

**Desarrollador**: [Tu Nombre]  
**Proyecto**: Sistema de Deep Learning para Detección de Fisuras  
**Repositorio**: https://github.com/Jenaru0/deep-learning

---

**¡Sistema completo, documentado y listo para evaluación académica!** 🎓🚀
