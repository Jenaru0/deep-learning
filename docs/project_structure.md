# 📁 Estructura del Proyecto - Crack Detection

## 🏗️ Organización de Carpetas

```
crack_detection_project/
│
├── 📁 data/                          # 📊 Datos y Datasets
│   ├── raw/                          # Datos originales sin procesar
│   │   ├── SDNET2018/               # Dataset principal (Utah State)
│   │   └── CRACK500/                # Dataset complementario (Kaggle)
│   ├── processed/                    # Datos procesados y listos para entrenamiento
│   │   ├── train/                   # Datos de entrenamiento
│   │   ├── validation/              # Datos de validación
│   │   └── test/                    # Datos de prueba
│   └── external/                     # Datos externos adicionales
│
├── 📁 models/                        # 🤖 Modelos Entrenados
│   ├── detection/                    # Modelos de detección binaria
│   ├── classification/               # Modelos de clasificación
│   ├── segmentation/                 # Modelos de segmentación
│   └── checkpoints/                  # Checkpoints durante entrenamiento
│
├── 📁 notebooks/                     # 📓 Jupyter Notebooks
│   ├── 01_exploratory_analysis.ipynb    # Análisis exploratorio
│   ├── 02_data_preprocessing.ipynb      # Preprocesamiento de datos
│   ├── 03_model_training.ipynb          # Entrenamiento de modelos
│   ├── 04_model_evaluation.ipynb       # Evaluación y métricas
│   └── 05_results_visualization.ipynb  # Visualización de resultados
│
├── 📁 src/                           # 💻 Código Fuente
│   ├── data/                         # Scripts de procesamiento de datos
│   │   ├── __init__.py
│   │   ├── download_datasets.py      # Descarga automática de datasets
│   │   ├── data_loader.py           # Carga y organización de datos
│   │   └── preprocessing.py         # Preprocesamiento de imágenes
│   ├── models/                       # Definición de modelos
│   │   ├── __init__.py
│   │   ├── detection_model.py       # Modelo de detección
│   │   ├── classification_model.py  # Modelo de clasificación
│   │   └── trainer.py              # Clase para entrenar modelos
│   ├── utils/                        # Utilidades y funciones auxiliares
│   │   ├── __init__.py
│   │   ├── visualization.py         # Funciones de visualización
│   │   ├── metrics.py              # Cálculo de métricas
│   │   └── config.py               # Configuraciones del proyecto
│   └── app/                          # Aplicación final
│       ├── __init__.py
│       ├── streamlit_app.py         # Aplicación web
│       └── predictor.py            # Predictor principal
│
├── 📁 results/                       # 📈 Resultados y Outputs
│   ├── figures/                      # Gráficos y visualizaciones
│   ├── metrics/                      # Archivos con métricas
│   └── reports/                      # Reportes finales
│
├── 📁 docs/                          # 📚 Documentación
│   ├── setup.md                     # Guía de instalación
│   ├── usage.md                     # Guía de uso
│   └── api_reference.md             # Referencia de API
│
├── 📄 requirements.txt               # Dependencias de Python
├── 📄 .gitignore                    # Archivos a ignorar en Git
├── 📄 config.yaml                   # Configuración principal
└── 📄 README.md                     # Documentación principal
```

## 🎯 Propósito de cada Carpeta

### 📊 `/data/`

- **`raw/`**: Datasets originales descargados sin modificar
- **`processed/`**: Datos limpios y organizados para entrenamiento
- **`external/`**: Datos adicionales de fuentes externas

### 🤖 `/models/`

- Almacena modelos entrenados en formato `.h5` o `.pkl`
- Organizado por tipo de tarea (detección, clasificación, etc.)

### 📓 `/notebooks/`

- Notebooks para experimentación y análisis
- Numerados para seguir flujo lógico de desarrollo

### 💻 `/src/`

- Código modular y reutilizable
- Organizado por funcionalidad
- Siguiendo principios de programación orientada a objetos

### 📈 `/results/`

- Outputs del proyecto: gráficos, métricas, reportes
- Organizados por tipo de resultado

### 📚 `/docs/`

- Documentación técnica del proyecto
- Guías de instalación y uso

## ✅ Ventajas de esta Estructura

1. **🔍 Claridad**: Fácil navegación y comprensión
2. **🔄 Modularidad**: Código reutilizable y mantenible
3. **📏 Estándares**: Sigue convenciones de proyectos de ML
4. **🚀 Escalabilidad**: Fácil agregar nuevas funcionalidades
5. **👥 Colaboración**: Estructura familiar para equipos de ML
