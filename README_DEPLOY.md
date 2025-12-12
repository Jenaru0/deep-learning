# 🏗️ Sistema de Detección de Fisuras Estructurales

Sistema de visión computacional basado en Deep Learning para la detección, segmentación y análisis automático de fisuras en elementos de concreto.

## 🚀 Demo en Línea

**🔗 [Probar la aplicación](TU_URL_AQUI)**

## 📊 Características

### ✅ Detección Binaria (MobileNetV2)

- **Recall:** 99.64% (casi sin omisiones)
- **Precisión:** 94.36%
- **F1-Score:** 96.77%

### 🎯 Segmentación Semántica (U-Net Ligera)

- **IoU:** 60.51%
- **Dice:** 73.04%
- **Velocidad:** ~7.6 img/s en RTX 2050

### 📐 Medición de Parámetros

- Ancho promedio/máximo
- Orientación (horizontal/vertical/diagonal)
- Longitud y área afectada

## 🛠️ Tecnologías

- **Framework:** TensorFlow 2.15 + Keras
- **Interfaz:** Streamlit
- **Visión:** OpenCV, scikit-image
- **Datasets:** SDNET2018 (56K imgs), Crack500 (3.3K imgs)

## 📦 Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/TU_USUARIO/investigacion_fisuras.git
cd investigacion_fisuras

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements_streamlit.txt

# Lanzar aplicación
streamlit run app_web/app.py
```

## 📚 Estructura del Proyecto

```
investigacion_fisuras/
├── app_web/              # Aplicación Streamlit
├── modelos/              # Modelos entrenados (.keras)
├── scripts/              # Scripts de entrenamiento
├── docs/                 # Documentación
└── resultados/           # Visualizaciones
```

## 🎓 Autores

- Candela Vargas Aitor Baruc
- Godoy Bautista Denilson Miguel
- Molina Lazaro Eduardo Jeampier
- Napanga Ruiz Jhonatan Jesus

## 📄 Licencia

MIT License - Ver `LICENSE` para más detalles

## 🙏 Agradecimientos

- Dataset SDNET2018: Dorafshan et al. (Utah State University)
- Dataset Crack500: Yang et al. (2019)
