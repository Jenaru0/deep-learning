# 🏗️ Aplicación Web - Detector de Fisuras

Interfaz gráfica web desarrollada con **Streamlit** para detectar fisuras en estructuras de concreto utilizando Deep Learning.

## 📋 Descripción

Esta aplicación proporciona una interfaz visual intuitiva que permite:

- ✅ Cargar imágenes de estructuras de concreto
- 🤖 Detectar automáticamente la presencia de fisuras
- 📊 Visualizar resultados con gráficos de confianza
- 💡 Obtener recomendaciones basadas en el análisis

## 🚀 Instalación

### Opción 1: Usar el entorno virtual existente

Si ya tienes configurado el entorno virtual del proyecto:

```bash
# Activar el entorno virtual
source venv/bin/activate  # En Linux/Mac/WSL
# o
venv\Scripts\activate  # En Windows

# Instalar Streamlit (si no está instalado)
pip install streamlit==1.31.0
```

### Opción 2: Instalar todas las dependencias

Si quieres crear un entorno específico para la aplicación web:

```bash
cd app_web
pip install -r requirements.txt
```

## ▶️ Ejecución

Para ejecutar la aplicación web:

```bash
# Desde la raíz del proyecto
streamlit run app_web/app.py
```

La aplicación se abrirá automáticamente en tu navegador en:

```
http://localhost:8501
```

### Ejecución en WSL (Windows Subsystem for Linux)

```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar la aplicación
streamlit run app_web/app.py
```

Si el navegador no se abre automáticamente, copia la URL que aparece en la terminal.

## 📖 Uso de la Aplicación

### Paso 1: Cargar Imagen

1. Haz clic en **"📁 Selecciona una imagen"**
2. Elige una foto en formato JPG, JPEG o PNG
3. Espera a que la imagen se cargue

### Paso 2: Análisis Automático

- El modelo procesará la imagen automáticamente
- Verás la imagen original y el resultado del análisis

### Paso 3: Interpretar Resultados

La aplicación mostrará:

#### ✅ Sin Fisura

- **Indicador verde**: No se detectan fisuras
- **Confianza**: Porcentaje de certeza del modelo
- **Recomendación**: Continuar con inspecciones de rutina

#### ⚠️ Fisura Detectada

- **Indicador rojo**: Se detectó una fisura
- **Confianza**: Porcentaje de certeza del modelo
- **Recomendación**: Acción según el nivel de confianza
  - 🔴 **Alta (>95%)**: Inspección urgente
  - 🟠 **Moderada-Alta (75-95%)**: Programar inspección
  - 🟡 **Moderada (50-75%)**: Monitoreo y seguimiento

### Paso 4: Ajustar Configuración (Opcional)

En la barra lateral izquierda puedes:

- **Umbral de Decisión**: Ajustar la sensibilidad del detector
  - Valores bajos (0.3): Más sensible, detecta más fisuras
  - Valores altos (0.7): Menos sensible, solo fisuras claras
  - Valor por defecto: **0.5** (equilibrado)

## 📊 Características de la Interfaz

### Panel Lateral

- ℹ️ **Información del modelo**: Métricas de rendimiento
- ⚙️ **Configuración**: Ajuste del umbral de decisión
- 📊 **Métricas**: Precisión, Recall, F1-Score, AUC

### Área Principal

- 📷 **Visualización**: Imagen original en tamaño completo
- 🤖 **Resultado**: Diagnóstico con nivel de confianza
- 📊 **Gráfico**: Distribución de probabilidades
- 📋 **Interpretación**: Análisis detallado y recomendaciones
- 🔬 **Detalles técnicos**: Información expandible para usuarios avanzados

## 💡 Consejos para Mejores Resultados

### Calidad de Imagen

- ✅ Usa **buena iluminación** (luz natural o artificial uniforme)
- ✅ Evita **sombras fuertes** que puedan confundir al modelo
- ✅ **Enfoca bien** la zona de interés
- ❌ Evita fotos borrosas o con movimiento

### Encuadre

- ✅ Toma la foto **perpendicular** a la superficie
- ✅ Mantén una **distancia adecuada** (0.5 - 2 metros)
- ✅ **Centra** la fisura sospechosa en el encuadre
- ❌ Evita ángulos muy inclinados

### Formato

- ✅ Formatos soportados: **JPG, JPEG, PNG**
- ✅ Resolución mínima: **224x224 píxeles**
- ✅ Máximo recomendado: **4096x4096 píxeles**
- ❌ No uses formatos como GIF, BMP o TIFF

## 🔧 Solución de Problemas

### Error: "No se encontró ningún modelo entrenado"

**Solución**: Asegúrate de haber entrenado un modelo primero

```bash
python3 scripts/entrenamiento/entrenar_deteccion.py
```

### Error: "Module 'streamlit' not found"

**Solución**: Instala Streamlit

```bash
pip install streamlit==1.31.0
```

### La aplicación no se abre en el navegador

**Solución**: Copia manualmente la URL que aparece en la terminal (generalmente `http://localhost:8501`)

### Error al cargar imágenes muy grandes

**Solución**: Redimensiona la imagen antes de subirla. Máximo recomendado: 4096x4096 px

### El modelo es muy sensible o poco sensible

**Solución**: Ajusta el **umbral de decisión** en la barra lateral:

- **Más sensible** (detecta más fisuras): Umbral = 0.3 - 0.4
- **Menos sensible** (solo fisuras claras): Umbral = 0.6 - 0.7
- **Equilibrado** (recomendado): Umbral = 0.5

## 📈 Rendimiento del Modelo

El modelo utilizado tiene las siguientes métricas en el conjunto de prueba:

| Métrica       | Valor  |
| ------------- | ------ |
| **Precisión** | 94.36% |
| **Recall**    | 99.64% |
| **F1-Score**  | 96.77% |
| **AUC**       | 94.13% |

- **Alta precisión**: 94.36% de las predicciones son correctas
- **Alto recall**: 99.64% de las fisuras son detectadas (solo 0.36% se pierden)
- **Equilibrio**: F1-Score de 96.77% demuestra balance entre precisión y recall

## ⚠️ Advertencias Importantes

### Uso Responsable

Este sistema es una **herramienta de apoyo** y **NO reemplaza** la inspección profesional de un ingeniero estructural certificado.

### Limitaciones

- El modelo fue entrenado con el dataset SDNET2018 (estructuras de concreto)
- Puede no funcionar correctamente en otros materiales (madera, metal, etc.)
- No proporciona mediciones de ancho, profundidad u orientación de fisuras
- No evalúa la severidad estructural (solo presencia/ausencia)

### Recomendaciones

- ✅ Úsalo como **screening inicial**
- ✅ Combínalo con **inspección visual** humana
- ✅ Ante resultados positivos, **consulta siempre** con un ingeniero
- ❌ **No tomes decisiones críticas** basándote únicamente en esta herramienta

## 📁 Estructura de Archivos

```
app_web/
├── app.py              # Aplicación principal de Streamlit
├── requirements.txt    # Dependencias específicas de la app
└── README.md          # Este archivo
```

## 🛠️ Personalización

### Cambiar el Puerto

Por defecto, Streamlit usa el puerto 8501. Para cambiarlo:

```bash
streamlit run app_web/app.py --server.port 8080
```

### Modo Oscuro

Para ejecutar en modo oscuro:

```bash
streamlit run app_web/app.py --theme.base dark
```

### Configuración Avanzada

Puedes crear un archivo `.streamlit/config.toml` en la raíz del proyecto:

```toml
[server]
port = 8501
headless = false

[theme]
base = "light"
primaryColor = "#e74c3c"
```

## 🔗 Recursos Adicionales

- **Documentación de Streamlit**: https://docs.streamlit.io/
- **TensorFlow**: https://www.tensorflow.org/
- **Guía de predicción (CLI)**: `docs/GUIA_USO_PREDICCION.md`
- **Análisis del proyecto**: `docs/ANALISIS_PROYECTO_COMPLETO.md`

## 📞 Soporte

Si encuentras problemas:

1. Revisa la sección de **Solución de Problemas** arriba
2. Verifica que todas las dependencias estén instaladas
3. Asegúrate de tener un modelo entrenado en `modelos/deteccion/`
4. Revisa los logs de la aplicación en la terminal

## 📝 Notas Técnicas

- **Framework**: Streamlit 1.31.0
- **Modelo**: MobileNetV2 (Transfer Learning)
- **Backend**: TensorFlow 2.17.0
- **Entrada**: Imágenes RGB de 224x224 píxeles
- **Salida**: Probabilidad binaria (FISURA / SIN FISURA)
- **Cache**: El modelo se carga una sola vez usando `@st.cache_resource`

---

**Desarrollado con ❤️ para la detección de fisuras en estructuras**

_Última actualización: Octubre 2025_
