# 🔬 PLAN DETALLADO: Análisis de Parámetros Estructurales de Fisuras

**Fecha:** 10 de octubre, 2025  
**Objetivo:** Implementar análisis cuantitativo de fisuras (ancho, orientación, profundidad)  
**Método:** Segmentación semántica con U-Net sobre dataset CRACK500

---

## 📊 ANÁLISIS DE VIABILIDAD

### ✅ **LO QUE TENEMOS:**

#### 1. **Dataset CRACK500 - Listo para Segmentación**

```
✅ 3,806 imágenes JPG (640x360 pixels)
✅ 3,364 máscaras PNG (ground truth binario)
✅ ~88% de imágenes tienen máscara asociada
✅ Máscaras binarias: [False=fondo, True=fisura]
✅ Densidad promedio de fisura: ~10% por imagen
✅ Ya está descargado y organizado
```

#### 2. **Infraestructura Existente**

```
✅ Entorno Python configurado con TensorFlow 2.17.0
✅ GPU RTX 2050 optimizada (Mixed Precision + XLA)
✅ Scripts de entrenamiento modulares y reutilizables
✅ Pipeline de evaluación con visualizaciones
✅ Sistema de callbacks (early stopping, LR reduction)
```

#### 3. **Conocimiento del Equipo**

```
✅ Experiencia con Transfer Learning (MobileNetV2)
✅ Manejo de data augmentation
✅ Optimización GPU (2.5x speed-up demostrado)
✅ Evaluación con métricas avanzadas (IoU, Dice)
✅ Interfaz web funcional (Streamlit)
```

---

## 🎯 OBJETIVOS ESPECÍFICOS

### **Módulo 1: Segmentación Semántica (Base)**

Entrenar modelo U-Net para **detectar píxeles de fisuras**

**Entradas:** Imagen RGB 256x256  
**Salidas:** Máscara binaria 256x256 (0=fondo, 1=fisura)  
**Métrica principal:** IoU (Intersection over Union) > 0.75

### **Módulo 2: Medición de Ancho**

Calcular **ancho promedio y máximo** de fisuras en milímetros

**Método:**

1. Esqueletonizar máscara predicha (morfología)
2. Calcular distancia de cada píxel al borde de la fisura
3. Promediar distancias en la máscara
4. Convertir píxeles → mm con calibración

**Salida:** `ancho_promedio=2.3mm, ancho_max=5.1mm`

### **Módulo 3: Detección de Orientación**

Clasificar fisuras en **Horizontal / Vertical / Diagonal**

**Método:**

1. Aplicar transformada de Hough sobre máscara
2. Detectar líneas principales
3. Calcular ángulo dominante
4. Clasificar:
   - H: 0°-30° o 150°-180°
   - V: 60°-120°
   - D: 30°-60° o 120°-150°

**Salida:** `orientacion='Diagonal', angulo=45°`

### **Módulo 4: Estimación de Profundidad (Proxy)**

Estimar **profundidad relativa** basado en intensidad de píxeles

**Método (heurístico):**

1. Analizar intensidad promedio en región de fisura
2. Fisuras oscuras = profundas
3. Fisuras claras = superficiales
4. Clasificar: Superficial / Moderada / Profunda

**Limitación:** Sin datos de profundidad real, solo proxy visual  
**Salida:** `profundidad='Moderada', confianza=0.7`

---

## 📐 ARQUITECTURA TÉCNICA

### **Modelo de Segmentación: U-Net**

```
Encoder (Contracting Path):
  Input: 256x256x3
  ├─ Conv2D(64) + ReLU + Conv2D(64) + ReLU → MaxPool → 128x128
  ├─ Conv2D(128) + ReLU + Conv2D(128) + ReLU → MaxPool → 64x64
  ├─ Conv2D(256) + ReLU + Conv2D(256) + ReLU → MaxPool → 32x32
  └─ Conv2D(512) + ReLU + Conv2D(512) + ReLU → MaxPool → 16x16 (bottleneck)

Decoder (Expanding Path):
  ├─ UpSampling + Conv2D(256) + Concatenate[skip] → 32x32
  ├─ UpSampling + Conv2D(128) + Concatenate[skip] → 64x64
  ├─ UpSampling + Conv2D(64) + Concatenate[skip] → 128x128
  └─ UpSampling + Conv2D(32) + Concatenate[skip] → 256x256

Output: Conv2D(1, activation='sigmoid') → 256x256x1 (máscara)

Parámetros: ~7.7M (vs 2.2M del MobileNetV2)
Loss: Binary Crossentropy + Dice Loss (combinado)
Optimizer: Adam, LR=1e-4
```

**¿Por qué U-Net?**

- ✅ Estándar de oro para segmentación médica/industrial
- ✅ Skip connections preservan detalles finos (fisuras delgadas)
- ✅ Funciona bien con pocos datos (~3K imágenes)
- ✅ Arquitectura probada en detección de grietas

---

## 📦 RECURSOS NECESARIOS

### **Datos**

```
✅ YA DISPONIBLE: 3,364 pares imagen-máscara
📊 Split propuesto:
   - Train: 2,691 (80%)
   - Val: 336 (10%)
   - Test: 337 (10%)
```

### **Computación**

```
✅ GPU: RTX 2050 (4GB VRAM)
   - U-Net (256x256, batch=8): ~3.2GB VRAM ✅ CABE
   - Mixed Precision: ~2.1GB VRAM ✅ ÓPTIMO

⏱️ Tiempo estimado de entrenamiento:
   - Sin optimización: ~4-5 horas (50 epochs)
   - Con Mixed Precision + XLA: ~2-2.5 horas ✅
```

### **Software**

```
✅ TensorFlow 2.17.0 (instalado)
✅ OpenCV para procesamiento morfológico
✅ scikit-image para esqueletonización
✅ scipy para transformada de Hough
```

---

## 🗓️ PLAN DE IMPLEMENTACIÓN

### **FASE 1: Preparación de Datos (Día 1 - 4 horas)**

**Tarea 1.1: Organizar dataset para segmentación**

```bash
datos/procesados/segmentacion/
├── train/
│   ├── images/  # 2,691 JPG
│   └── masks/   # 2,691 PNG
├── val/
│   ├── images/  # 336 JPG
│   └── masks/   # 336 PNG
└── test/
    ├── images/  # 337 JPG
    └── masks/   # 337 PNG
```

**Script:** `scripts/preprocesamiento/preparar_crack500.py`

- Leer pares válidos imagen-máscara
- Dividir estratificadamente (80/10/10)
- Copiar a estructura organizada
- Validar integridad (SHA256)

**Tarea 1.2: Pipeline de carga con tf.data**

```python
def load_segmentation_data():
    # Data augmentation:
    - Random flip (horizontal/vertical)
    - Random rotation (±15°)
    - Random brightness (±20%)
    - Elastic deformation (para fisuras)

    # Normalización:
    - Imágenes: [0, 255] → [0, 1]
    - Máscaras: {False, True} → {0, 1}
```

---

### **FASE 2: Entrenamiento del Modelo U-Net (Día 2 - 3 horas)**

**Tarea 2.1: Implementar arquitectura U-Net**

**Script:** `scripts/entrenamiento/entrenar_segmentacion.py`

```python
def construir_unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder
    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D()(c1)

    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D()(c2)

    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D()(c3)

    c4 = conv_block(p3, 512)
    p4 = MaxPooling2D()(c4)

    # Bottleneck
    c5 = conv_block(p4, 1024)

    # Decoder con skip connections
    u6 = UpSampling2D()(c5)
    u6 = Concatenate()([u6, c4])
    c6 = conv_block(u6, 512)

    u7 = UpSampling2D()(c6)
    u7 = Concatenate()([u7, c3])
    c7 = conv_block(u7, 256)

    u8 = UpSampling2D()(c7)
    u8 = Concatenate()([u8, c2])
    c8 = conv_block(u8, 128)

    u9 = UpSampling2D()(c8)
    u9 = Concatenate()([u9, c1])
    c9 = conv_block(u9, 64)

    # Output
    outputs = Conv2D(1, 1, activation='sigmoid')(c9)

    model = Model(inputs, outputs)
    return model
```

**Tarea 2.2: Loss function combinada**

```python
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - (numerator + 1) / (denominator + 1)
```

**Tarea 2.3: Configuración de entrenamiento**

```python
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=combined_loss,
    metrics=['accuracy', iou_metric, dice_coefficient]
)

callbacks = [
    ModelCheckpoint('best_unet.keras', save_best_only=True),
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

# Optimizaciones GPU
tf.config.optimizer.set_jit(True)  # XLA
policy = tf.keras.mixed_precision.Policy('mixed_float16')
```

**Métricas de evaluación:**

- **IoU (Intersection over Union):** > 0.75 objetivo
- **Dice Coefficient:** > 0.85 objetivo
- **Pixel Accuracy:** > 95% objetivo
- **Precision/Recall** para píxeles de fisura

---

### **FASE 3: Post-procesamiento y Mediciones (Día 3 - 4 horas)**

**Tarea 3.1: Medición de ancho**

**Script:** `scripts/analisis/medir_parametros.py`

```python
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

def medir_ancho_fisura(mask, pixels_per_mm=10):
    """
    Mide el ancho de fisura en la máscara.

    Args:
        mask: Máscara binaria (0=fondo, 1=fisura)
        pixels_per_mm: Factor de calibración (depende de distancia de cámara)

    Returns:
        dict: {ancho_promedio_mm, ancho_max_mm, ancho_min_mm}
    """
    # 1. Esqueletonizar para encontrar eje central
    skeleton = skeletonize(mask > 0.5)

    # 2. Calcular distancia de cada píxel al borde
    distance_map = distance_transform_edt(mask > 0.5)

    # 3. Extraer anchos en el esqueleto
    anchos_pixels = distance_map[skeleton] * 2  # *2 porque distance es radio

    # 4. Convertir a mm
    ancho_prom_mm = np.mean(anchos_pixels) / pixels_per_mm
    ancho_max_mm = np.max(anchos_pixels) / pixels_per_mm
    ancho_min_mm = np.min(anchos_pixels) / pixels_per_mm

    return {
        'ancho_promedio_mm': round(ancho_prom_mm, 2),
        'ancho_maximo_mm': round(ancho_max_mm, 2),
        'ancho_minimo_mm': round(ancho_min_mm, 2),
        'area_total_mm2': round(np.sum(mask > 0.5) / (pixels_per_mm**2), 2)
    }
```

**Tarea 3.2: Detección de orientación**

```python
def detectar_orientacion(mask):
    """
    Detecta la orientación dominante de la fisura.

    Returns:
        dict: {orientacion, angulo, confianza}
    """
    # 1. Detección de bordes
    edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)

    # 2. Transformada de Hough para detectar líneas
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

    if lines is None:
        return {'orientacion': 'Indefinida', 'angulo': None, 'confianza': 0.0}

    # 3. Calcular ángulo dominante
    angulos = []
    for line in lines:
        rho, theta = line[0]
        angulo_deg = np.degrees(theta)
        angulos.append(angulo_deg)

    angulo_dominante = np.median(angulos)

    # 4. Clasificar orientación
    if (0 <= angulo_dominante <= 30) or (150 <= angulo_dominante <= 180):
        orientacion = 'Horizontal'
    elif 60 <= angulo_dominante <= 120:
        orientacion = 'Vertical'
    else:
        orientacion = 'Diagonal'

    # 5. Calcular confianza (clustering de ángulos)
    std_angulos = np.std(angulos)
    confianza = max(0, 1 - std_angulos / 90)  # Menor std = mayor confianza

    return {
        'orientacion': orientacion,
        'angulo': round(angulo_dominante, 1),
        'confianza': round(confianza, 2)
    }
```

**Tarea 3.3: Estimación de profundidad (proxy visual)**

```python
def estimar_profundidad(imagen, mask):
    """
    Estima profundidad relativa basándose en intensidad de píxeles.

    LIMITACIÓN: Es un proxy visual, NO mide profundidad real.
    Se requeriría visión estéreo o sensor de profundidad para precisión.

    Returns:
        dict: {profundidad, intensidad_promedio, confianza}
    """
    # 1. Extraer región de fisura
    fisura_region = imagen[mask > 0.5]

    if len(fisura_region) == 0:
        return {'profundidad': 'Desconocida', 'confianza': 0.0}

    # 2. Calcular intensidad promedio en escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    intensidad_fisura = np.mean(gray[mask > 0.5])
    intensidad_fondo = np.mean(gray[mask <= 0.5])

    # 3. Contraste relativo
    contraste = abs(intensidad_fisura - intensidad_fondo)

    # 4. Clasificación heurística
    if intensidad_fisura < 80 and contraste > 50:
        profundidad = 'Profunda'
        confianza = 0.7
    elif intensidad_fisura < 120:
        profundidad = 'Moderada'
        confianza = 0.6
    else:
        profundidad = 'Superficial'
        confianza = 0.5

    return {
        'profundidad': profundidad,
        'intensidad_promedio': round(intensidad_fisura, 1),
        'contraste': round(contraste, 1),
        'confianza': confianza
    }
```

---

### **FASE 4: Integración con Interfaz Web (Día 4 - 3 horas)**

**Tarea 4.1: Actualizar app.py con modo segmentación**

```python
# En app_web/app.py añadir:

@st.cache_resource
def cargar_modelo_segmentacion():
    """Carga modelo U-Net para segmentación."""
    modelo_path = Path(MODELOS_DIR) / "segmentacion" / "best_unet.keras"
    if modelo_path.exists():
        return tf.keras.models.load_model(modelo_path, compile=False)
    return None

def analizar_parametros_fisura(imagen, modelo_det, modelo_seg):
    """
    Análisis completo de fisura.

    Returns:
        dict: {
            tiene_fisura: bool,
            confianza_deteccion: float,
            mascara: np.array,
            ancho_mm: float,
            orientacion: str,
            profundidad: str,
            visualizacion: PIL.Image
        }
    """
    # 1. Detección binaria
    tiene_fisura, conf_det, _ = predecir(modelo_det, imagen)

    if not tiene_fisura or modelo_seg is None:
        return {'tiene_fisura': False}

    # 2. Segmentación
    img_prep = preprocesar_para_segmentacion(imagen)
    mascara = modelo_seg.predict(img_prep, verbose=0)[0, :, :, 0]

    # 3. Análisis de parámetros
    ancho_info = medir_ancho_fisura(mascara)
    orient_info = detectar_orientacion(mascara)
    prof_info = estimar_profundidad(np.array(imagen), mascara)

    # 4. Visualización
    overlay = crear_overlay_mascara(imagen, mascara)

    return {
        'tiene_fisura': True,
        'confianza_deteccion': conf_det,
        'mascara': mascara,
        'ancho_promedio_mm': ancho_info['ancho_promedio_mm'],
        'ancho_maximo_mm': ancho_info['ancho_maximo_mm'],
        'orientacion': orient_info['orientacion'],
        'angulo': orient_info['angulo'],
        'profundidad': prof_info['profundidad'],
        'visualizacion': overlay
    }
```

**Tarea 4.2: Diseño de panel de resultados**

```python
# En la interfaz Streamlit mostrar:

if resultado['tiene_fisura']:
    st.success("✅ FISURA DETECTADA - Análisis Detallado")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Ancho Promedio",
            f"{resultado['ancho_promedio_mm']} mm",
            f"Max: {resultado['ancho_maximo_mm']} mm"
        )

    with col2:
        st.metric(
            "Orientación",
            resultado['orientacion'],
            f"{resultado['angulo']}°"
        )

    with col3:
        st.metric(
            "Profundidad (Est.)",
            resultado['profundidad'],
            delta=None
        )

    # Visualización de máscara
    st.image(resultado['visualizacion'], caption="Segmentación de Fisura")
```

---

## ⏱️ CRONOGRAMA COMPLETO

| Día       | Fase          | Tareas                                | Tiempo       | Entregable                     |
| --------- | ------------- | ------------------------------------- | ------------ | ------------------------------ |
| **1**     | Preparación   | Organizar CRACK500, pipeline de carga | 4h           | `preparar_crack500.py` ✅      |
| **2**     | Entrenamiento | Implementar U-Net, entrenar modelo    | 3h           | `modelo_segmentacion.keras` ✅ |
| **3**     | Análisis      | Medición de parámetros, validación    | 4h           | `medir_parametros.py` ✅       |
| **4**     | Integración   | Actualizar interfaz web, testing      | 3h           | App web actualizada ✅         |
| **TOTAL** |               |                                       | **14 horas** | Sistema completo 🎉            |

---

## 📊 MÉTRICAS DE ÉXITO

### **Modelo de Segmentación**

- ✅ IoU > 0.75 en test set
- ✅ Dice Coefficient > 0.85
- ✅ Tiempo de inferencia < 100ms por imagen

### **Mediciones de Parámetros**

- ✅ Ancho: error < 0.5mm vs medición manual
- ✅ Orientación: concordancia > 90%
- ✅ Profundidad: clasificación correcta > 70% (limitado por falta de ground truth)

### **Interfaz de Usuario**

- ✅ Análisis completo en < 2 segundos
- ✅ Visualización clara de máscara
- ✅ Métricas presentadas de forma comprensible

---

## 🚧 LIMITACIONES Y CONSIDERACIONES

### **Limitaciones Técnicas**

1. **Calibración de Escala**

   - ⚠️ `pixels_per_mm` depende de distancia de cámara
   - Solución: Incluir objeto de referencia (moneda, regla) o EXIF metadata
   - Por defecto: asumir 10 pixels/mm (ajustable)

2. **Profundidad es Proxy Visual**

   - ⚠️ NO es medición real de profundidad
   - ⚠️ Requeriría visión estéreo o sensor LiDAR
   - ⚠️ Solo clasifica: superficial/moderada/profunda basándose en intensidad
   - Documentar claramente esta limitación

3. **Dataset de Pavimento vs Estructuras**

   - ⚠️ CRACK500 es principalmente pavimento
   - ⚠️ Puede no generalizar perfectamente a muros/decks
   - Solución: Fine-tuning con imágenes de SDNET2018 (si se anotan máscaras)

4. **Evolución Temporal**
   - ❌ No implementable sin dataset longitudinal
   - ❌ Requeriría imágenes del mismo lugar en diferentes fechas
   - Postponer para trabajo futuro

### **Requerimientos de Datos Adicionales**

Para mejorar precisión:

- Ground truth de anchos reales (mediciones manuales)
- Imágenes con escala conocida (regla en foto)
- Anotaciones de profundidad por expertos

---

## ✅ VIABILIDAD: **ALTA (95%)**

### **Resumen Ejecutivo:**

**¿SE PUEDE HACER?** → **SÍ ✅**

**¿CON LO QUE TENEMOS?** → **SÍ ✅**

**¿EN CUÁNTO TIEMPO?** → **14 horas (2 días laborales)** ✅

**¿QUÉ OBTENDREMOS?**

1. ✅ **Segmentación precisa** de fisuras pixel por pixel
2. ✅ **Medición de ancho** en milímetros (con calibración)
3. ✅ **Clasificación de orientación** (H/V/D) con ángulo
4. ✅ **Estimación de profundidad** (proxy visual, limitado)
5. ✅ **Interfaz web actualizada** con análisis completo
6. ✅ **Visualizaciones** de máscaras superpuestas

**¿QUÉ NO PODREMOS HACER (todavía)?**

1. ❌ **Análisis de evolución temporal** (falta dataset longitudinal)
2. ❌ **Medición de profundidad real** (requiere hardware especializado)
3. ❌ **Clasificación de severidad estructural** (requiere expertos)

---

## 🎯 RECOMENDACIÓN FINAL

### **Proceder con Implementación: SÍ ✅**

**Razones:**

1. **Dataset disponible y adecuado** (3,364 pares imagen-máscara)
2. **Infraestructura probada** (GPU optimizada, 2.5x speed-up)
3. **Arquitectura estándar** (U-Net es estado del arte)
4. **Tiempo razonable** (14 horas distribuidas en 2-4 días)
5. **Alto impacto académico** (pasa de clasificación a medición cuantitativa)

**Próximo Paso:**

¿Quieres que empiece con la **Fase 1 (Preparación de Datos)** creando el script `scripts/preprocesamiento/preparar_crack500.py`?

Esto organizará el dataset CRACK500 en la estructura train/val/test necesaria para entrenar U-Net.

---

**Autor:** GitHub Copilot  
**Revisión:** Pendiente por usuario  
**Estado:** Plan completo - Listo para ejecución ✅
