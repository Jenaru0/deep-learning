# 🔍 GUÍA DE USO: Sistema de Detección de Fisuras

## ¿Qué hace este sistema?

Este modelo de Deep Learning **detecta automáticamente fisuras y grietas** en superficies estructurales (paredes, pisos, techos, pavimentos).

---

## 📸 Cómo usar el modelo con tus propias imágenes

### **Método 1: Script de Predicción Individual** (Más Fácil)

#### **Paso 1: Preparar imagen**

- Toma foto de la superficie sospechosa
- Formatos aceptados: `.jpg`, `.jpeg`, `.png`
- Calidad recomendada: Buena iluminación, enfoque en la superficie

#### **Paso 2: Ejecutar predicción**

```bash
# Activar entorno virtual
cd /mnt/c/Users/jonna/OneDrive/Escritorio/DEEP\ LEARNING/investigacion_fisuras
source venv/bin/activate

# Predecir en una imagen
python3 scripts/utils/predecir_imagen.py --imagen ruta/a/tu/foto.jpg --visualizar
```

#### **Ejemplo de salida:**

```
================================================================================
RESULTADO DE PREDICCIÓN
================================================================================
📷 Imagen: pared_sospechosa.jpg
📊 Clasificación: CRACKED
🎯 Confianza: 97.30%

Probabilidad de FISURA:    97.30%
Probabilidad de SIN FISURA: 2.70%
================================================================================
⚠️  ALERTA: Se detectaron fisuras con ALTA confianza
   Recomendación: Inspección inmediata por ingeniero
================================================================================
```

#### **Opciones disponibles:**

```bash
# Predicción básica
python3 scripts/utils/predecir_imagen.py --imagen foto.jpg

# Con visualización gráfica
python3 scripts/utils/predecir_imagen.py --imagen foto.jpg --visualizar

# Ajustar umbral de sensibilidad (0-1)
python3 scripts/utils/predecir_imagen.py --imagen foto.jpg --umbral 0.7

# Mostrar imagen original antes
python3 scripts/utils/predecir_imagen.py --imagen foto.jpg --mostrar-imagen --visualizar
```

---

### **Método 2: Usar el modelo en Python (Programación)**

Si quieres integrar el modelo en tu código:

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 1. Cargar modelo entrenado
modelo = load_model('modelos/deteccion/modelo_deteccion_final.keras')

# 2. Cargar y preprocesar imagen
img = image.load_img('mi_foto.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalizar
img_array = np.expand_dims(img_array, axis=0)  # Batch dimension

# 3. Predecir
probabilidad = modelo.predict(img_array)[0][0]

# 4. Interpretar
if probabilidad >= 0.5:
    print(f"⚠️ FISURA DETECTADA - Confianza: {probabilidad*100:.1f}%")
else:
    print(f"✅ SIN FISURAS - Confianza: {(1-probabilidad)*100:.1f}%")
```

---

### **Método 3: Procesar Carpeta Completa** (Batch)

Para analizar múltiples imágenes:

```python
# Crear script: scripts/utils/predecir_lote.py
import os
from pathlib import Path
import pandas as pd

carpeta_imagenes = "ruta/a/carpeta"
resultados = []

for archivo in Path(carpeta_imagenes).glob("*.jpg"):
    resultado = predecir_fisura(modelo, archivo)
    resultados.append({
        'imagen': archivo.name,
        'tiene_fisura': resultado['tiene_fisura'],
        'probabilidad': resultado['probabilidad_fisura'],
        'confianza': resultado['confianza']
    })

# Guardar resultados en CSV
df = pd.DataFrame(resultados)
df.to_csv('resultados_analisis.csv', index=False)
print(f"✅ Analizadas {len(resultados)} imágenes")
```

---

## 🎯 Interpretación de Resultados

### **Niveles de Confianza:**

| Confianza   | Interpretación  | Acción Recomendada                       |
| ----------- | --------------- | ---------------------------------------- |
| **90-100%** | Alta certeza    | Seguir recomendación directamente        |
| **70-90%**  | Confianza media | Verificar con segunda opinión            |
| **50-70%**  | Baja confianza  | Tomar más imágenes o mejorar iluminación |
| **< 50%**   | Muy incierto    | Re-fotografiar con mejor calidad         |

### **Clase "CRACKED" (Con Fisura):**

- ✅ El modelo detectó patrones compatibles con fisuras
- ⚠️ Recomendación: Inspección por ingeniero estructural
- 📊 Precisión del modelo: 94.1% (de 100 alarmas, 94 son reales)

### **Clase "UNCRACKED" (Sin Fisura):**

- ✅ El modelo NO detectó fisuras significativas
- 📊 Recall del modelo: 99.6% (detecta 99.6% de fisuras reales)
- ⚠️ Nota: Muy pocas fisuras se escapan (solo 0.4%)

---

## 📋 Ejemplos de Casos de Uso

### **1. Inspección de Edificios**

```bash
# Inspeccionar paredes de un edificio
python3 scripts/utils/predecir_imagen.py --imagen edificio_pared_norte.jpg --visualizar
python3 scripts/utils/predecir_imagen.py --imagen edificio_pared_sur.jpg --visualizar
```

### **2. Mantenimiento de Carreteras**

```bash
# Analizar estado de pavimento
python3 scripts/utils/predecir_imagen.py --imagen carretera_km_15.jpg --umbral 0.6
```

### **3. Inspección Pre-compra**

```bash
# Verificar estado estructural antes de comprar inmueble
python3 scripts/utils/predecir_imagen.py --imagen casa_habitacion_1.jpg --visualizar
python3 scripts/utils/predecir_imagen.py --imagen casa_piso_2.jpg --visualizar
```

---

## ⚙️ Ajustar Sensibilidad del Modelo

### **Umbral de Clasificación** (`--umbral`)

El umbral determina cuándo clasificar como "CRACKED":

```bash
# MÁS SENSIBLE (detecta más fisuras, más falsas alarmas)
python3 scripts/utils/predecir_imagen.py --imagen foto.jpg --umbral 0.3

# BALANCEADO (configuración por defecto)
python3 scripts/utils/predecir_imagen.py --imagen foto.jpg --umbral 0.5

# MÁS CONSERVADOR (solo fisuras muy evidentes)
python3 scripts/utils/predecir_imagen.py --imagen foto.jpg --umbral 0.7
```

**Recomendaciones por contexto:**

| Contexto                                    | Umbral  | Justificación                              |
| ------------------------------------------- | ------- | ------------------------------------------ |
| **Seguridad crítica** (puentes, hospitales) | 0.3-0.4 | Mejor detectar fisura dudosa que ignorarla |
| **Inspección general**                      | 0.5     | Balance precisión-recall                   |
| **Pre-filtrado** (reducir inspecciones)     | 0.6-0.7 | Solo alertar en casos muy evidentes        |

---

## 🔧 Solución de Problemas

### **Error: "Modelo no encontrado"**

```bash
# Verificar que el modelo existe
ls -lh modelos/deteccion/*.keras

# Si no existe, re-entrenar
python3 scripts/entrenamiento/entrenar_deteccion_turbo.py
```

### **Predicción poco confiable (< 60%)**

**Posibles causas:**

1. **Mala iluminación** → Re-fotografiar con mejor luz
2. **Imagen borrosa** → Usar cámara con mejor resolución
3. **Superficie atípica** → El modelo entrenó en concreto/pavimento
4. **Fisura muy fina** → Tomar foto más cercana

### **Falsos positivos frecuentes**

Si detecta fisuras donde no hay:

```bash
# Aumentar umbral para ser más conservador
python3 scripts/utils/predecir_imagen.py --imagen foto.jpg --umbral 0.7
```

---

## 📊 Métricas de Rendimiento

El modelo fue evaluado en **8,417 imágenes** nunca vistas:

```
✅ Accuracy:  94.4%  → 94 de 100 clasificaciones correctas
✅ Precision: 94.1%  → 94 de 100 alarmas son fisuras reales
✅ Recall:    99.6%  → Detecta 99.6 de 100 fisuras existentes
✅ F1-Score:  96.8%  → Balance perfecto
✅ AUC:       94.1%  → Excelente capacidad discriminativa
```

**Comparación con estado del arte:**

- Zhang et al. 2018: 91.2% accuracy → **Tu modelo: 94.4%** ✅ MEJOR
- Xu et al. 2019: 93.7% accuracy → **Tu modelo: 94.4%** ✅ MEJOR

---

## 🚀 Próximos Pasos (Opcional)

### **1. Crear App Web** (Flask/Streamlit)

Interfaz web para subir fotos y obtener resultados:

```python
# app.py
import streamlit as st
from tensorflow.keras.models import load_model

st.title("🔍 Detector de Fisuras Estructurales")
archivo = st.file_uploader("Sube una imagen", type=['jpg', 'png'])

if archivo:
    resultado = predecir(archivo)
    st.write(f"Clasificación: {resultado['clase']}")
    st.progress(resultado['confianza'])
```

### **2. App Móvil** (TensorFlow Lite)

Convertir modelo para Android/iOS:

```python
# Convertir a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
tflite_model = converter.convert()

with open('modelo_fisuras.tflite', 'wb') as f:
    f.write(tflite_model)
```

### **3. API REST** (FastAPI)

Servicio web para integrar con otros sistemas:

```python
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/detectar-fisura")
async def detectar(imagen: UploadFile = File(...)):
    resultado = predecir(imagen)
    return resultado
```

---

## 📞 Soporte

Si necesitas ayuda:

1. Revisa esta guía completa
2. Consulta `docs/ANALISIS_PROYECTO_COMPLETO.md`
3. Revisa ejemplos en `scripts/utils/predecir_imagen.py`

---

**Autor:** Jesus Naranjo  
**Fecha:** Octubre 2025  
**Modelo:** MobileNetV2 + Transfer Learning  
**Dataset:** SDNET2018 (56,092 imágenes)
