# 📚 Marco Técnico y Bibliográfico - Detección de Fisuras

## **🎯 Clasificación Técnica de Fisuras y Grietas**

### **1. Diferenciación Fundamental**

#### **Fisuras (Superficiales)**

- **Ancho**: < 0.3 mm (hasta 1 mm en algunos casos)
- **Profundidad**: Superficial, no compromete integridad estructural
- **Riesgo**: Bajo, principalmente estético
- **Causas**: Retracción por secado, contracción normal, fraguado deficiente

#### **Grietas (Estructurales)**

- **Ancho**: > 0.3 mm (críticas > 3-4 mm, severas > 6 mm)
- **Profundidad**: Atraviesa elementos estructurales
- **Riesgo**: Alto, compromete seguridad estructural
- **Causas**: Asentamientos diferenciales, sobrecargas, fallas de diseño

---

## **📐 Clasificación por Orientación y Forma**

### **2.1 Fisuras Verticales**

- **Características**: Generalmente por contracciones o asentamientos leves
- **Riesgo**: Bajo si < 3mm, ALTO si ≥ 3-4mm
- **Causa**: Asentamientos, retracción de materiales

### **2.2 Fisuras Horizontales**

- **Características**: Más preocupantes, especialmente en parte baja de muros
- **Riesgo**: Moderado a Alto
- **Causa**: Problemas en cimientos, presión del terreno, filtraciones

### **2.3 Fisuras Diagonales/Zigzag**

- **Características**: Forma de "escalera" en paredes de ladrillo
- **Riesgo**: ALTO - Señal de alerta importante
- **Causa**: Asentamientos irregulares, torsión, actividad sísmica
- **Ubicación típica**: Esquinas de puertas y ventanas

### **2.4 Fisuras Reticuladas ("Piel de Cocodrilo")**

- **Características**: Red de líneas finas, patrón poligonal
- **Riesgo**: Bajo a Moderado
- **Causa**: Retracción plástica, cambios térmicos, secado rápido

---

## **⏰ Clasificación Temporal**

### **3.1 Fisuración en Estado Fresco**

- **Retracción Plástica**: 0.2-0.4mm, primeras 6 horas
- **Asentamiento Plástico**: Por obstáculos en armaduras
- **Movimiento del Subsuelo**: Durante el fraguado

### **3.2 Fisuración en Estado Endurecido**

- **Por Cargas Estructurales**: Flexión, cortante, torsión, compresión
- **Por Exposición Ambiental**: Retracción por secado, cambios térmicos
- **Por Corrosión**: Óxido ejerce presión, fisuras > 1mm

---

## **🔍 Métodos de Evaluación y Medición**

### **4.1 Herramientas de Medición**

- **Comparadores de Fisuras**: Microscopio con escala (precisión 0.025mm)
- **Tarjetas de Comparación**: Líneas de anchos específicos
- **Fisurómetros**: Medición en tiempo real (precisión 0.1mm)
- **Testigos de Yeso**: Detección de movimiento activo

### **4.2 Cronograma de Monitoreo**

- **7 días**: Primera evaluación
- **14 días**: Tendencia inicial
- **30 días**: Confirmación de actividad
- **60-90 días**: Caracterización completa

### **4.3 Ensayos No Destructivos (END)**

- **Ultrasonido**: Detección de fisuras internas
- **Pacómetro**: Localización de armaduras
- **Termografía**: Análisis térmico
- **Radiografía**: Discontinuidades internas

---

## **📊 Criterios de Gravedad y Urgencia**

### **5.1 Clasificación por Gravedad**

#### **LEVE**

- Ancho: < 0.3mm
- Ubicación: Superficial
- Evolución: Estable
- Acción: Monitoreo rutinario

#### **MODERADA**

- Ancho: 0.3-3mm
- Ubicación: Elementos no portantes
- Evolución: Lenta progresión
- Acción: Reparación planificada

#### **GRAVE**

- Ancho: > 3mm
- Ubicación: Elementos estructurales
- Evolución: Activa/progresiva
- Acción: **INTERVENCIÓN INMEDIATA**

### **5.2 Indicadores de Alerta Crítica**

- Fisuras diagonales anchas (> 3mm)
- Grietas horizontales con inclinación
- Separación entre ladrillos > 1.30cm
- Fisuras en cimientos
- Fisuras activas con crecimiento rápido

---

## **🔧 Implicaciones para el Modelo de Deep Learning**

### **6.1 Características a Detectar**

#### **Dimensionales**

- **Ancho**: 0.06mm - 25mm (rango SDNET2018)
- **Profundidad**: Estimación por análisis morfológico
- **Longitud**: Medición de contorno completo

#### **Morfológicas**

- **Orientación**: Vertical, horizontal, diagonal
- **Patrón**: Lineal, reticulada, ramificada
- **Densidad**: Fisuras por unidad de área

#### **Contextuales**

- **Ubicación**: Muro, viga, columna, losa
- **Superficie**: Deck, pavimento, pared
- **Entorno**: Interior, exterior, condiciones ambientales

### **6.2 Clasificación Multi-Etapa**

#### **Etapa 1: Detección Binaria**

- **Objetivo**: ¿Hay fisura presente?
- **Dataset**: SDNET2018 completo (49,363 imágenes)
- **Arquitectura**: EfficientNetB0 + Transfer Learning

#### **Etapa 2: Clasificación de Severidad**

```python
severity_classification = {
    "superficial": {
        "ancho_max": 0.3,  # mm
        "riesgo": "bajo",
        "accion": "monitoreo"
    },
    "moderada": {
        "ancho_range": [0.3, 3.0],  # mm
        "riesgo": "medio",
        "accion": "reparacion_planificada"
    },
    "estructural": {
        "ancho_min": 3.0,  # mm
        "riesgo": "alto",
        "accion": "intervencion_inmediata"
    }
}
```

#### **Etapa 3: Análisis Dimensional**

```python
dimensional_analysis = {
    "ancho_estimado": "morfología + calibración",
    "profundidad_estimada": "análisis de contornos",
    "area_afectada": "píxeles de fisura / área total",
    "densidad": "número_fisuras / área_imagen"
}
```

#### **Etapa 4: Clasificación Contextual**

```python
contextual_features = {
    "orientacion": ["vertical", "horizontal", "diagonal"],
    "patron": ["lineal", "reticulada", "ramificada"],
    "ubicacion": ["muro", "viga", "columna", "losa"],
    "superficie": ["concreto", "ladrillo", "revoque"]
}
```

---

## **📈 Métricas de Validación Técnica**

### **7.1 Métricas de Detección**

- **Sensibilidad**: Detección de fisuras críticas (>3mm)
- **Especificidad**: Evitar falsos positivos en texturas
- **Precisión dimensional**: Error < 15% vs medición manual

### **7.2 Validación con Expertos**

- **Correlación**: > 85% concordancia con ingenieros civiles
- **Casos críticos**: 100% detección de fisuras estructurales
- **Tiempo de análisis**: < 30 segundos por imagen

---

## **⚠️ Consideraciones de Seguridad**

### **8.1 Protocolos de Alerta**

- **Automática**: Fisuras > 3mm → Alerta inmediata
- **Tendencia**: Crecimiento > 0.1mm/mes → Monitoreo intensivo
- **Ubicación crítica**: Cimientos/columnas → Evaluación profesional

### **8.2 Limitaciones del Sistema**

- No reemplaza inspección profesional en casos críticos
- Requiere calibración para diferentes tipos de construcción
- Precisión limitada por resolución de imagen y condiciones de iluminación

---

_Documento basado en revisión bibliográfica especializada en patologías estructurales y normativas de construcción._
