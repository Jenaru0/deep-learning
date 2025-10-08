# 📋 LOG DÍA 2 - Preprocesamiento SDNET2018

**Fecha:** 7 de octubre, 2025  
**Tarea:** TAREA #2 - División estratificada de SDNET2018  
**Duración:** ~2 horas  
**Estado:** ✅ Completada

---

## 🎯 Objetivo del Día

Dividir el dataset SDNET2018 (56,092 imágenes) en splits train/val/test (70/15/15) con estratificación por categoría y clase, generando una estructura lista para entrenamiento.

---

## ✅ Tareas Completadas

### 1. Creación de Script de Preprocesamiento

- **Archivo:** `scripts/preprocesamiento/dividir_sdnet2018.py`
- **Funcionalidades implementadas:**
  - Recolección automática de imágenes desde SDNET2018
  - División estratificada usando `sklearn.train_test_split`
  - Estratificación por categoría (Deck/Pavement/Wall) y clase (Cracked/Uncracked)
  - Copia organizada a `datos/procesados/deteccion/{train,val,test}/{cracked,uncracked}/`
  - Generación de reporte JSON con estadísticas detalladas
  - Semilla fija (42) para reproducibilidad total

### 2. Ejecución del Preprocesamiento

- **Resultado:** 56,092 imágenes procesadas exitosamente
- **Splits generados:**
  ```
  Train:  39,261 imágenes (70.0%) → 5,937 cracked + 33,324 uncracked
  Val:     8,414 imágenes (15.0%) → 1,273 cracked + 7,141 uncracked
  Test:    8,417 imágenes (15.0%) → 1,274 cracked + 7,143 uncracked
  ```
- **Balance de clases mantenido:** ~15.12% cracked en cada split

### 3. Validación de Integridad

- **Archivo:** `scripts/preprocesamiento/validar_splits.py`
- **Verificaciones realizadas:**
  - ✅ No duplicados entre train/val/test
  - ✅ Ratios exactos: 69.99% / 15.00% / 15.01%
  - ✅ Balance de clases: 15.12% / 15.13% / 15.14%
  - ✅ Total de archivos: 56,092 (100% integridad)

### 4. Documentación Generada

- ✅ `splits_info.json` - Estadísticas completas de división
- ✅ `LOG_DIA_2.md` - Este documento

---

## 📊 Resultados Clave

### Estadísticas de División

| Split     | Total      | Cracked   | Uncracked  | % Cracked  | % Dataset |
| --------- | ---------- | --------- | ---------- | ---------- | --------- |
| Train     | 39,261     | 5,937     | 33,324     | 15.12%     | 70.0%     |
| Val       | 8,414      | 1,273     | 7,141      | 15.13%     | 15.0%     |
| Test      | 8,417      | 1,274     | 7,143      | 15.14%     | 15.0%     |
| **TOTAL** | **56,092** | **8,484** | **47,608** | **15.13%** | **100%**  |

### Distribución por Categoría

**Deck (13,620 imágenes):**

- Train: 9,533 (1,417 cracked + 8,116 uncracked)
- Val: 2,043 (304 cracked + 1,739 uncracked)
- Test: 2,044 (304 cracked + 1,740 uncracked)

**Pavement (24,334 imágenes):**

- Train: 17,033 (1,825 cracked + 15,208 uncracked)
- Val: 3,650 (391 cracked + 3,259 uncracked)
- Test: 3,651 (392 cracked + 3,259 uncracked)

**Wall (18,138 imágenes):**

- Train: 12,695 (2,695 cracked + 10,000 uncracked)
- Val: 2,721 (578 cracked + 2,143 uncracked)
- Test: 2,722 (578 cracked + 2,144 uncracked)

---

## 🔧 Decisiones Técnicas

### 1. Estratificación Multinivel

**Decisión:** Dividir por categoría Y clase simultáneamente  
**Razón:** Mantener distribución representativa de ambas dimensiones  
**Resultado:** Cada split tiene proporciones similares de D/P/W y cracked/uncracked

### 2. Semilla Aleatoria Fija

**Decisión:** RANDOM_SEED = 42 (centralizado en config.py)  
**Razón:** Reproducibilidad total del experimento  
**Impacto:** Cualquier ejecución futura generará exactamente los mismos splits

### 3. Validación Automática

**Decisión:** Crear script de validación independiente  
**Razón:** Detectar errores de división antes de entrenar  
**Beneficio:** Previene data leakage y errores de integridad

### 4. Documentación JSON

**Decisión:** Generar `splits_info.json` con estadísticas  
**Razón:** Trazabilidad para la tesis y reproducibilidad  
**Uso futuro:** Referencia obligatoria en Capítulo IV (Metodología)

---

## 📁 Archivos Generados

```
scripts/preprocesamiento/
├── dividir_sdnet2018.py      (Script principal de división)
└── validar_splits.py          (Script de validación)

datos/procesados/deteccion/
├── train/
│   ├── cracked/     (5,937 imágenes)
│   └── uncracked/   (33,324 imágenes)
├── val/
│   ├── cracked/     (1,273 imágenes)
│   └── uncracked/   (7,141 imágenes)
└── test/
    ├── cracked/     (1,274 imágenes)
    └── uncracked/   (7,143 imágenes)

splits_info.json              (Estadísticas de división)
LOG_DIA_2.md                  (Este documento)
```

---

## 🚀 Próximos Pasos (TAREA #3)

### Objetivo

Preparar dataset CRACK500 para segmentación

### Pasos a seguir

1. Usar splits predefinidos de CRACK500 (train.txt/val.txt/test.txt)
2. Copiar imágenes y máscaras a `datos/procesados/segmentacion/{train,val,test}/`
3. Verificar correspondencia imagen-máscara (1:1)
4. Generar `crack500_info.json` con estadísticas
5. Validar que todas las máscaras tengan anotaciones válidas

### Entregables esperados

- ✅ Script: `scripts/preprocesamiento/preparar_crack500.py`
- ✅ Datos organizados en `datos/procesados/segmentacion/`
- ✅ Archivo: `crack500_info.json`
- ✅ LOG_DIA_3.md

### Tiempo estimado

1.5 horas

---

## ✅ Checklist de Validación TAREA #2

- [x] Script `dividir_sdnet2018.py` creado y funcional
- [x] 56,092 imágenes copiadas exitosamente
- [x] Splits generados con ratios exactos (70/15/15)
- [x] Balance de clases mantenido (~15% cracked en cada split)
- [x] No duplicados entre train/val/test verificado
- [x] Archivo `splits_info.json` generado con estadísticas
- [x] Script de validación `validar_splits.py` creado
- [x] Todas las verificaciones pasadas (✅ PASS)
- [x] Documentación completa en `LOG_DIA_2.md`

---

## 📝 Observaciones

### Éxitos

- ✅ División estratificada perfecta: balance mantenido en todos los splits
- ✅ Proceso completamente automatizado y reproducible
- ✅ Validación exhaustiva confirma 100% integridad
- ✅ Código modular y reutilizable para futuros datasets

### Aprendizajes

- La estratificación multinivel requirió dos llamadas a `train_test_split` secuenciales
- El shuffle final previene agrupaciones por categoría
- La validación automática es esencial antes de entrenar (previene errores costosos)

### Para la Tesis

- **Capítulo IV (Metodología):** Documentar división estratificada con tablas de este LOG
- **Reproducibilidad:** Mencionar semilla fija (42) y disponibilidad de `splits_info.json`
- **Figuras:** Crear gráficos de distribución por categoría y clase para el documento final

---

## ⏱️ Resumen de Tiempo Invertido

| Actividad                        | Tiempo       |
| -------------------------------- | ------------ |
| Creación de script de división   | 30 min       |
| Ejecución y copia de archivos    | 15 min       |
| Creación de script de validación | 20 min       |
| Generación de documentación      | 25 min       |
| Verificación y tests             | 20 min       |
| **TOTAL**                        | **~2 horas** |

---

**Siguiente sesión:** TAREA #3 - Preparación de CRACK500 para segmentación
