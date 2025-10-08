# 📝 LOG DE TRABAJO - DÍA 1

---

**Proyecto:** Sistema de Detección y Segmentación de Fisuras Estructurales  
**Fecha:** 7 de octubre, 2025  
**Investigador:** [Tu nombre]  
**Tiempo invertido:** 2.5 horas  
**Estado:** ✅ COMPLETADO

---

## ✅ TAREAS COMPLETADAS

### 1. Estructura de Carpetas Organizada (45 min)

Creada la siguiente estructura jerárquica:

```
investigacion_fisuras/
├── datos/
│   └── procesados/
│       ├── deteccion/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── segmentacion/
│           ├── train/
│           ├── val/
│           └── test/
├── modelos/
│   ├── deteccion/
│   └── segmentacion/
├── resultados/
│   ├── deteccion/
│   ├── segmentacion/
│   └── visualizaciones/
├── notebooks/
└── scripts/
    ├── preprocesamiento/
    ├── entrenamiento/
    └── evaluacion/
```

**Decisión tomada:** Separar claramente `deteccion` y `segmentacion` para mantener modularidad y facilitar experimentación independiente.

---

### 2. Inventario de Datasets Documentado (40 min)

#### SDNET2018 - Verificación completa:

- ✅ Deck: 2,025 con fisura + 11,595 sin fisura = 13,620 total
- ✅ Pavement: 2,608 con fisura + 21,726 sin fisura = 24,334 total
- ✅ Wall: 3,851 con fisura + 14,287 sin fisura = 18,138 total
- **TOTAL VERIFICADO: 56,092 imágenes**
- **Balance de clases: 15.12% positivas (CON FISURA)**

#### CRACK500 - Verificación completa:

- ✅ Train: 1,896 pares imagen-máscara
- ✅ Validation: 348 pares
- ✅ Test: 1,124 pares
- **TOTAL VERIFICADO: 3,368 imágenes**
- División predefinida en archivos .txt confirmada

**Archivos generados:**

- `inventario_SDNET2018.csv`
- `inventario_CRACK500.csv`

---

### 3. Verificación de Integridad (30 min)

Ejecutados comandos de conteo en PowerShell:

```powershell
# SDNET2018 - Resultados
CD: 2,025 ✅
UD: 11,595 ✅
CP: 2,608 ✅
UP: 21,726 ✅
CW: 3,851 ✅
UW: 14,287 ✅

# CRACK500 - Resultados
Train.txt: 1,896 líneas ✅
Val.txt: 348 líneas ✅
Test.txt: 1,124 líneas ✅
```

**Resultado:** ✅ **TODOS LOS ARCHIVOS ÍNTEGROS** - No se encontraron discrepancias

---

### 4. Archivo config.py Creado (45 min)

Configuración central con:

- ✅ Rutas absolutas validadas
- ✅ Parámetros de entrenamiento definidos
- ✅ Inventario embebido en código
- ✅ Semilla aleatoria fija (RANDOM_SEED = 42)
- ✅ Configuración de data augmentation
- ✅ Umbrales de severidad definidos
- ✅ Función de validación de rutas

**Decisión crítica:** Usar `RANDOM_SEED = 42` en TODOS los experimentos para garantizar reproducibilidad total.

---

## 📊 OBSERVACIONES Y HALLAZGOS

### Desafíos Identificados:

1. **Desbalance severo en SDNET2018:**

   - Solo 15.12% de imágenes con fisuras
   - Requiere estrategia de balanceo (class weights o undersampling controlado)
   - Solución propuesta: Usar `class_weight` en entrenamiento + data augmentation agresivo en clase minoritaria

2. **Formato de máscaras en CRACK500:**

   - Archivo `20160222_081031_mask.txt` contiene solo valores 0 (sin fisura visible)
   - Sugiere que no todas las imágenes tienen fisuras anotadas
   - Acción: Verificar mañana cuántas máscaras tienen píxeles >0

3. **Diferencia de volumen entre datasets:**
   - SDNET2018: 56K imágenes (detección)
   - CRACK500: 3.4K imágenes (segmentación)
   - Estrategia: Usar SDNET para pre-entrenar backbone, fine-tune en CRACK500

---

## 🎯 DECISIONES TÉCNICAS TOMADAS

| Aspecto               | Decisión      | Justificación                         |
| --------------------- | ------------- | ------------------------------------- |
| **Tamaño imagen**     | 224×224       | Estándar de EfficientNetB0            |
| **Batch size**        | 32            | Balance entre velocidad y memoria GPU |
| **División SDNET**    | 70/15/15      | Estándar académico                    |
| **División CRACK500** | Predefinida   | Usar splits oficiales del dataset     |
| **Semilla aleatoria** | 42            | Reproducibilidad total                |
| **Formato salida**    | CSV + logs MD | Fácil integración en tesis            |

---

## ⚠️ ERRORES EVITADOS

Durante la ejecución, tuve cuidado de:

✅ **NO modificar datasets originales** - Todo trabajo en carpeta `procesados/`  
✅ **NO usar rutas relativas** - Todas absolutas en `config.py`  
✅ **NO empezar a entrenar** - Primero organizar, luego ejecutar  
✅ **NO mezclar configuraciones** - Un solo archivo central  
✅ **NO omitir verificación** - Confirmé cada número del inventario

---

## 📋 CHECKLIST FINAL

- [x] Estructura de carpetas creada y organizada
- [x] Inventario de SDNET2018 documentado (CSV)
- [x] Inventario de CRACK500 documentado (CSV)
- [x] Verificación de integridad ejecutada (100% OK)
- [x] Archivo config.py creado con rutas absolutas
- [x] Parámetros de entrenamiento definidos
- [x] Semilla aleatoria configurada (42)
- [x] Log de trabajo documentado (este archivo)

---

## 🚀 PRÓXIMO PASO - DÍA 2

**TAREA #2: Preprocesar SDNET2018 y crear división estratificada train/val/test**

**Objetivos:**

1. Crear script `scripts/preprocesamiento/dividir_sdnet2018.py`
2. Dividir 56K imágenes en 70% train / 15% val / 15% test
3. Mantener balance de clases en cada split (estratificación)
4. Copiar archivos a `datos/procesados/deteccion/`
5. Generar archivo `splits_info.json` con estadísticas

**Tiempo estimado:** 3-4 horas

**Entregable esperado:**

- Script funcional de división
- Archivos copiados en estructura correcta
- Reporte de distribución de clases por split
- Validación de que no hay data leakage

---

## 💡 REFLEXIÓN DEL DÍA

> "La organización inicial es el 50% del éxito. Hoy invertí 2.5 horas que me ahorrarán 10+ horas en búsquedas de archivos y correcciones de rutas. Bases sólidas = proyecto exitoso."

**Lección aprendida:** Nunca subestimar la importancia de la estructura. Un proyecto ML bien organizado desde el día 1 es 10x más fácil de mantener y documentar.

---

**Firma:** [Tu nombre]  
**Próxima sesión:** 8 de octubre, 2025  
**Estado emocional:** 😊 Motivado - Buena base establecida

---

_Este log será parte de la documentación anexa de la tesis._
