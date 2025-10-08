# ⚠️ ERRORES COMUNES A EVITAR - Guía de Prevención

---

## 🚫 CATEGORÍA 1: GESTIÓN DE DATOS

### ❌ Error #1: Modificar datasets originales

**Problema:** Copiar, mover o editar archivos en `datasets/SDNET2018` o `datasets/CRACK500`  
**Impacto:** Pérdida de datos originales, imposibilidad de replicar experimentos  
**Solución:** SIEMPRE trabajar en `datos/procesados/`. Los originales son SAGRADOS.

### ❌ Error #2: No validar integridad de datos

**Problema:** Asumir que todos los archivos están completos sin verificar  
**Impacto:** Errores silenciosos durante entrenamiento, resultados inconsistentes  
**Solución:** Verificar conteos, formatos y que no haya archivos corruptos ANTES de entrenar

### ❌ Error #3: Mezclar splits train/val/test

**Problema:** Usar mismas imágenes en entrenamiento y evaluación  
**Impacto:** Data leakage = métricas infladas = resultados inválidos  
**Solución:** Dividir UNA VEZ con semilla fija, documentar splits, nunca re-mezclar

### ❌ Error #4: No balancear clases

**Problema:** Entrenar con 85% negativos / 15% positivos sin ajustes  
**Impacto:** Modelo aprende a predecir "sin fisura" siempre = 85% accuracy inútil  
**Solución:** Usar `class_weight`, data augmentation en minoritaria, o undersampling controlado

---

## 🚫 CATEGORÍA 2: CONFIGURACIÓN Y CÓDIGO

### ❌ Error #5: Rutas relativas inconsistentes

**Problema:** Usar `../datos/train` en un script, `./datos/train` en otro  
**Impacto:** Scripts fallan según desde dónde se ejecuten  
**Solución:** TODAS las rutas absolutas definidas en `config.py`, importar ese módulo siempre

### ❌ Error #6: No fijar semilla aleatoria

**Problema:** Cada ejecución da resultados distintos  
**Impacto:** Imposible reproducir experimentos, comparaciones inválidas  
**Solución:** `RANDOM_SEED = 42` en config.py, aplicar en NumPy, TensorFlow, Python random

### ❌ Error #7: Hardcodear valores

**Problema:** Escribir `epochs=50` en 10 lugares distintos del código  
**Impacto:** Cambiar un parámetro requiere editar múltiples archivos = errores  
**Solución:** Centralizar TODOS los parámetros en `config.py`

### ❌ Error #8: No versionar modelos

**Problema:** Sobrescribir `modelo.h5` cada vez que entrenas  
**Impacto:** Pierdes versiones anteriores, no puedes comparar  
**Solución:** Guardar con nombres descriptivos: `modelo_deteccion_v1_acc_0.92.h5`

---

## 🚫 CATEGORÍA 3: ENTRENAMIENTO

### ❌ Error #9: No usar callbacks

**Problema:** Dejar entrenar 50 epochs sin early stopping ni guardado de mejor modelo  
**Impacto:** Overfitting, pérdida de mejor versión, tiempo desperdiciado  
**Solución:** Usar `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`

### ❌ Error #10: Entrenar desde cero modelos grandes

**Problema:** Inicializar ResNet50 aleatoriamente en lugar de usar pesos ImageNet  
**Impacto:** 10x más epochs necesarios, peores resultados  
**Solución:** SIEMPRE usar `weights='imagenet'` en transfer learning

### ❌ Error #11: No validar durante entrenamiento

**Problema:** Solo mirar loss de entrenamiento, ignorar validación  
**Impacto:** No detectas overfitting hasta el final  
**Solución:** Monitorear `val_loss` y `val_accuracy` cada epoch

### ❌ Error #12: Batch size muy grande o muy pequeño

**Problema:** Usar batch=256 en GPU de 4GB, o batch=2 en GPU de 24GB  
**Impacto:** OOM error o desperdicio de recursos  
**Solución:** Empezar con batch=32, ajustar según memoria disponible

---

## 🚫 CATEGORÍA 4: EVALUACIÓN

### ❌ Error #13: Evaluar solo en accuracy

**Problema:** Reportar "95% accuracy" sin precision/recall en dataset desbalanceado  
**Impacto:** Métrica engañosa, modelo inútil para clase minoritaria  
**Solución:** Reportar precision, recall, F1, confusion matrix, ROC-AUC

### ❌ Error #14: No hacer análisis cualitativo

**Problema:** Solo mostrar números, sin casos visuales de aciertos/fallos  
**Impacto:** Tesis sin profundidad, no entiendes errores del modelo  
**Solución:** Documentar 10+ casos críticos con imágenes y explicaciones

### ❌ Error #15: Comparar sin estandarizar test set

**Problema:** Comparar tu modelo en CRACK500 vs baseline en otro dataset  
**Impacto:** Comparación inválida, resultados no comparables  
**Solución:** Entrenar TODOS los modelos en mismo train, evaluar en MISMO test

### ❌ Error #16: No validar con experto humano

**Problema:** Asumir que IoU=0.85 significa "sistema útil"  
**Impacto:** Falta evidencia de aplicabilidad práctica  
**Solución:** Validar 50 casos con ingeniero civil, calcular Cohen's Kappa

---

## 🚫 CATEGORÍA 5: DOCUMENTACIÓN

### ❌ Error #17: No llevar logs de experimentos

**Problema:** Entrenar 20 modelos, perder track de cuál fue el mejor y por qué  
**Impacto:** No puedes justificar decisiones en la tesis  
**Solución:** Mantener `LOG_DIA_X.md` diarios + tabla de experimentos

### ❌ Error #18: No documentar hiperparámetros

**Problema:** Reportar "usé Adam optimizer" sin especificar learning rate  
**Impacto:** Experimento no reproducible  
**Solución:** Documentar TODOS los hiperparámetros en tabla de la tesis

### ❌ Error #19: No explicar decisiones técnicas

**Problema:** Decir "usé EfficientNet" sin justificar por qué  
**Impacto:** Jurado pregunta, no sabes responder  
**Solución:** Para cada decisión, tener 2-3 razones técnicas documentadas

### ❌ Error #20: No mantener código limpio

**Problema:** 10 archivos `test.py`, `test2.py`, `test_final.py`, `test_final_v2.py`  
**Impacto:** Código incomprensible, no reproducible  
**Solución:** Nombres descriptivos, estructura clara, comentarios en código

---

## 🚫 CATEGORÍA 6: PRESENTACIÓN

### ❌ Error #21: Slides con mucho texto

**Problema:** Poner párrafos completos en presentación de defensa  
**Impacto:** Audiencia no lee, tú lees slides = mala presentación  
**Solución:** Máximo 5 bullets por slide, usar visuales

### ❌ Error #22: Demo sin preparar

**Problema:** Mostrar demo en vivo sin haberla probado 5+ veces  
**Impacto:** Falla durante defensa = pérdida de credibilidad  
**Solución:** 10 ensayos de demo, tener video backup

### ❌ Error #23: No anticipar preguntas

**Problema:** Defender sin preparar respuestas a preguntas obvias  
**Impacto:** Nervios, respuestas vagas  
**Solución:** Lista de 20 preguntas probables con respuestas preparadas

---

## ✅ CHECKLIST DE PREVENCIÓN

**Antes de cada sesión:**

- [ ] Verificar que `config.py` está actualizado
- [ ] Confirmar que rutas en config existen
- [ ] Comprobar semilla aleatoria está fija
- [ ] Revisar que test set no se contamina

**Durante entrenamiento:**

- [ ] Monitorear val_loss cada epoch
- [ ] Guardar checkpoints periódicamente
- [ ] Anotar hiperparámetros en log
- [ ] Verificar uso de GPU (nvidia-smi)

**Después de experimento:**

- [ ] Documentar resultados en LOG
- [ ] Guardar modelo con nombre descriptivo
- [ ] Generar gráficas de loss/accuracy
- [ ] Anotar observaciones y siguientes pasos

**Antes de defensa:**

- [ ] Ensayar presentación 5+ veces
- [ ] Probar demo 10+ veces
- [ ] Preparar respuestas a 20 preguntas
- [ ] Tener backup de todo en USB

---

## 🎯 REGLA DE ORO

> **"Si no está documentado, no existe."**

- Resultados sin logs = no replicables
- Código sin comentarios = incomprensible en 1 semana
- Decisiones sin justificación = vulnerables en defensa

**Invierte 10 minutos en documentar, ahorra 10 horas en confusión.**

---

_Mantén este archivo abierto durante todo el proyecto. Revísalo antes de cada sesión._

**Actualizado:** 7 de octubre, 2025
