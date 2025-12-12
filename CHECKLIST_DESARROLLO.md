# ✅ Checklist de Desarrollo de Versiones

## Estado Actual: v1.0-production ✅ COMPLETADO

---

## 🎯 v1.1-ensemble (Próxima - 30 minutos)

### Pre-requisitos

- [x] Backup de v1.0 creado
- [x] Documentación actualizada
- [ ] Crear rama `feature/ensemble-detection`
- [ ] Confirmar que ambos modelos están disponibles localmente

### Implementación (30 min estimado)

#### 1. Crear estructura de código (10 min)

- [ ] Mover `st.file_uploader` fuera de tabs
- [ ] Implementar `st.session_state` para imagen compartida
- [ ] Crear función `detectar_ensemble(imagen, modelo_det, modelo_seg)`
- [ ] Actualizar lógica de tabs para usar session_state

#### 2. Función de Ensemble (5 min)

```python
- [ ] Implementar predicción con MobileNetV2
- [ ] Implementar predicción con U-Net
- [ ] Lógica OR: Si CUALQUIERA detecta → CON FISURA
- [ ] Retornar ambas predicciones para debug
```

#### 3. UI/UX Updates (10 min)

- [ ] Agregar uploader global antes de tabs
- [ ] Mostrar warning si no hay imagen subida
- [ ] Agregar indicador de qué modelo detectó la fisura
- [ ] Actualizar descripciones de tabs

#### 4. Testing (5 min)

- [ ] Probar con imagen SDNET2018
- [ ] Probar con imagen CRACK500
- [ ] Verificar que ensemble detecta ambas
- [ ] Verificar métricas de performance

### Post-implementación

- [ ] Commit con mensaje descriptivo
- [ ] Crear tag `v1.1-ensemble`
- [ ] Crear rama backup `backup/v1.1-ensemble`
- [ ] Push a GitHub
- [ ] Actualizar VERSIONES.md con resultados
- [ ] Testing en Streamlit Cloud

### Criterios de Aceptación

- [ ] Una imagen puede analizar ambos modelos
- [ ] No más falsos negativos con CRACK500
- [ ] Performance aceptable (<5s para ambos modelos)
- [ ] UI clara sobre qué modelo detectó

---

## 🚀 v2.0-retrained (Futura - 2-3 horas)

### Pre-requisitos

- [ ] Backup de v1.1 creado
- [ ] Confirmar espacio en disco (>5GB)
- [ ] Confirmar GPU disponible
- [ ] Dataset CRACK500 disponible localmente

### Fase 1: Preparación de Datos (30 min)

#### Script: `preparar_crack500_clasificacion.py`

- [ ] Crear estructura train/val/test
- [ ] Copiar imágenes de CRACK500 a carpeta "cracked"
- [ ] Generar imágenes "uncracked" sintéticas (opcional)
- [ ] Balancear distribución si es necesario
- [ ] Verificar splits (70/15/15)

#### Script: `combinar_datasets.py`

- [ ] Cargar estadísticas de SDNET2018
- [ ] Cargar estadísticas de CRACK500
- [ ] Combinar directorios train/val/test
- [ ] Generar nuevo `dataset_combined_summary.json`
- [ ] Verificar integridad de datos

**Resultado esperado:**

```
datos/procesados/deteccion_combinado/
├── train/ (41,157 imgs)
│   ├── cracked/ (11,785)
│   └── uncracked/ (29,372)
├── val/ (8,762 imgs)
└── test/ (9,541 imgs)
```

### Fase 2: Entrenamiento (1.5-2 horas)

#### Script: `entrenar_deteccion_combinado.py`

- [ ] Configurar GPU para máximo rendimiento
- [ ] Cargar MobileNetV2 base (ImageNet)
- [ ] Freeze base + entrenar head (Stage 1: 8 epochs)
- [ ] Fine-tune completo (Stage 2: 22 epochs)
- [ ] Callbacks: ModelCheckpoint, EarlyStopping, ReduceLR
- [ ] Guardar modelo como `modelo_deteccion_v2_combinado.keras`

**Parámetros:**

```python
EPOCHS_STAGE1 = 8
EPOCHS_STAGE2 = 22
BATCH_SIZE = 64
IMG_SIZE = 224
LEARNING_RATE_STAGE1 = 2e-3
LEARNING_RATE_STAGE2 = 1e-4
```

### Fase 3: Validación (30 min)

#### Métricas Objetivo:

- [ ] Accuracy > 95%
- [ ] Recall > 98%
- [ ] F1-Score > 95%
- [ ] Specificity > 70%

#### Testing Específico:

- [ ] Test set SDNET2018 (mantener performance)
- [ ] Test set CRACK500 (mejorar desde 0%)
- [ ] Confusion matrix análisis
- [ ] ROC curve comparison

#### Validación Cruzada:

- [ ] 50 imágenes SDNET2018 aleatorias
- [ ] 50 imágenes CRACK500 aleatorias
- [ ] Comparar v1.0 vs v2.0
- [ ] Documentar mejoras/regresiones

### Fase 4: Integración (30 min)

#### Actualizar app.py:

- [ ] Agregar opción de modelo en sidebar
- [ ] Función `cargar_modelo_v2()`
- [ ] Selector: "v1.0 (SDNET)" | "v2.0 (Combinado)"
- [ ] Mantener compatibilidad con v1.0

#### Deployment:

- [ ] Subir modelo v2 a Google Drive
- [ ] Actualizar download_models.py con nuevo ID
- [ ] Actualizar requirements si es necesario
- [ ] Testing en cloud

### Post-implementación

- [ ] Commit con resultados detallados
- [ ] Crear tag `v2.0-retrained`
- [ ] Crear rama backup `backup/v2.0-retrained`
- [ ] Actualizar VERSIONES.md con métricas
- [ ] Crear release en GitHub con changelog
- [ ] Documentar comparación v1.0 vs v2.0

### Criterios de Aceptación

- [ ] Detecta fisuras de SDNET2018 (mantiene performance)
- [ ] Detecta fisuras de CRACK500 (mejora desde ~0% a >90%)
- [ ] Accuracy >= 95%
- [ ] Tiempo de inferencia <2s
- [ ] Compatible con deployment cloud
- [ ] Documentación completa de mejoras

---

## 📊 Comparación de Versiones (Actualizar después de cada versión)

| Métrica                | v1.0        | v1.1 Ensemble | v2.0 Retrained |
| ---------------------- | ----------- | ------------- | -------------- |
| **Accuracy**           | 94.36%      | TBD           | TBD            |
| **SDNET Detection**    | ✅ 99.64%   | TBD           | TBD            |
| **CRACK500 Detection** | ❌ ~30%     | TBD           | TBD            |
| **Inference Time**     | ~1s         | TBD           | TBD            |
| **Modelos Usados**     | 2 separados | 2 ensemble    | 1 combinado    |
| **Complejidad**        | Baja        | Media         | Baja           |
| **Mantenibilidad**     | Alta        | Media         | Alta           |

---

## 🔄 Rollback Plan

### Si v1.1 falla:

```bash
git checkout backup/v1.0-production
git checkout -b hotfix/v1.1-fix
# Fix issues
git checkout main
git merge hotfix/v1.1-fix
```

### Si v2.0 falla:

```bash
git checkout backup/v1.1-ensemble  # o v1.0 si v1.1 no existe
# Continue usando versión estable
```

---

## 📝 Notas de Desarrollo

### v1.1-ensemble

**Fecha inicio:** [TBD]  
**Fecha fin:** [TBD]  
**Problemas encontrados:** [TBD]  
**Soluciones aplicadas:** [TBD]  
**Lecciones aprendidas:** [TBD]

### v2.0-retrained

**Fecha inicio:** [TBD]  
**Fecha fin:** [TBD]  
**Tiempo total entrenamiento:** [TBD]  
**Mejoras vs v1.0:** [TBD]  
**Problemas encontrados:** [TBD]  
**Lecciones aprendidas:** [TBD]

---

**Última actualización:** 12 Diciembre 2025  
**Próximo hito:** v1.1-ensemble
