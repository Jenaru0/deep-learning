# 📦 Control de Versiones del Sistema

## Sistema de Detección de Fisuras - Roadmap de Versiones

Este documento mantiene el registro de todas las versiones del sistema, sus características, y cómo restaurarlas.

---

## 🎯 Versiones Disponibles

### **v1.0-production** ✅ ESTABLE (Actual en Producción)
**Tag Git:** `v1.0-production`  
**Rama Backup:** `backup/v1.0-production`  
**Fecha:** 12 Diciembre 2025  
**Estado:** 🟢 Estable y funcional

#### Características:
- ✅ Modelo detección: MobileNetV2 (SDNET2018, 56,092 imgs)
- ✅ Modelo segmentación: U-Net Lite (CRACK500, 3,368 imgs)
- ✅ UI/UX moderno con gradientes y colores vibrantes
- ✅ Deployment en Streamlit Cloud
- ✅ Python 3.10 + TensorFlow 2.15.0

#### Métricas:
| Modelo | Accuracy | Recall | F1-Score | IoU | Dice |
|--------|----------|--------|----------|-----|------|
| Detección | 94.36% | 99.64% | 96.77% | - | - |
| Segmentación | 97.4% | - | - | 60.5% | 73.0% |

#### Limitaciones Conocidas:
⚠️ **Detección:** Entrenada solo con SDNET2018 (fisuras gruesas en edificios)
- Puede fallar con fisuras finas tipo CRACK500
- Mejor para fisuras evidentes en concreto

⚠️ **Segmentación:** Entrenada solo con CRACK500 (fisuras finas en pavimento)
- Optimizada para fisuras sutiles
- Puede sobre-segmentar en fisuras muy gruesas

#### Restaurar esta versión:
```bash
# Opción 1: Con tag
git checkout v1.0-production

# Opción 2: Con rama backup
git checkout backup/v1.0-production

# Opción 3: Crear nueva rama desde backup
git checkout -b hotfix/v1.0 backup/v1.0-production
```

---

### **v1.1-ensemble** 🚧 EN DESARROLLO (Próxima)
**Tag Git:** `v1.1-ensemble` (pendiente)  
**Rama:** `feature/ensemble-detection`  
**Fecha Estimada:** 12 Diciembre 2025  
**Estado:** 🟡 En desarrollo

#### Características Planificadas:
- ✅ Mantiene modelos actuales (sin re-entrenar)
- ✅ Ensemble de ambos modelos para detección
- ✅ Uploader global único (mejor UX)
- ✅ Tabs dinámicos con session_state
- ✅ Detección combinada: OR logic entre modelos
- ✅ Mejora inmediata sin tiempo de entrenamiento

#### Mejoras Esperadas:
- 🎯 **Cobertura completa:** SDNET2018 + CRACK500
- 🎯 **Menos falsos negativos:** Ensemble OR logic
- 🎯 **Mejor UX:** 1 imagen → múltiples análisis
- 🎯 **Sin degradación:** Mantiene métricas actuales

#### Implementación:
```python
def detectar_ensemble(imagen, modelo_det, modelo_seg):
    """
    Combina MobileNetV2 y U-Net para detección robusta.
    Lógica: Si CUALQUIERA detecta fisura → Clasificar como CON FISURA
    """
    # Detección rápida (SDNET2018)
    pred_det = modelo_det.predict(resize(imagen, 224))
    
    # Segmentación (CRACK500)
    mascara = modelo_seg.predict(resize(imagen, 128))
    area_fisura = np.sum(mascara > 0.5)
    
    # Ensemble: OR logic
    tiene_fisura = (pred_det > 0.5) or (area_fisura > 100)
    
    return tiene_fisura, pred_det, area_fisura, mascara
```

#### Restaurar cuando esté lista:
```bash
git checkout v1.1-ensemble
# o
git checkout feature/ensemble-detection
```

---

### **v2.0-retrained** 🔮 PLANIFICADA (Futura)
**Tag Git:** `v2.0-retrained` (pendiente)  
**Rama:** `feature/combined-training`  
**Fecha Estimada:** Semana del 16-20 Diciembre 2025  
**Estado:** 🔴 Planificada

#### Características Planificadas:
- 🔄 **RE-ENTRENAMIENTO COMPLETO** de modelo detección
- ✅ Dataset combinado: SDNET2018 (56K) + CRACK500 (3.3K) = 59.4K imgs
- ✅ Modelo único generalizado para ambos tipos de fisuras
- ✅ Arquitectura: MobileNetV2 (mantiene velocidad)
- ✅ Optimizado para RTX 2050
- ✅ Solución científicamente correcta

#### Mejoras Esperadas:
- 🎯 **Generalización:** Aprende de 2 distribuciones distintas
- 🎯 **Sin ensemble:** 1 modelo robusto
- 🎯 **Mejor precisión:** Esperada 95%+
- 🎯 **Más rápido:** Sin necesidad de ensemble

#### Dataset Combinado:
```
SDNET2018 (56,092 imgs):
├─ Cracked: 8,417 (15%)
└─ Uncracked: 47,675 (85%)

CRACK500 (3,368 imgs):
└─ Cracked: 3,368 (100%) ← Balancear distribución

COMBINADO (59,460 imgs):
├─ Cracked: 11,785 (19.8%)
└─ Uncracked: 47,675 (80.2%)
```

#### Pipeline de Entrenamiento:
```bash
# 1. Preparar CRACK500 para clasificación
python scripts/preprocesamiento/preparar_crack500_clasificacion.py

# 2. Combinar datasets
python scripts/preprocesamiento/combinar_datasets.py

# 3. Re-entrenar con datos combinados
python scripts/entrenamiento/entrenar_deteccion_combinado.py

# Resultado: modelo_deteccion_v2.keras
```

#### Tiempo Estimado:
- Preprocesamiento: ~30 min
- Entrenamiento: ~1.5-2 horas (RTX 2050)
- Validación: ~30 min
- **Total:** ~2.5-3 horas

---

## 🔄 Estrategia de Migración

### Fase 1: Backup (✅ COMPLETADO)
```bash
✅ Tag creado: v1.0-production
✅ Rama backup: backup/v1.0-production
✅ Subido a GitHub
```

### Fase 2: Desarrollo Ensemble (🚧 SIGUIENTE)
```bash
1. Crear rama feature/ensemble-detection
2. Implementar uploader global + session_state
3. Implementar función detectar_ensemble()
4. Testing con imágenes SDNET + CRACK500
5. Commit y tag v1.1-ensemble
6. Backup rama backup/v1.1-ensemble
```

### Fase 3: Re-entrenamiento (🔮 FUTURA)
```bash
1. Crear rama feature/combined-training
2. Preparar scripts de preprocesamiento
3. Entrenar modelo combinado
4. Validar métricas (target: 95%+)
5. Integrar en app.py
6. Testing exhaustivo
7. Commit y tag v2.0-retrained
8. Backup rama backup/v2.0-retrained
```

---

## 📊 Comparación de Versiones

| Característica | v1.0 | v1.1 Ensemble | v2.0 Retrained |
|----------------|------|---------------|----------------|
| **Modelos** | 2 separados | 2 combinados | 1 generalizado |
| **Cobertura** | SDNET ó CRACK | SDNET + CRACK | SDNET + CRACK |
| **Velocidad** | Rápida | Media | Rápida |
| **Precisión** | 94.36% | ~94% | ~95%+ |
| **Falsos Neg.** | Altos (CRACK) | Bajos | Muy bajos |
| **Tiempo Dev** | - | 30 min | 2-3 horas |
| **Re-entrenar** | No | No | Sí |
| **Producción** | ✅ Estable | 🟡 Testing | 🔴 Desarrollo |

---

## 🚨 Recuperación de Emergencia

### Si algo falla en v1.1 o v2.0:

```bash
# 1. Restaurar a v1.0-production
git checkout backup/v1.0-production

# 2. Crear hotfix si es necesario
git checkout -b hotfix/v1.0.1 backup/v1.0-production

# 3. Re-desplegar en Streamlit Cloud
# (automático al hacer push a main)

# 4. Investigar el problema en rama separada
git checkout -b debug/issue-description
```

---

## 📝 Logs de Cambios

### v1.0-production (12 Dic 2025)
- ✅ UI moderna con gradientes
- ✅ Tabs para Detección/Segmentación/Ayuda
- ✅ Progress bars y feedback visual
- ✅ Cards informativos con colores
- ✅ FAQ y documentación integrada
- ✅ Deployment en Streamlit Cloud funcional

---

## 🔗 Enlaces Útiles

- **Repositorio:** https://github.com/Jenaru0/deep-learning
- **Streamlit Cloud:** (URL de deployment)
- **Issues:** https://github.com/Jenaru0/deep-learning/issues
- **Releases:** https://github.com/Jenaru0/deep-learning/releases

---

**Última actualización:** 12 Diciembre 2025  
**Mantenedor:** Jesus Naranjo
