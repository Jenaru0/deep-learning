# 🚀 GUÍA DE DEPLOYMENT A STREAMLIT CLOUD

**Fecha**: 12 de Diciembre de 2025  
**Versión**: 2.0 (Optimizada)

---

## 📋 PREREQUISITOS

Antes de empezar, verifica que tienes:

- ✅ Cuenta de GitHub (gratis)
- ✅ Cuenta de Streamlit Cloud (gratis, vinculada a GitHub)
- ✅ Modelos subidos a Google Drive (ya hecho)
- ✅ URLs de Google Drive configuradas en `download_models.py`

---

## 🎯 ESTRATEGIA DE DEPLOYMENT

### Arquitectura:
```
GitHub (código) 
    ↓
Streamlit Cloud (hosting)
    ↓ (descarga automática en primera ejecución)
Google Drive (modelos: 22MB total)
```

### Por qué esta estrategia:
- ✅ **GitHub**: Gratuito, versionado, integración directa con Streamlit
- ✅ **Streamlit Cloud**: 1GB RAM gratis, ideal para modelos pequeños
- ✅ **Google Drive**: Modelos grandes (no van a GitHub por .gitignore)

---

## 📝 PASO 1: PREPARAR REPOSITORIO GITHUB

### 1.1 Inicializar Git (si no está inicializado)

```powershell
cd 'c:\Users\jonna\OneDrive\Escritorio\DEEP LEARNING\investigacion_fisuras'

# Verificar si ya existe repo
git status

# Si no existe, inicializar
git init
git branch -M main
```

### 1.2 Verificar .gitignore

El archivo `.gitignore` ya está configurado para **EXCLUIR**:
- ❌ `datasets/` (7.46 GB - muy grande)
- ❌ `venv/` (5.38 GB - no necesario en cloud)
- ❌ `datos/procesados/` (datos intermedios)
- ❌ `__pycache__/` (archivos Python compilados)
- ❌ `*.pyc`, `*.pyo` (bytecode)

Y **SÍ incluir**:
- ✅ `app_web/` (código de la aplicación)
- ✅ `scripts/` (módulos de análisis)
- ✅ `config.py` (configuración)
- ✅ `requirements_streamlit.txt` (dependencias)
- ✅ `.streamlit/config.toml` (configuración Streamlit)
- ✅ `docs/` (documentación)

**⚠️ IMPORTANTE**: Los modelos `.keras` en `modelos/` están **EXCLUIDOS** del repo (se descargan de Google Drive).

### 1.3 Verificar archivos a subir

```powershell
# Ver qué archivos se van a subir
git add .
git status

# Deberías ver ~50-100 archivos (SIN datasets ni venv)
```

**Tamaño esperado del repo**: ~5-10 MB (código + docs)

---

## 📝 PASO 2: CREAR REPOSITORIO EN GITHUB

### 2.1 Crear repositorio vacío

1. Ve a: https://github.com/new
2. **Repository name**: `deteccion-fisuras-deep-learning`
3. **Description**: `Sistema de detección y segmentación de fisuras en estructuras de concreto usando MobileNetV2 y U-Net`
4. **Visibility**: 
   - ✅ **Public** (recomendado para Streamlit Cloud gratis)
   - ⚠️ Private (requiere plan Streamlit Cloud de pago)
5. **NO marcar**: "Initialize with README" (ya lo tienes localmente)
6. Click: **Create repository**

### 2.2 Conectar repo local con GitHub

```powershell
# Agregar remote (reemplaza TU_USUARIO con tu username de GitHub)
git remote add origin https://github.com/TU_USUARIO/deteccion-fisuras-deep-learning.git

# Verificar remote
git remote -v
```

### 2.3 Commit inicial

```powershell
# Agregar todos los archivos
git add .

# Crear commit inicial
git commit -m "feat: Sistema completo de detección de fisuras v2.0

- Detección binaria con MobileNetV2 (99.64% recall)
- Segmentación con U-Net Lite (60.51% IoU, 73.04% Dice)
- Medición de parámetros estructurales (ancho, orientación, profundidad)
- Optimizaciones de rendimiento (caché, lazy loading)
- Deployment ready para Streamlit Cloud"

# Subir a GitHub
git push -u origin main
```

**Verifica en GitHub**: Deberías ver todos los archivos (excepto datasets, venv, modelos)

---

## 📝 PASO 3: CONFIGURAR STREAMLIT CLOUD

### 3.1 Crear cuenta (si no tienes)

1. Ve a: https://share.streamlit.io/
2. Click: **Sign up** (usa tu cuenta de GitHub)
3. Autoriza acceso a Streamlit

### 3.2 Crear nueva app

1. Click: **New app** (botón azul superior derecho)
2. Selecciona:
   - **Repository**: `TU_USUARIO/deteccion-fisuras-deep-learning`
   - **Branch**: `main`
   - **Main file path**: `app_web/app.py`
   - **App URL**: `deteccion-fisuras` (o el que prefieras)
   
3. **Advanced settings** (opcional):
   - **Python version**: 3.10 (o 3.11)
   - **Secrets**: Vacío por ahora (no necesitamos)

4. Click: **Deploy!**

### 3.3 Proceso de deployment

Streamlit Cloud hará automáticamente:

1. ✅ Clonar repositorio desde GitHub
2. ✅ Instalar dependencias de `requirements_streamlit.txt`
3. ✅ Ejecutar `app_web/app.py`
4. ✅ **Descargar modelos desde Google Drive** (primera vez, ~30-60s)
5. ✅ Iniciar servidor en URL pública

**Tiempo estimado**: 5-10 minutos

---

## 📝 PASO 4: VERIFICAR DEPLOYMENT

### 4.1 Logs de deployment

Durante el deployment, verás logs en tiempo real:

```
[15:23:01] 📦 Installing packages...
[15:23:45] ✅ numpy==1.23.5
[15:23:50] ✅ tensorflow==2.10.0
[15:24:30] ✅ streamlit==1.32.0
[15:24:35] 🚀 Starting app...
[15:24:40] 📥 Downloading models from Google Drive...
[15:25:10] ✅ Model downloaded: modelo_deteccion_final.keras
[15:25:20] ✅ Model downloaded: unet_segmentacion_final.keras
[15:25:25] 🌐 App is live at https://deteccion-fisuras.streamlit.app
```

### 4.2 Primera carga (usuario final)

**Primera vez** (sin caché):
- Carga de modelos: 30-50s (normal, solo primera vez)
- TensorFlow en CPU: Más lento que local (sin GPU)

**Siguientes veces** (con caché):
- Modelos ya descargados: Instantáneo
- Inferencia: 3-5s (CPU es más lento que GPU local)

### 4.3 Checklist de verificación

- [ ] ✅ App carga sin errores
- [ ] ✅ Modos "Detección" y "Segmentación" visibles
- [ ] ✅ Subir imagen funciona
- [ ] ✅ Detección clasifica correctamente
- [ ] ✅ Segmentación genera máscaras
- [ ] ✅ Parámetros estructurales se calculan
- [ ] ✅ Checkbox de lazy loading funciona
- [ ] ✅ No hay errores en logs

---

## ⚠️ POSIBLES PROBLEMAS Y SOLUCIONES

### Problema 1: Error al descargar modelos de Google Drive

**Síntoma**: 
```
FileNotFoundError: [Errno 2] No such file or directory: 'modelos/deteccion/modelo_deteccion_final.keras'
```

**Causa**: Enlaces de Google Drive incorrectos o archivos no públicos

**Solución**:
```powershell
# Verificar que los archivos están públicos en Google Drive
# Compartir → Cualquier persona con el enlace → Visualizador

# Verificar IDs en download_models.py
# Líneas 30-31:
# FILE_IDS = {
#     'detection': '1toZrp6q8-qCrRk7DUkz12wH9DAMVv0_V',
#     'segmentation': '1Ug_h1flAfLNHLMyNTP2yb_4wdqE5HxIt'
# }
```

### Problema 2: Out of Memory (OOM)

**Síntoma**: 
```
MemoryError: Unable to allocate array
```

**Causa**: Streamlit Cloud gratis tiene 1GB RAM, modelos grandes pueden exceder

**Solución** (si ocurre):
1. Reducir batch_size en inferencia (ya configurado en 1)
2. Procesar imágenes en resolución menor
3. Considerar plan Streamlit Cloud de pago (más RAM)

### Problema 3: Instalación lenta de TensorFlow

**Síntoma**: Deployment tarda >10 minutos en instalar TensorFlow

**Causa**: TensorFlow es pesado (~500MB)

**Solución**: 
- ✅ Ya configurado: `tensorflow==2.10.0` (versión ligera)
- ⏳ Esperar pacientemente (solo primera vez)

### Problema 4: App se reinicia constantemente

**Síntoma**: App se recarga cada 5-10 segundos

**Causa**: Error en código que genera excepción

**Solución**:
```powershell
# Ver logs en Streamlit Cloud (esquina inferior derecha)
# Buscar traceback de Python
# Corregir error localmente
# Hacer commit + push para actualizar
```

---

## 🔄 ACTUALIZACIONES FUTURAS

### Flujo de actualización:

```powershell
# 1. Hacer cambios localmente
# 2. Probar localmente
streamlit run app_web/app.py

# 3. Commit y push
git add .
git commit -m "feat: descripción del cambio"
git push

# 4. Streamlit Cloud detecta cambio automáticamente
# 5. Re-deployment automático (~2-3 minutos)
```

**Auto-deployment**: Cada push a `main` actualiza la app automáticamente

---

## 📊 MONITOREO Y MÉTRICAS

### Streamlit Cloud Dashboard

Ver en: https://share.streamlit.io/

**Métricas disponibles**:
- 📈 **Viewers**: Número de usuarios activos
- ⏱️ **Uptime**: Tiempo que la app ha estado activa
- 💾 **Resource usage**: CPU, RAM, almacenamiento
- 🔄 **Deployments**: Historial de deployments

### Límites del plan gratuito:
- **RAM**: 1 GB
- **CPU**: Compartido
- **Storage**: 1 GB
- **Viewers concurrentes**: Ilimitado
- **Apps**: 1 app pública gratis

---

## 🎯 MEJORES PRÁCTICAS

### 1. Versionado semántico

```powershell
# Usa tags para versiones importantes
git tag -a v2.0.0 -m "Release: Versión optimizada con caché"
git push --tags
```

### 2. Branches para features

```powershell
# Desarrollo en branch separado
git checkout -b feature/nueva-funcionalidad

# Merge a main cuando esté listo
git checkout main
git merge feature/nueva-funcionalidad
git push
```

### 3. Secrets en Streamlit Cloud

Si necesitas API keys o passwords:

1. Streamlit Cloud → App settings → Secrets
2. Agregar en formato TOML:
```toml
# .streamlit/secrets.toml (solo en cloud, NO subir a GitHub)
[google_drive]
api_key = "tu_api_key_aqui"
```

3. Acceder en código:
```python
import streamlit as st
api_key = st.secrets["google_drive"]["api_key"]
```

---

## ✅ CHECKLIST FINAL DE DEPLOYMENT

- [ ] ✅ Repositorio GitHub creado
- [ ] ✅ Código subido (sin datasets, venv, modelos)
- [ ] ✅ `.gitignore` configurado correctamente
- [ ] ✅ `requirements_streamlit.txt` completo
- [ ] ✅ Modelos en Google Drive públicos
- [ ] ✅ `download_models.py` con IDs correctos
- [ ] ✅ Streamlit Cloud app creada
- [ ] ✅ Deployment exitoso (sin errores)
- [ ] ✅ App funciona en URL pública
- [ ] ✅ Detección y segmentación operativos
- [ ] ✅ Parámetros estructurales calculan
- [ ] ✅ Caché funciona correctamente

---

## 🌐 COMPARTIR TU APP

Una vez desplegada, comparte la URL:

```
https://deteccion-fisuras.streamlit.app
```

**Usos**:
- 📄 Incluir en tesis/paper
- 🎓 Presentación de proyecto
- 👥 Compartir con profesores/evaluadores
- 💼 Portfolio profesional

---

## 📞 SOPORTE

### Documentación oficial:
- Streamlit: https://docs.streamlit.io/streamlit-cloud
- GitHub: https://docs.github.com/
- TensorFlow: https://www.tensorflow.org/

### Comunidad:
- Streamlit Forum: https://discuss.streamlit.io/
- Stack Overflow: Tag `streamlit`

---

**¡Listo para deployment!** 🚀

**Próximo paso**: Ejecuta los comandos de PASO 2.3 para subir a GitHub
