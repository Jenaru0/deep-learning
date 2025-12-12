# 📤 GUÍA: SUBIR MODELOS A GOOGLE DRIVE

## PASO 1: Subir modelos a Google Drive

1. **Ve a:** https://drive.google.com/
2. **Crea una carpeta:** "fisuras_modelos"
3. **Sube estos archivos:**
   - `modelos/deteccion/modelo_deteccion_final.keras`
   - `modelos/segmentacion/unet_segmentacion_final.keras`

## PASO 2: Hacer los archivos públicos

1. **Click derecho** en cada archivo → "Compartir"
2. **Cambiar acceso** a "Cualquiera con el enlace"
3. **Permisos:** "Lector"
4. **Copiar enlace** de cada archivo

## PASO 3: Obtener ID del archivo

Del enlace:

```
https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0/view?usp=sharing
                              ^^^^^^^^^^^^^^^^^^^^
                              Este es el FILE_ID
```

## PASO 4: Actualizar download_models.py

Edita `app_web/download_models.py` y reemplaza:

```python
MODELS = {
    'deteccion': {
        'url': 'https://drive.google.com/uc?id=TU_FILE_ID_DETECCION',  # ← Pega aquí
        'output': 'modelos/deteccion/modelo_deteccion_final.keras'
    },
    'segmentacion': {
        'url': 'https://drive.google.com/uc?id=TU_FILE_ID_SEGMENTACION',  # ← Pega aquí
        'output': 'modelos/segmentacion/unet_segmentacion_final.keras'
    }
}
```

## PASO 5: Verificar localmente

```bash
# En terminal:
python app_web/download_models.py

# Debe descargar los modelos automáticamente
```

## EJEMPLO COMPLETO:

Si tu enlace es:

```
https://drive.google.com/file/d/1xYz9AbC123DeF456/view?usp=sharing
```

Tu código quedará:

```python
'url': 'https://drive.google.com/uc?id=1xYz9AbC123DeF456',
```

---

## ✅ CHECKLIST:

- [ ] Subir `modelo_deteccion_final.keras` a Drive
- [ ] Subir `unet_segmentacion_final.keras` a Drive
- [ ] Hacer ambos archivos públicos
- [ ] Copiar FILE_IDs
- [ ] Actualizar `download_models.py`
- [ ] Probar descarga local
- [ ] Commit y push a GitHub
