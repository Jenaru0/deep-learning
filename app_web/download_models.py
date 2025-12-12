"""
Script para descargar modelos desde Google Drive automáticamente
Se ejecuta al iniciar la app en Streamlit Cloud
"""

import os
import gdown
import streamlit as st

# IDs de los archivos en Google Drive
MODELS = {
    'deteccion': {
        'url': 'https://drive.google.com/uc?id=1toZrp6q8-qCrRk7DUkz12wH9DAMVv0_V',
        'output': 'modelos/deteccion/modelo_deteccion_final.keras'
    },
    'segmentacion': {
        'url': 'https://drive.google.com/uc?id=1Ug_h1flAfLNHLMyNTP2yb_4wdqE5HxIt',
        'output': 'modelos/segmentacion/unet_segmentacion_final.keras'
    }
}

def download_models():
    """Descarga modelos si no existen localmente"""
    
    for model_name, model_info in MODELS.items():
        output_path = model_info['output']
        
        # Verificar si ya existe
        if os.path.exists(output_path):
            st.info(f"✅ Modelo {model_name} ya existe localmente")
            continue
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Descargar desde Google Drive
        st.info(f"📥 Descargando modelo {model_name}...")
        try:
            gdown.download(model_info['url'], output_path, quiet=False)
            st.success(f"✅ Modelo {model_name} descargado exitosamente")
        except Exception as e:
            st.error(f"❌ Error descargando {model_name}: {str(e)}")
            st.stop()

if __name__ == "__main__":
    download_models()
